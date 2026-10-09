import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from uhm.cli import app
from uhm.runtime.haptic_preview import envelope_at, preview_contacts, preview_events
from uhm.schemas import BodyForceEvent


def contact_report(action="foot_landing", states=("positive", "positive", "unknown", "positive")):
    return {
        "duration_s": 2, "mode": "local_vlm", "task": "visible_contact",
        "actions": [action], "contact_regions": {action: "feet"},
        "windows": [{"start_s": i * .5, "end_s": (i + 1) * .5, "status": "ok",
                     "labels": {action: {"state": state, "probability": None}}}
                    for i, state in enumerate(states)],
        "contact_events": [{"event_type": action, "contact_region": "feet",
                            "start_s": 0, "end_s": 2, "intensity": None}],
    }


def test_merge_before_rendering_and_unknown_gap_stays_silent():
    report = contact_report()
    original = copy.deepcopy(report)
    preview = preview_contacts(report)
    assert len(preview.cues) == 2
    first, second = preview.cues
    assert (first.start_ms, first.end_ms, first.source_end_ms) == (0, 220, 1000)
    assert second.start_ms == 1500
    assert first.perceptual_intensity is None
    assert first.target == "observed_subject"
    assert envelope_at(first, 22) == 1
    assert all(envelope_at(c, 1250) == 0 for c in preview.cues)
    assert report == original


def test_support_sustains_and_adjacent_pov_events_merge():
    report = contact_report("foot_support", ("positive",) * 4)
    report["contact_events"] = [dict(report["contact_events"][0], start_s=i * .5, end_s=(i + 1) * .5)
                                for i in range(4)]
    preview = preview_contacts(report)
    assert len(preview.cues) == 1
    cue = preview.cues[0]
    assert cue.effect == "sustain" and cue.end_ms == 2000
    assert len(cue.source_ids) == 4
    assert envelope_at(cue, 1900) == 1
    assert envelope_at(cue, 2000) == 0


def test_inference_is_opt_in_and_unknown_is_never_promoted():
    report = contact_report()
    report["task"] = "egocentric_motion"
    report["contact_events"][0]["basis"] = "inferred"
    assert not preview_contacts(report).cues
    cues = preview_contacts(report, include_inferred=True).cues
    assert len(cues) == 2
    assert all(c.target == "camera_wearer" and c.basis == "inferred" for c in cues)
    report["windows"][0]["labels"]["foot_landing"]["basis"] = "unknown"
    assert preview_contacts(report, include_inferred=True).cues[0].start_ms == 500


@pytest.mark.parametrize("change", ["dry_run", "unsupported", "error", "abstention", "region"])
def test_untrusted_or_unconfirmed_candidates_do_not_render(change):
    report = contact_report()
    if change == "dry_run":
        report["mode"] = "dry_run"
    elif change == "unsupported":
        report["task"] = "visible_actions"
    elif change == "error":
        for window in report["windows"]:
            window["status"] = "error"
    elif change == "abstention":
        for window in report["windows"]:
            window["labels"]["foot_landing"]["abstention_reason"] = "no_contact_evidence"
    else:
        report["contact_events"][0]["contact_region"] = "head"
    assert not preview_contacts(report, include_inferred=True).cues


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1])
def test_invalid_source_timing_is_rejected(value):
    report = contact_report()
    report["windows"][0]["start_s"] = value
    with pytest.raises(ValueError):
        preview_contacts(report)


def test_overlapping_windows_and_future_evidence_are_rejected():
    report = contact_report()
    report["windows"][1]["start_s"] = .25
    with pytest.raises(ValueError, match="overlap"):
        preview_contacts(report)
    report = contact_report()
    report["windows"][0]["labels"]["foot_landing"]["evidence_times_s"] = [.25, .75]
    with pytest.raises(ValueError, match="causal"):
        preview_contacts(report)


def body_event(**changes):
    return BodyForceEvent.model_validate({
        "event_id": "evt-1", "timestamp_ms": 100, "event_class": "impact",
        "recipient": "player", "body_region": "head", "intensity": .4, "duration_ms": 220,
        "confidence": {"event": .9, "recipient": .99, "region": .9, "intensity": 0}, **changes,
    })


def test_body_events_preserve_salience_and_identity_but_do_not_use_confidence_as_amplitude():
    event = body_event()
    preview = preview_events([event, event, body_event(event_id="npc", recipient="other_character"),
                              body_event(event_id="hidden", body_region="unknown")], 250)
    assert len(preview.cues) == 1
    cue = preview.cues[0]
    assert (cue.start_ms, cue.end_ms, cue.source_ids) == (100, 250, ["evt-1"])
    assert cue.perceptual_intensity == .4
    assert max(point.level for point in cue.envelope) == 1
    assert preview.omitted == {"duplicate_event": 1, "unlocalized_or_nonplayer": 2}
    with pytest.raises(ValueError, match="Conflicting"):
        preview_events([event, body_event(timestamp_ms=120)], 1000)


def test_seek_pause_and_short_clips_are_stateless_and_bounded():
    cue = preview_events([body_event()], 101).cues[0]
    times = [100.05, 500, 100.05, float("nan"), -1, 101]
    values = [envelope_at(cue, t) for t in times]
    assert values[0] == values[2]
    assert values[1] == values[3] == values[4] == values[5] == 0
    assert all(0 <= value <= 1 for value in values)


def test_cli_exports_without_overwriting_source(tmp_path):
    source, destination = tmp_path / "contacts.json", tmp_path / "preview.json"
    source.write_text(json.dumps(contact_report()), encoding="utf-8")
    content = source.read_bytes()
    runner = CliRunner()
    result = runner.invoke(app, ["preview-haptics", str(source), "--output", str(destination)])
    assert result.exit_code == 0, result.output
    assert json.loads(destination.read_text())["mode"] == "design_preview"
    result = runner.invoke(app, ["preview-haptics", str(source), "--output", str(source)])
    assert result.exit_code != 0
    assert source.read_bytes() == content


def test_endpoint_is_available_without_a_torch_model(tmp_path, monkeypatch):
    from uhm.runtime import actions_server

    video = tmp_path / "clip.mp4"
    monkeypatch.setattr(actions_server, "video_metadata", lambda path: {"duration_ms": 2000})
    report = contact_report()
    report["video"] = str(video)
    timeline = tmp_path / "timeline.json"
    timeline.write_text(json.dumps(report))
    client = TestClient(actions_server.create_actions_app(video, timeline))
    response = client.get("/api/haptic-preview")
    assert response.status_code == 200
    assert len(response.json()["cues"]) == 2
    assert response.json()["hardware_output"] is False
    assert client.get("/api/action-timeline").json() == report


def test_cli_accepts_existing_body_event_stream(tmp_path):
    source: Path = tmp_path / "events.jsonl"
    source.write_text(body_event().model_dump_json() + "\n")
    result = CliRunner().invoke(app, ["preview-haptics", str(source), "--source-kind", "events",
                                    "--duration-ms", "1000", "--output", str(tmp_path / "out.json")])
    assert result.exit_code == 0, result.output
