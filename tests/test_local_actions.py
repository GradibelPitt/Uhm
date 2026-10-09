"""Local visual decisions preserve unknowns without inventing numeric confidence."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from uhm.inference.local_actions import (
    build_local_request,
    infer_local_actions,
    load_local_config,
    parse_local_response,
)


def response(actions, subject="clear"):
    return {"message": {"content": json.dumps({"subject": subject, "actions": actions})}}


def test_visibility_and_missing_actions_abstain():
    status, labels = parse_local_response(response({
        "walking": {"state": "negative", "observable": "no"},
        "waving": {"state": "positive", "observable": "yes"},
    }), ["walking", "waving", "jumping"])
    assert status == "partial"
    assert labels["walking"]["state"] == "unknown"
    assert labels["jumping"]["state"] == "unknown"
    assert labels["waving"] == {"state": "positive", "probability": None}
    _, ambiguous = parse_local_response(response({
        "waving": {"state": "positive", "observable": "yes"},
    }, subject="ambiguous"), ["waving"])
    assert ambiguous["waving"]["state"] == "unknown"


def test_duplicate_and_truncated_json_is_unknown():
    duplicate = {"message": {"content": '{"subject":"clear","subject":"none","actions":{}}'}}
    assert parse_local_response(duplicate, ["walking"])[0] == "invalid_response"
    truncated = response({"walking": {"state": "positive", "observable": "yes"}})
    truncated["done_reason"] = "length"
    assert parse_local_response(truncated, ["walking"])[0] == "invalid_response"


@pytest.mark.parametrize("indices", [None, [], [0], [1, 0], [0, 0], [-1, 2], [0, 3], [True, 2]])
def test_transition_requires_two_distinct_ordered_existing_frames(indices):
    status, labels = parse_local_response(response({
        "foot_landing": {"state": "positive", "observable": "yes", "evidence_frames": indices},
    }), ["foot_landing"], transition_actions=["foot_landing"], frame_times=[0, .25, .5],
        anchor_start=.5, anchor_end=1)
    assert status == "partial"
    assert labels["foot_landing"]["state"] == "unknown"


def test_earlier_context_must_not_trigger_current_impact():
    item = {"state": "positive", "observable": "yes", "evidence_frames": [0, 1]}
    kwargs = {"transition_actions": ["torso_impact"], "frame_times": [.25, .5, .75],
              "anchor_start": .75, "anchor_end": 1}
    _, labels = parse_local_response(response({"torso_impact": item}), ["torso_impact"], **kwargs)
    assert labels["torso_impact"]["state"] == "unknown"
    item["evidence_frames"] = [1, 2]
    _, labels = parse_local_response(response({"torso_impact": item}), ["torso_impact"], **kwargs)
    assert labels["torso_impact"]["evidence_times_s"] == [.5, .75]
    assert labels["torso_impact"]["state"] == "positive"


def test_indexed_contact_sheet_is_decodable_and_preserves_frame_order():
    import base64

    from uhm.inference.local_actions import contact_sheet

    frames = []
    for index, color in enumerate([30, 200]):
        _, encoded = cv2.imencode(".jpg", np.full((64, 64, 3), color, dtype=np.uint8))
        frames.append((index * .5, "data:image/jpeg;base64," + base64.b64encode(encoded).decode()))
    result = cv2.imdecode(np.frombuffer(base64.b64decode(contact_sheet(frames, .5)), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert result.shape == (268, 640, 3)
    assert result[150, 160].mean() < 50
    assert result[150, 480].mean() > 180


def test_playback_summary_does_not_claim_realtime_or_force():
    from uhm.runtime.actions_server import playback_summary

    summary = playback_summary({"step_s": .5, "windows": [{"latency_s": 4}, {"latency_s": 6}]})
    assert summary["request_p50_s"] == 5
    assert summary["request_to_step_ratio"] == 10
    assert summary["mode"] == "synchronized_replay"
    assert summary["physical_force_measured"] is False
    assert summary["independently_validated"] is False
    assert playback_summary({})["request_p50_s"] is None


def test_phase_verification_can_only_remove_candidates():
    import base64

    from uhm.inference.local_actions import verify_jump_phase

    _, image = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    frames = [(0.1, "data:image/jpeg;base64," + base64.b64encode(image).decode())]
    window = {"start_s": 0, "end_s": .5, "status": "ok", "labels": {
        "foot_support": {"state": "positive", "probability": None},
        "torso_impact": {"state": "negative", "probability": None},
    }}
    requester = lambda payload, timeout: {"message": {"content": json.dumps({
        "phase": "torso_landing", "feet_touch_surface": "no", "torso_touch_surface": "yes",
    })}}
    verify_jump_phase(load_local_config(Path("configs/jump_contacts.yaml")), frames, window, requester)
    assert window["labels"]["foot_support"]["state"] == "unknown"
    assert window["labels"]["torso_impact"]["state"] == "negative"


def test_video_replacement_invalidates_cached_contacts(tmp_path):
    import hashlib

    from uhm.inference.decisions_actions import video_metadata
    from uhm.runtime.actions_server import create_actions_app

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 4, (64, 64))
    writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    report = {"video": str(video.resolve()), "duration_s": video_metadata(video)["duration_ms"] / 1000,
              "video_sha256": hashlib.sha256(b"different video").hexdigest()}
    timeline = tmp_path / "contacts.json"
    timeline.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="content has changed"):
        create_actions_app(video, timeline)


def test_local_request_contains_ordered_images_and_no_key():
    config = load_local_config(Path("configs/local_contacts.yaml"))
    payload = build_local_request(config, [(0, "data:image/jpeg;base64,AAA"),
                                          (.25, "data:image/jpeg;base64,BBB")], 0, .5)
    assert payload["messages"][0]["images"] == ["AAA", "BBB"]
    assert "[0, 0.25]" in payload["messages"][0]["content"]
    assert payload["format"]["properties"]["actions"]["required"] == list(config["actions"])
    assert "Authorization" not in json.dumps(payload)
    assert config["contact_regions"]["foot_push_off"] == "feet"
    assert config["contact_regions"]["hand_load"] == "hands_and_arms"
    assert config["contact_regions"]["torso_impact"] == "torso"
    assert "sports" in payload["messages"][0]["content"]


def test_local_pipeline_merges_categorical_labels_without_probability(tmp_path):
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 4, (64, 64))
    assert writer.isOpened()
    for value in range(4):
        writer.write(np.full((64, 64, 3), value * 40, dtype=np.uint8))
    writer.release()

    def requester(payload, timeout):
        actions = {name: {"state": "negative", "observable": "yes"}
                   for name in payload["format"]["properties"]["actions"]["required"]}
        actions["foot_support"]["state"] = "positive"
        return response(actions)

    report = infer_local_actions(video, tmp_path / "local.json", requester=requester)
    assert report["mode"] == "local_vlm"
    assert report["segments"] == [{"label": "foot_support", "start_s": 0.0, "end_s": 1.0,
                                   "probability": None}]
    assert all(window["labels"]["foot_support"]["probability"] is None for window in report["windows"])
    assert report["contact_events"] == [{
        "event_type": "foot_support", "contact_region": "feet",
        "start_s": 0.0, "end_s": 1.0, "intensity": None, "probability": None,
    }]
