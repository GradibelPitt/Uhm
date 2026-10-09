"""Behavioral tests for the independent action-labeling path; no network or haptic model."""

import json
from io import BytesIO
from urllib.error import HTTPError

import cv2
import numpy as np
from fastapi.testclient import TestClient

from uhm.inference.decisions_actions import (
    anchors,
    classify_response,
    infer_actions,
    load_config,
    merge_segments,
    request_decision,
)
from uhm.runtime.actions_server import create_actions_app


def make_video(path, fps=4, frames=8):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((64, 64, 3), index * 25, dtype=np.uint8))
    writer.release()
    return path


def answer(**probabilities):
    probabilities = {**{f"observable_{name}": .95 for name in probabilities if name != "subject_visible"}, **probabilities}
    return {"answers": [{"type": "predicate", "name": name, "probability": probability}
                        for name, probability in probabilities.items()]}


def test_config_and_anchor_bounds():
    config = load_config("configs/decisions_actions.yaml")
    assert "walking" in config["actions"]
    assert list(anchors(1.2, 0.5)) == [(0.0, 0.5), (0.5, 1.0), (1.0, 1.2)]


def test_ordered_frames_multilabel_intervals_and_unknown_gap(tmp_path):
    video = make_video(tmp_path / "short.mp4")
    calls = []
    responses = [
        answer(subject_visible=.95, walking=.9, running=.2, jumping=.1, sitting=.1, pointing=.1, waving=.1),
        answer(subject_visible=.95, walking=.8, running=.7, jumping=.1, sitting=.1, pointing=.1, waving=.1),
        {"answers": [{"type": "refusal", "name": "subject_visible"}]},
        answer(subject_visible=.95, walking=.75, running=.1, jumping=.1, sitting=.1, pointing=.1, waving=.1),
    ]

    def requester(payload, timeout):
        calls.append(payload)
        return responses[len(calls) - 1]

    output = tmp_path / "actions.json"
    report = infer_actions(video, output, requester=requester)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert [(segment["label"], segment["start_s"], segment["end_s"]) for segment in report["segments"]] == [
        ("walking", 0.0, 1.0), ("running", 0.5, 1.0), ("walking", 1.5, 2.0),
    ]
    assert report["windows"][2]["status"] == "refusal"
    assert report["windows"][2]["labels"]["walking"]["state"] == "unknown"
    assert all(call["model"] == "gpt-6-luna" for call in calls)
    assert calls[0]["questions"][0]["name"] == "subject_visible"
    parts = calls[3]["input"][0]["content"]
    assert [part["type"] for part in parts].count("input_image") == 8
    assert all(part["image_url"].startswith("data:image/jpeg;base64,") for part in parts if part["type"] == "input_image")
    assert "[1.500, 2.000)" in parts[0]["text"]
    assert all(0 <= stamp <= 2 for stamp in report["windows"][3]["sample_times_s"])


def test_invalid_visibility_and_partial_refusal_are_not_negative():
    config = load_config("configs/decisions_actions.yaml")
    status, labels, _ = classify_response(answer(subject_visible=.4, walking=.99), config)
    assert status == "unknown_visibility"
    assert labels["walking"]["state"] == "unknown"
    status, labels, _ = classify_response({"answers": [
        {"type": "predicate", "name": "subject_visible", "probability": .9},
        {"type": "refusal", "name": "walking"},
        {"type": "predicate", "name": "observable_running", "probability": .9},
        {"type": "predicate", "name": "running", "probability": .1},
    ]}, config)
    assert status == "refusal"
    assert labels["walking"]["state"] == "unknown"
    assert labels["running"]["state"] == "negative"
    status, labels, _ = classify_response(answer(subject_visible=.9, walking=float("nan")), config)
    assert status == "partial" and labels["walking"]["state"] == "unknown"


def test_request_failures_remain_visible(tmp_path):
    video = make_video(tmp_path / "short.mp4", frames=2)

    def failing_requester(payload, timeout):
        raise RuntimeError("deliberate test failure")

    report = infer_actions(video, tmp_path / "failure.json", requester=failing_requester)
    assert report["segments"] == []
    assert report["windows"][0]["status"] == "error"
    assert "deliberate test failure" in report["windows"][0]["error"]


def test_merging_never_bridges_unknown_windows():
    windows = [
        {"start_s": 0, "end_s": .5, "labels": {"walk": {"state": "positive", "probability": .8}}},
        {"start_s": .5, "end_s": 1, "labels": {"walk": {"state": "unknown", "probability": None}}},
        {"start_s": 1, "end_s": 1.5, "labels": {"walk": {"state": "positive", "probability": .9}}},
    ]
    assert [(s["start_s"], s["end_s"]) for s in merge_segments(windows, ["walk"])] == [(0, .5), (1, 1.5)]

def test_visible_upper_body_without_legs_keeps_locomotion_unknown():
    config = load_config("configs/decisions_actions.yaml")
    response = answer(subject_visible=.99, walking=.1, running=.1, waving=.9)
    for item in response["answers"]:
        if item["name"] in {"observable_walking", "observable_running"}:
            item["probability"] = .2
    status, labels, _ = classify_response(response, config)
    assert status == "partial"
    assert labels["walking"]["state"] == "unknown"
    assert labels["running"]["state"] == "unknown"
    assert labels["waving"]["state"] == "positive"


def test_dry_run_decodes_frames_and_standalone_viewer_serves_them(tmp_path):
    video = make_video(tmp_path / "short.mp4")
    timeline = tmp_path / "dry_run.json"
    report = infer_actions(video, timeline, dry_run=True)
    assert report["mode"] == "dry_run"
    assert all(window["status"] == "not_run" for window in report["windows"])
    assert all(window["labels"]["walking"]["state"] == "unknown" for window in report["windows"])
    assert report["segments"] == []
    app = create_actions_app(video, timeline)
    with TestClient(app) as client:
        assert "DRY RUN" in client.get("/").text
        assert client.get("/api/meta").json()["haptic_model_loaded"] is False
        assert client.get("/api/action-timeline").json() == report
        response = client.get("/api/video", headers={"Range": "bytes=0-31"})
        assert response.status_code == 206
        assert response.content == video.read_bytes()[:32]
def test_probability_middle_band_and_duplicate_answers_abstain():
    config = load_config("configs/decisions_actions.yaml")
    response = answer(subject_visible=.99, walking=.59, running=.2, waving=.6)
    status, labels, _ = classify_response(response, config)
    assert status == "partial"
    assert labels["walking"] == {"state": "unknown", "probability": .59}
    assert labels["running"]["state"] == "negative"
    assert labels["waving"]["state"] == "positive"
    response["answers"].append({"type": "predicate", "name": "waving", "probability": .99})
    status, labels, _ = classify_response(response, config)
    assert status == "partial"
    assert labels["waving"]["state"] == "unknown"
    response["answers"].append({"type": "predicate", "name": "subject_visible", "probability": .99})
    status, labels, _ = classify_response(response, config)
    assert status == "invalid_response"
    assert all(label["state"] == "unknown" for label in labels.values())


def test_low_fps_and_one_frame_never_create_motion_labels(tmp_path):
    video = make_video(tmp_path / "single.mp4", fps=1, frames=1)
    calls = []

    def requester(payload, timeout):
        calls.append(payload)
        return answer(subject_visible=1, walking=1)

    report = infer_actions(video, tmp_path / "single.json", requester=requester)
    assert calls == []
    assert report["segments"] == []
    assert all(window["status"] == "insufficient_frames" for window in report["windows"])
    assert all(sample["frame_index"] == 0 and sample["cfr_time_s"] == 0
               for window in report["windows"] for sample in window["frame_samples"])
    assert report["timestamp_basis"].startswith("CFR estimate")

    video = make_video(tmp_path / "low_fps.mp4", fps=2, frames=3)
    report = infer_actions(video, tmp_path / "low_fps.json", dry_run=True)
    last = report["windows"][-1]
    assert {sample["frame_index"] for sample in last["frame_samples"]} == {0, 1, 2}
    assert last["sample_times_s"] == [0.0, .5, 1.0]
    assert last["frame_samples"][-1]["cfr_time_s"] == 1.0
    assert last["max_sampling_error_s"] > 0


def test_api_billing_error_preserves_code_without_raw_message(monkeypatch):
    from uhm.inference import decisions_actions

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")

    def exhausted(*args, **kwargs):
        raise HTTPError("https://api.openai.com/v1/decisions", 429, "quota", {}, BytesIO(
            b'{"error":{"code":"credit_balance_exhausted","message":"private details"}}'
        ))

    monkeypatch.setattr(decisions_actions.request, "urlopen", exhausted)
    try:
        request_decision({}, 1)
    except RuntimeError as exc:
        assert str(exc) == "Decisions API HTTP 429 (credit_balance_exhausted)"
    else:
        raise AssertionError("Expected an API billing error")
