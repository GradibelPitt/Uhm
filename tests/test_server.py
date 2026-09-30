"""Transport tests with an explicitly injected deterministic test predictor."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from uhm.data.video import write_video
from uhm.runtime import server as server_module
from uhm.runtime.server import create_app
from uhm.schemas import BodyForceEvent, Confidence, Prediction


class DeterministicTestPredictor:
    """Protocol fixture only; never selected by the CLI or production runtime."""

    def __init__(self):
        self.config = {"model": {"temporal": "gru"}}
        self.data_config = {"num_frames": 6, "window_ms": 1000}
        self.metadata = {"synthetic": True, "provenance": {"supported_event_classes": ["impact"]}}
        self.calls = []

    def predict_video(self, video, timestamp_ms):
        self.calls.append(timestamp_ms)
        return Prediction(
            timestamp_ms=timestamp_ms, event_class="impact", recipient="player",
            body_region="left_torso", intensity=0.7, observable_probability=0.95,
            confidence=Confidence(event=0.95, recipient=0.95, region=0.9, intensity=0.0),
        ), {"model_ms": 1.0, "sampling_ms": 1.0, "latency_ms": 2.0}


@pytest.fixture
def server(tmp_path):
    video = write_video(tmp_path / "silent-fixture.mp4",
                        [np.full((64, 64, 3), index * 10, dtype=np.uint8) for index in range(12)], 12)
    predictor = DeterministicTestPredictor()
    log_directory = tmp_path / "logs"
    app = create_app(video, tmp_path / "injected-test-fixture.pt",
                     log_directory=log_directory, predictor=predictor)
    return app, predictor, video, log_directory


def _sample(socket, timestamp, request_id):
    socket.send_json({"type": "sample", "timestamp_ms": timestamp, "request_id": request_id})
    return socket.receive_json()


def test_health_metadata_schema_and_video_ranges(server):
    app, _, video, _ = server
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ready", "model_loaded": True}
        metadata = client.get("/api/meta").json()
        assert metadata["video"]["duration_ms"] == 1000
        assert metadata["video"]["viewpoint"] == "unknown"
        assert metadata["video"]["synthetic"] is False
        assert metadata["model"]["synthetic"] is True
        assert metadata["model"]["supported_classes"] == ["impact"]
        assert "body_region" in client.get("/api/schema").json()["properties"]
        assert client.get("/api/body-regions").json()["regions"]
        response = client.get("/api/video", headers={"Range": "bytes=0-31"})
        assert response.status_code == 206
        assert response.content == video.read_bytes()[:32]
        assert response.headers["content-range"].startswith("bytes 0-31/")


@pytest.mark.parametrize("relative_video, viewpoint", [
    ("data/raw/demo.mp4", "third_person"),
    ("custom/demo.mp4", "unknown"),
])
def test_only_the_builtin_demo_has_a_known_viewpoint(tmp_path, monkeypatch, relative_video, viewpoint):
    monkeypatch.setattr(server_module, "ROOT", tmp_path)
    video = write_video(tmp_path / relative_video, [np.zeros((64, 64, 3), dtype=np.uint8)], 12)
    app = create_app(video, tmp_path / "fixture.pt", predictor=DeterministicTestPredictor())
    with TestClient(app) as client:
        metadata = client.get("/api/meta").json()
        assert metadata["video"]["viewpoint"] == viewpoint
        assert metadata["video"]["synthetic"] is (viewpoint == "third_person")


def test_websocket_validation_dedup_reset_and_canonical_log(server):
    app, predictor, _, logs = server
    with TestClient(app) as client, client.websocket_connect("/ws/inference") as socket:
        ready = socket.receive_json()
        assert ready["type"] == "ready"
        assert Path(ready["log_path"]).parent == logs
        invalid = [
            {"type": "sample", "timestamp_ms": -1, "request_id": 1},
            {"type": "sample", "timestamp_ms": 1000, "request_id": 2},
            {"type": "sample", "timestamp_ms": "100", "request_id": 3},
            {"type": "sample", "timestamp_ms": 100, "request_id": 4, "body_region": "head"},
            {"type": "reset", "request_id": -1},
        ]
        for message in invalid:
            socket.send_json(message)
            response = socket.receive_json()
            assert response["type"] == "error"
            assert response["request_id"] == message["request_id"]
        assert predictor.calls == []
        first = _sample(socket, 100, 11)
        assert first["request_id"] == 11 and first["event"] is None
        event = _sample(socket, 225, 12)["event"]
        BodyForceEvent.model_validate(event)
        assert event["body_region"] == "left_torso"
        assert _sample(socket, 225, 13)["event"] is None
        assert _sample(socket, 350, 14)["event"] is None
        socket.send_json({"type": "reset", "request_id": 500})
        assert socket.receive_json() == {"type": "reset", "request_id": 500}
        assert _sample(socket, 100, 501)["event"] is None
        after_reset = _sample(socket, 225, 502)["event"]
        assert after_reset["event_id"] != event["event_id"]
    records = [BodyForceEvent.model_validate_json(line)
               for path in logs.glob("*.events.jsonl") for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert all(record.recipient.value == "player" for record in records)
    profiles = [json.loads(path.read_text()) for path in logs.glob("*.profile.json")]
    assert len(profiles) == 1 and profiles[0]["events"] == 2


def test_websocket_clients_keep_separate_confirmation_and_unique_ids(server):
    app, _, _, _ = server
    with TestClient(app) as client, client.websocket_connect("/ws/inference") as first:
        ready_first = first.receive_json()
        with client.websocket_connect("/ws/inference") as second:
            ready_second = second.receive_json()
            assert ready_first["session_id"] != ready_second["session_id"]
            assert _sample(first, 100, 1)["event"] is None
            assert _sample(second, 225, 1)["event"] is None
            first_event = _sample(first, 225, 2)["event"]
            second_event = _sample(second, 350, 2)["event"]
            assert first_event and second_event
            assert first_event["event_id"] != second_event["event_id"]


def test_websocket_rejects_nonlocal_browser_origin(server):
    app, _, _, _ = server
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as error, client.websocket_connect(
            "/ws/inference", headers={"origin": "https://untrusted.example"}
        ):
            pass
        assert error.value.code == 1008
