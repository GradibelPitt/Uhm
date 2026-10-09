"""Local playback-synchronized inference service. The UI sends time, never labels."""

import asyncio
import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from uhm.data.video import video_metadata
from uhm.inference.predictor import ModelPredictor
from uhm.inference.profiling import Profiler
from uhm.ontology import BODY_ONTOLOGY, ROOT
from uhm.schemas import SCHEMA_VERSION, BodyForceEvent

from .aggregation import EventAggregator, load_runtime_config


class SampleMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["sample"]
    timestamp_ms: int = Field(ge=0)
    request_id: int = Field(ge=0)


class ResetMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["reset"]
    request_id: int = Field(ge=0)


def create_app(
    video: Path, checkpoint: Path, device: str = "cpu",
    runtime_config: Path | None = None, log_directory: Path | None = None,
    *, predictor=None, actions_path: Path | None = None,
) -> FastAPI:
    video = Path(video).resolve()
    builtin_demo = video == (ROOT / "data/raw/demo.mp4").resolve()
    video_info = video_metadata(video)
    predictor = predictor or ModelPredictor(checkpoint, device)
    config = load_runtime_config(runtime_config)
    logs = Path(log_directory or ROOT / "runs")
    logs.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Universal Haptic Model", version=SCHEMA_VERSION)
    app.state.predictor = predictor
    action_timeline = None
    if actions_path is not None:
        action_timeline = json.loads(Path(actions_path).read_text(encoding="utf-8"))
        if not isinstance(action_timeline, dict) or Path(action_timeline.get("video", "")).resolve() != video:
            raise ValueError("Action timeline does not match the configured video")
    app.state.action_timeline = action_timeline
    # Model is shared and stateless across calls; aggregation and logs are per socket.
    model_lock = asyncio.Lock()

    @app.get("/healthz")
    def health():
        return {"status": "ready", "model_loaded": True}

    @app.get("/api/meta")
    def metadata():
        return {
            "schema_version": SCHEMA_VERSION, "video_url": "/api/video",
            "video_name": video.name,
            "video": video_info | {
                "synthetic": builtin_demo,
                "viewpoint": "third_person" if builtin_demo else "unknown",
            },
            "model": {
                "name": f"MobileNetV3 + {predictor.config['model']['temporal'].upper()}",
                "checkpoint": Path(checkpoint).name, "device": device,
                "synthetic": bool(predictor.metadata.get("synthetic", False)),
                "supported_classes": predictor.metadata.get("provenance", {}).get("supported_event_classes", []),
                "confidence_calibrated": False,
            },
            "sample_step_ms": 125, "context_window_ms": predictor.data_config["window_ms"],
            "runtime": asdict(config),
        }

    @app.get("/api/action-timeline")
    def actions():
        if app.state.action_timeline is None:
            return JSONResponse({"message": "No action timeline loaded"}, status_code=404)
        return app.state.action_timeline
    @app.get("/api/video")
    def source_video():
        # Starlette FileResponse supports byte ranges, necessary for native video seeking.
        return FileResponse(video, media_type="video/mp4", filename=video.name,
                            content_disposition_type="inline")

    @app.get("/api/body-regions")
    def body_regions():
        return BODY_ONTOLOGY

    @app.get("/api/schema")
    def event_schema():
        return BodyForceEvent.model_json_schema()

    @app.websocket("/ws/inference")
    async def inference(socket: WebSocket):
        origin = socket.headers.get("origin")
        if origin and urlparse(origin).hostname not in {"127.0.0.1", "localhost", "::1"}:
            await socket.close(code=1008)
            return
        await socket.accept()
        session_id = uuid.uuid4().hex
        event_path = logs / f"session-{session_id}.events.jsonl"
        diagnostic_path = logs / f"session-{session_id}.predictions.jsonl"
        profile_path = logs / f"session-{session_id}.profile.json"
        aggregator = EventAggregator(config)
        profiler = Profiler(device, predictor.data_config["num_frames"])
        count = 0
        await socket.send_json({"type": "ready", "session_id": session_id,
                                "log_path": str(event_path)})
        try:
            with event_path.open("w", encoding="utf-8") as event_log, diagnostic_path.open("w", encoding="utf-8") as diagnostic_log:
                while True:
                    raw = await socket.receive_json()
                    request_id = raw.get("request_id") if isinstance(raw, dict) else None
                    try:
                        if isinstance(raw, dict) and raw.get("type") == "reset":
                            message = ResetMessage.model_validate(raw)
                            aggregator.reset()
                            diagnostic_log.write(json.dumps({"type": "reset", "request_id": message.request_id}) + "\n")
                            diagnostic_log.flush()
                            await socket.send_json({"type": "reset", "request_id": message.request_id})
                            continue
                        message = SampleMessage.model_validate(raw)
                        if message.timestamp_ms >= video_info["duration_ms"]:
                            raise ValueError("Sample time must be inside the video duration")
                    except (ValidationError, ValueError) as error:
                        await socket.send_json({"type": "error", "request_id": request_id, "message": str(error)})
                        continue
                    async with model_lock:
                        prediction, timing = await asyncio.to_thread(predictor.predict_video, video, message.timestamp_ms)
                    event = aggregator.update(prediction)
                    profiler.record(timing, prediction)
                    if event:
                        event_log.write(event.model_dump_json() + "\n")
                        event_log.flush()
                        print(event.model_dump_json(), flush=True)
                        count += 1
                    diagnostics = prediction.model_dump(mode="json") | timing
                    diagnostic_log.write(json.dumps({"type": "prediction", "request_id": message.request_id,
                                                     "diagnostics": diagnostics}) + "\n")
                    diagnostic_log.flush()
                    await socket.send_json({
                        "type": "prediction", "timestamp_ms": message.timestamp_ms,
                        "request_id": message.request_id, "event": event.model_dump(mode="json") if event else None,
                        "diagnostics": diagnostics, "stats": profiler.summary() | {"events": count},
                    })
        except WebSocketDisconnect:
            pass
        except (ValueError, RuntimeError) as error:
            # Decoding/model errors stay visible and never substitute a mock event.
            try:
                await socket.send_json({"type": "error", "message": str(error)})
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            profile_path.write_text(json.dumps(profiler.summary() | {
                "events": count, "session_id": session_id, "video": str(video),
                "checkpoint": str(checkpoint), "runtime": asdict(config),
            }, indent=2) + "\n", encoding="utf-8")

    distribution = ROOT / "frontend/dist"
    if (distribution / "index.html").exists():
        app.mount("/", StaticFiles(directory=distribution, html=True), name="frontend")
    else:
        @app.get("/")
        def missing_build():
            return JSONResponse({"message": "Build the UI with npm run build in frontend, or run uv run uhm demo."}, status_code=503)
    return app
