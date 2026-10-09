"""Standalone action-timeline viewer; deliberately independent of the haptic model."""

import hashlib
import json
import math
import mimetypes
import statistics
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from uhm.inference.decisions_actions import video_metadata
from uhm.ontology import ROOT
from uhm.runtime.haptic_preview import HapticPreview, preview_contacts


def playback_summary(report: dict) -> dict:
    """Report measured offline performance without implying live force inference."""
    latencies = sorted(window["latency_s"] for window in report.get("windows", [])
                       if isinstance(window.get("latency_s"), (float, int))
                       and math.isfinite(window["latency_s"]) and window["latency_s"] >= 0)
    step = report.get("step_s", 0)
    return {
        "mode": "synchronized_replay",
        "model": report.get("model", "unknown"),
        "provider": report.get("provider", "unknown"),
        "window_count": len(report.get("windows", [])),
        "analysis_wall_s": report.get("analysis_wall_s"),
        "request_p50_s": statistics.median(latencies) if latencies else None,
        "request_p95_s": latencies[math.ceil(len(latencies) * .95) - 1] if latencies else None,
        "request_to_step_ratio": statistics.median(latencies) / step if latencies and step > 0 else None,
        "physical_force_measured": False,
        "anatomical_side_known": False,
        "independently_validated": False,
    }


def create_actions_app(video: Path, timeline: Path) -> FastAPI:
    video = Path(video).resolve()
    metadata = video_metadata(video)
    report = json.loads(Path(timeline).read_text(encoding="utf-8"))
    if not isinstance(report, dict) or Path(report.get("video", "")).resolve() != video:
        raise ValueError("Action timeline does not match the configured video")
    if report.get("video_sha256"):
        with video.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != report["video_sha256"]:
                raise ValueError("Video content has changed since analysis")
    if abs(report.get("duration_s", -1) - metadata["duration_ms"] / 1000) > .1:
        raise ValueError("Action timeline duration does not match the configured video")
    for window in report.get("windows", []):
        if not 0 <= window["start_s"] < window["end_s"] <= report["duration_s"] + .001:
            raise ValueError("Action window is outside the video")
    for event in report.get("contact_events", []):
        if not 0 <= event["start_s"] < event["end_s"] <= report["duration_s"] + .001:
            raise ValueError("Contact event is outside the video")
    app = FastAPI(title="UHM action timeline")

    @app.get("/")
    def viewer():
        studio = ROOT / "frontend/dist/studio.html"
        return FileResponse(studio if studio.exists() and report.get("task") in {"visible_contact", "egocentric_motion"}
                            else ROOT / "frontend/action_viewer.html",
                            media_type="text/html")

    @app.get("/api/video")
    def source_video():
        return FileResponse(video, media_type=mimetypes.guess_type(video.name)[0] or "application/octet-stream", filename=video.name,
                            content_disposition_type="inline")

    @app.get("/api/action-timeline")
    def action_timeline():
        return JSONResponse(report)

    @app.get("/api/haptic-preview", response_model=HapticPreview)
    def haptic_preview(include_inferred: bool = False):
        try:
            return preview_contacts(report, include_inferred=include_inferred)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/meta")
    def video_info():
        sources_path = ROOT / "docs/action-video-sources.json"
        sources = json.loads(sources_path.read_text("utf-8")) if sources_path.exists() else []
        source = next((item for item in sources if item["id"] == video.stem), None) if (
            video.parent == (ROOT / "data/raw/action_validation/mp4").resolve()
        ) else None
        if video.name == "prague_demo_4k.mp4" and video.parent == (ROOT / "data/raw/pov").resolve():
            source = {"source": "https://www.youtube.com/watch?v=U5yHk3hY2MU",
                      "attribution": "Prague Parkour POV · AMPISOUND · 原片 00:02–02:00 · 4K / 30 FPS"}
        return {"video_name": video.name, "video_url": "/api/video", "video": metadata,
                "mode": report.get("mode", "unknown"), "haptic_model_loaded": False,
                "playback": playback_summary(report), "source": source}

    assets = ROOT / "frontend/dist/assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    return app
