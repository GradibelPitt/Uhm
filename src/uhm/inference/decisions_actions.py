"""Offline, reviewable video action labels from ordered sampled frames."""

import base64
import json
import math
import os
import re
from pathlib import Path
from urllib import error, request

import cv2
import yaml

from uhm.ontology import ROOT

API_URL = "https://api.openai.com/v1/decisions"
NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def video_metadata(path: Path) -> dict:
    """Read CFR video metadata without importing the haptic training stack."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError(f"Video has invalid frame count or FPS: {path}")
        return {"fps": fps, "frame_count": count, "duration_ms": round(count * 1000 / fps),
                "width": round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        capture.release()

def load_config(path: Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("actions"), dict) or not config["actions"]:
        raise ValueError("Action config needs a nonempty actions mapping")
    if config.get("model") != "gpt-6-luna":
        raise ValueError("This backend expects model gpt-6-luna")
    for key in ("window_s", "step_s", "sample_fps", "request_timeout_s"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a positive finite number")
    if config["step_s"] > config["window_s"]:
        raise ValueError("step_s must not exceed window_s")
    if math.ceil(config["window_s"] * config["sample_fps"]) + 1 > 128:
        raise ValueError("A window may not contain more than 128 sampled images")
    for key in ("negative_threshold", "threshold", "visibility_threshold"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if config["negative_threshold"] >= config["threshold"]:
        raise ValueError("negative_threshold must be less than threshold")
    for key, low, high in (("max_image_edge", 64, 2048), ("jpeg_quality", 1, 100)):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key} must be an integer in [{low}, {high}]")
    for name, description in config["actions"].items():
        if not isinstance(name, str) or not NAME.fullmatch(name) or name == "subject_visible" or name.startswith("observable_"):
            raise ValueError(f"Invalid action name: {name!r}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Action {name} needs a description")
    return config


def anchors(duration_s: float, step_s: float):
    """Non-overlapping output spans. Context may overlap, but labels never do."""
    count = math.ceil((duration_s - 1e-9) / step_s)
    for index in range(count):
        start = round(index * step_s, 6)
        yield start, round(min(duration_s, (index + 1) * step_s), 6)


def sample_times(context_start: float, context_end: float, fps: float) -> list[float]:
    count = max(1, math.ceil((context_end - context_start) * fps))
    # Center each sample in its temporal bin, with the last bin inside the anchor.
    return [round(context_start + (index + 0.5) * (context_end - context_start) / count, 6)
            for index in range(count)]


def encode_frames(capture, times: list[float], source_fps: float, frame_count: int, edge: int,
                  quality: int) -> tuple[list[tuple[float, str]], list[dict]]:
    """Return unique decoded frames and their CFR frame-index timing evidence."""
    frames = []
    samples = []
    seen = set()
    for requested_s in times:
        frame_index = min(frame_count - 1, max(0, math.floor(requested_s * source_fps + 1e-9)))
        cfr_time_s = round(frame_index / source_fps, 6)
        samples.append({"requested_s": requested_s, "frame_index": frame_index,
                        "cfr_time_s": cfr_time_s,
                        "sampling_error_s": round(requested_s - cfr_time_s, 6)})
        if frame_index in seen:
            continue
        seen.add(frame_index)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Cannot decode frame {frame_index}")
        height, width = frame.shape[:2]
        scale = min(1.0, edge / max(height, width))
        if scale < 1:
            frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                               interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError(f"Cannot encode frame {frame_index}")
        frames.append((cfr_time_s, "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")))
    return frames, samples


def decision_request(config: dict, frames: list[tuple[float, str]], anchor_start: float, anchor_end: float) -> dict:
    content = [{"type": "input_text", "text": (
        "These ordered images are frames from one video. Their timestamps are source-video seconds. "
        f"Classify only the primary visible person's actions during [{anchor_start:.3f}, {anchor_end:.3f}) seconds. "
        "Earlier frames provide context only. Judge visible motion, not identity, intent or inferred events. "
        "If the person or motion is unclear, use low probability."
    )}]
    for timestamp, image in frames:
        content.append({"type": "input_text", "text": f"Frame at estimated {timestamp:.3f} seconds (frame index / source FPS):"})
        content.append({"type": "input_image", "image_url": image})
    questions = [{"type": "predicate", "name": "subject_visible", "instructions":
                  "One clear primary person is visible in the anchor interval; multiple equally prominent people are ambiguous."}]
    for name, description in config["actions"].items():
        observation = ("The legs and lower-body motion are visible enough to distinguish steps in the anchor interval."
                       if name in {"walking", "running"} else
                       f"The body parts and motion needed to judge {name} are visible enough in the anchor interval.")
        questions.append({"type": "predicate", "name": f"observable_{name}", "instructions": observation})
        questions.append({"type": "predicate", "name": name, "instructions":
                          f"During the anchor interval, is this visible action occurring? {description}"})
    return {"model": config["model"], "input": [{"role": "user", "content": content}], "questions": questions}


def request_decision(payload: dict, timeout_s: float) -> dict:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for action inference")
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(API_URL, body, {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            return json.load(response)
    except error.HTTPError as exc:
        # Preserve diagnostic codes, never the raw response or its message.
        diagnostic = ""
        try:
            api_error = json.load(exc).get("error", {})
            code = api_error.get("code") or api_error.get("type")
            if isinstance(code, str) and re.fullmatch(r"[a-zA-Z0-9_]{1,80}", code):
                diagnostic = f" ({code})"
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(f"Decisions API HTTP {exc.code}{diagnostic}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Decisions API network error: {exc.reason}") from exc


def classify_response(response: dict, config: dict) -> tuple[str, dict[str, dict], dict]:
    labels = {name: {"state": "unknown", "probability": None} for name in config["actions"]}
    if not isinstance(response, dict) or response.get("refusal") or response.get("status") == "refused":
        return "refusal", labels, {}
    answer_list = response.get("answers")
    if not isinstance(answer_list, list):
        return "invalid_response", labels, {}
    usage = response.get("usage") or {}
    known_names = {"subject_visible", *config["actions"],
                   *(f"observable_{name}" for name in config["actions"])}
    seen = set()
    duplicated = set()
    refusals = set()
    probabilities = {}
    for answer in answer_list:
        if not isinstance(answer, dict) or answer.get("name") not in known_names:
            continue
        name = answer["name"]
        if name in seen:
            duplicated.add(name)
            probabilities.pop(name, None)
            continue
        seen.add(name)
        if answer.get("type") == "refusal":
            refusals.add(name)
            continue
        if answer.get("type") != "predicate":
            continue
        value = answer.get("probability")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            continue
        probabilities[name] = float(value)
    for name in duplicated:
        probabilities.pop(name, None)
    if "subject_visible" in refusals:
        return "refusal", labels, usage
    if "subject_visible" in duplicated:
        return "invalid_response", labels, usage
    if probabilities.get("subject_visible", 0) < config["visibility_threshold"]:
        return "unknown_visibility", labels, usage
    for name in labels:
        if name in refusals or f"observable_{name}" in refusals:
            continue
        if probabilities.get(f"observable_{name}", 0) < config["visibility_threshold"]:
            continue
        probability = probabilities.get(name)
        if probability is None:
            continue
        state = ("positive" if probability >= config["threshold"] else
                 "negative" if probability <= config["negative_threshold"] else "unknown")
        labels[name] = {"state": state, "probability": probability}
    status = "ok" if all(label["state"] != "unknown" for label in labels.values()) else "refusal" if refusals else "partial"
    return status, labels, usage


def merge_segments(windows: list[dict], action_names: list[str]) -> list[dict]:
    segments = []
    def score(values):
        return round(sum(values) / len(values), 4) if values else None

    for name in action_names:
        active = None
        scores = []
        for window in windows:
            label = window["labels"][name]
            if label["state"] == "positive":
                if active is not None and abs(active["end_s"] - window["start_s"]) < 1e-6:
                    active["end_s"] = window["end_s"]
                else:
                    if active is not None:
                        active["probability"] = score(scores)
                        segments.append(active)
                    active = {"label": name, "start_s": window["start_s"], "end_s": window["end_s"]}
                    scores = []
                if label["probability"] is not None:
                    scores.append(label["probability"])
            elif active is not None:
                active["probability"] = score(scores)
                segments.append(active)
                active, scores = None, []
        if active is not None:
            active["probability"] = score(scores)
            segments.append(active)
    return sorted(segments, key=lambda item: (item["start_s"], item["label"]))


def infer_actions(video: Path, output: Path, config_path: Path = ROOT / "configs/decisions_actions.yaml",
                  max_seconds: float | None = None, requester=None, dry_run: bool = False) -> dict:
    config = load_config(config_path)
    video = Path(video).resolve()
    metadata = video_metadata(video)
    total_s = metadata["duration_ms"] / 1000
    if max_seconds is not None:
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be positive and finite")
        total_s = min(total_s, max_seconds)
    if not dry_run and requester is None and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for action inference")
    requester = requester or request_decision
    capture = cv2.VideoCapture(str(video))
    windows = []
    try:
        for start, end in anchors(total_s, config["step_s"]):
            context_start = max(0, end - config["window_s"])
            context = [round(context_start, 6), end]
            times = sample_times(*context, config["sample_fps"])
            labels = {name: {"state": "unknown", "probability": None} for name in config["actions"]}
            window = {"start_s": start, "end_s": end, "context_s": context,
                      "sample_times_s": times, "status": "error", "labels": labels}
            try:
                # Fresh decoder per window avoids unreliable seek-after-EOF on short clips.
                capture.release()
                capture = cv2.VideoCapture(str(video))
                frames, samples = encode_frames(capture, times, metadata["fps"], metadata["frame_count"],
                                                config["max_image_edge"], config["jpeg_quality"])
                window["frame_samples"] = samples
                window["sample_times_s"] = [timestamp for timestamp, _ in frames]
                window["max_sampling_error_s"] = max(abs(sample["sampling_error_s"]) for sample in samples)
                if len(frames) < 2:
                    window["status"] = "insufficient_frames"
                elif dry_run:
                    window["status"] = "not_run"
                else:
                    payload = decision_request(config, frames, start, end)
                    response = requester(payload, config["request_timeout_s"])
                    window["status"], window["labels"], window["usage"] = classify_response(response, config)
            except (RuntimeError, ValueError, error.URLError) as exc:
                window["error"] = str(exc)
            windows.append(window)
    finally:
        capture.release()
    report = {"video": str(video), "duration_s": total_s, "source_fps": metadata["fps"],
              "model": config["model"], "mode": "dry_run" if dry_run else "api", "config": str(Path(config_path).resolve()),
              "actions": list(config["actions"]), "threshold": config["threshold"],
              "step_s": config["step_s"], "window_s": config["window_s"],
              "negative_threshold": config["negative_threshold"],
              "timestamp_basis": "CFR estimate: decoded frame_index / source_fps; VFR is unsupported",
              "segments": merge_segments(windows, list(config["actions"])), "windows": windows,
              "notes": ("DRY RUN ONLY: frames decoded; no API request or action decision was made." if dry_run else
                        "Offline labels with approximate step-sized boundaries; probabilities are not calibrated.")}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report



