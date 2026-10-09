"""Categorical video action labels from a locally running vision language model."""

import base64
import hashlib
import json
import math
import time
from pathlib import Path
from urllib import error, request
from urllib.parse import urlparse

import cv2
import numpy as np
import yaml

from uhm.inference.decisions_actions import (
    NAME,
    anchors,
    encode_frames,
    merge_segments,
    sample_times,
    video_metadata,
)
from uhm.ontology import ROOT

STATES = {"positive", "negative", "unknown"}


def load_local_config(path: Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("actions"), dict) or not config["actions"]:
        raise ValueError("Local action config needs a nonempty actions mapping")
    if not isinstance(config.get("model"), str) or not config["model"]:
        raise ValueError("A local vision model name is required")
    endpoint = urlparse(config.get("endpoint", ""))
    if endpoint.scheme != "http" or endpoint.hostname not in {"localhost", "127.0.0.1", "::1"} or endpoint.username:
        raise ValueError("Local action endpoint must be an HTTP loopback address")
    for key in ("window_s", "step_s", "sample_fps", "request_timeout_s"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if config["step_s"] > config["window_s"]:
        raise ValueError("step_s must not exceed window_s")
    if math.ceil(config["window_s"] * config["sample_fps"]) > 128:
        raise ValueError("A local window may not contain more than 128 images")
    for key, low, high in (("max_image_edge", 64, 2048), ("jpeg_quality", 1, 100)):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"Invalid {key}")
    for name, description in config["actions"].items():
        if not isinstance(name, str) or not NAME.fullmatch(name) or not isinstance(description, str) or not description:
            raise ValueError("Actions require identifier names and descriptions")
    if config.get("task") in {"visible_contact", "egocentric_motion"}:
        regions = config.get("contact_regions")
        if not isinstance(regions, dict) or set(regions) != set(config["actions"]):
            raise ValueError("Contact labels require a corresponding body region")
        allowed_regions = {"feet", "hands", "hands_and_arms", "arms", "legs", "torso", "head", "hips", "unknown"}
        if any(region not in allowed_regions for region in regions.values()):
            raise ValueError("Contact body region is not supported")
    transitions = config.get("transition_actions", [])
    if not isinstance(transitions, list) or any(name not in config["actions"] for name in transitions):
        raise ValueError("Transition actions must belong to the configured actions")
    if config.get("frame_layout", "images") not in {"images", "contact_sheet"}:
        raise ValueError("Unsupported frame layout")
    if config.get("task") == "egocentric_motion":
        if config.get("batch_s") != config["step_s"] or config["sample_fps"] != 4:
            raise ValueError("POV inference uses one interval per request and 4 fps visual evidence")
    return config


def contact_sheet(frames: list[tuple[float, str]], anchor_start: float) -> str:
    """One indexed visual sequence keeps frame identity visible to image-only runtimes."""
    columns, width, height, header = 2, 320, 240, 28
    sheet = np.full((math.ceil(len(frames) / columns) * (height + header), columns * width, 3), 24, dtype=np.uint8)
    for index, (stamp, encoded) in enumerate(frames):
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(encoded.split(",", 1)[1]), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid contact sheet frame")
        scale = min(width / frame.shape[1], height / frame.shape[0])
        frame = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
        x, y = (index % columns) * width, (index // columns) * (height + header)
        dx, dy = (width - frame.shape[1]) // 2, (height - frame.shape[0]) // 2
        sheet[y + header + dy:y + header + dy + frame.shape[0], x + dx:x + dx + frame.shape[1]] = frame
        current = stamp >= anchor_start
        color = (120, 230, 170) if current else (170, 170, 170)
        cv2.putText(sheet, f"Frame {index} | {stamp:.2f}s | {'CURRENT' if current else 'CONTEXT'}",
                    (x + 8, y + 19), cv2.FONT_HERSHEY_SIMPLEX, .44, color, 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Cannot encode contact sheet")
    return base64.b64encode(encoded).decode("ascii")


def build_local_request(config: dict, frames: list[tuple[float, str]], start: float, end: float) -> dict:
    label_schema = {
        "type": "object",
        "properties": {
            "state": {"type": "string", "enum": sorted(STATES)},
            "observable": {"type": "string", "enum": ["yes", "no", "unknown"]},
        },
        "required": ["state", "observable"],
        "additionalProperties": False,
    }
    if config.get("transition_actions"):
        label_schema["properties"]["evidence_frames"] = {
            "type": "array", "items": {"type": "integer", "minimum": 0, "maximum": len(frames) - 1},
            "maxItems": 2,
        }
        label_schema["required"].append("evidence_frames")
    schema = {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "enum": ["clear", "ambiguous", "none"]},
            "actions": {
                "type": "object",
                "properties": {name: label_schema for name in config["actions"]},
                "required": list(config["actions"]),
                "additionalProperties": False,
            },
        },
        "required": ["subject", "actions"],
        "additionalProperties": False,
    }
    task_instruction = (
        "The task is to detect visible contacts, support, external loads and force-relevant body interactions. "
        "Do not classify sports or name the activity. For example, a planted foot, a push-off and a foot landing "
        "all relate to the feet regardless of the sport. Evidence is the visible contact and movement in this "
        "anchor interval, not an activity name. "
        if config.get("task") == "visible_contact" else ""
    )
    prompt = task_instruction + (
        f"The images are ordered video frames at CFR-estimated seconds: {[stamp for stamp, _ in frames]}. "
        f"Classify only the primary person's actions during the anchor interval [{start:.3f}, {end:.3f}) seconds. "
        "Earlier frames are context. Judge observable body motion only. "
        "Set subject to clear when one primary person can be followed across the frames, even if an action is uncertain. "
        "Set subject to none only if no person is visible, or ambiguous if the primary person cannot be distinguished. "
        "Small background bystanders do not alone make the subject ambiguous; equally prominent main people do. "
        "For each action, observable=yes requires the relevant body parts and motion to be visible. "
        "Each contact requires the relevant body part and the interaction to be visible. "
        "Foot contacts require visible feet and their surface interaction; a crop that hides the feet is unknown. "
        "A hand moving freely in air alone is not evidence of hand impact or external load. "
        "Use positive for a clearly occurring action, negative only for clearly visible absence, and unknown "
        "for ambiguity, insufficient motion evidence, occlusion or missing body parts. "
        f"Action definitions: {json.dumps(config['actions'])}. "
        + (f"For positive transition actions {config['transition_actions']}, evidence_frames must contain "
           "exactly two zero-based image indices [before, after], in increasing order. The first must show "
           "the state BEFORE the contact change; the second must show the changed contact DURING the anchor "
           "interval. Both frames must show the SAME person and relevant body part. For landing/impact, "
           "require separation then contact; for push-off, require planted support then separation. "
           "An impact in earlier context is NOT an impact in the current interval. Static contact alone "
           "is support, not a new impact. Ordinary running steps are not jump landings. If these two "
           "states cannot be seen, use unknown. Use [] for negative, unknown and sustained support labels. "
           if config.get("transition_actions") else "")
        +
        f"Return a JSON object matching this schema: {json.dumps(schema)}."
    )
    images = [encoded.split(",", 1)[1] for _, encoded in frames]
    if config.get("frame_layout") == "contact_sheet":
        images = [contact_sheet(frames, start)]
        prompt = ("The single attached image is a contact sheet of chronological video frames. "
                  "Read the cells LEFT TO RIGHT then TOP TO BOTTOM. Each cell has a frame index and timestamp. "
                  "CURRENT cells are the interval to classify; CONTEXT cells show only earlier evidence. "
                  "Use the printed indices for evidence_frames. Follow the moving primary athlete, not bystanders. "
                  + prompt)
    return {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt,
                      "images": images}],
        "format": schema,
        "stream": False,
        "think": False,
        "keep_alive": "5m",
        "options": {"num_ctx": 8192, "num_predict": 1024, "temperature": 0, "seed": 42},
    }


def request_local(payload: dict, endpoint: str, timeout_s: float) -> dict:
    req = request.Request(endpoint.rstrip("/") + "/api/chat", json.dumps(payload).encode(),
                          {"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            return json.load(response)
    except error.HTTPError as exc:
        raise RuntimeError(f"Local vision runtime HTTP {exc.code}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Local vision runtime unavailable: {exc.reason}") from exc


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON field: {name}")
        result[name] = value
    return result


def parse_local_response(response: dict, action_names: list[str], *, transition_actions=(),
                         frame_times=(), anchor_start=0, anchor_end=math.inf) -> tuple[str, dict]:
    labels = {name: {"state": "unknown", "probability": None} for name in action_names}
    if not isinstance(response, dict) or response.get("done_reason") == "length":
        return "invalid_response", labels
    try:
        content = response["message"]["content"]
        result = json.loads(content, object_pairs_hook=_unique_object)
    except (KeyError, TypeError, ValueError):
        return "invalid_response", labels
    if not isinstance(result, dict) or not isinstance(result.get("actions"), dict):
        return "invalid_response", labels
    if result.get("subject") != "clear":
        return "unknown_visibility", labels
    for name, label in labels.items():
        item = result["actions"].get(name)
        if isinstance(item, dict) and item.get("observable") == "yes" and isinstance(item.get("state"), str) and item["state"] in STATES:
            if item["state"] == "positive" and name in transition_actions:
                indices = item.get("evidence_frames")
                valid = (isinstance(indices, list) and len(indices) == 2
                         and all(type(index) is int for index in indices)
                         and 0 <= indices[0] < indices[1] < len(frame_times))
                if not valid or not anchor_start <= frame_times[indices[1]] < anchor_end:
                    label["abstention_reason"] = "missing_or_outside_anchor_transition"
                    continue
                label["evidence_times_s"] = [frame_times[index] for index in indices]
            label["state"] = item["state"]
    return ("ok" if all(item["state"] != "unknown" for item in labels.values()) else "partial"), labels


JUMP_PHASES = {
    "foot_support": {"running_step", "push_off", "foot_landing"},
    "foot_push_off": {"push_off"},
    "foot_landing": {"foot_landing"},
    "torso_impact": {"torso_landing"},
    "torso_support": {"torso_landing", "torso_resting"},
}


def verify_jump_phase(config: dict, frames: list[tuple[float, str]], window: dict, requester) -> None:
    """A second, candidate-blind view can veto contacts, never add them."""
    affected = [name for name, label in window["labels"].items()
                if label["state"] == "positive" and name in JUMP_PHASES]
    if not affected:
        return
    phases = sorted(set().union(*JUMP_PHASES.values()) | {"airborne", "unknown"})
    schema = {"type": "object", "properties": {
        "phase": {"type": "string", "enum": phases},
        "feet_touch_surface": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "torso_touch_surface": {"type": "string", "enum": ["yes", "no", "unknown"]},
    }, "required": ["phase", "feet_touch_surface", "torso_touch_surface"], "additionalProperties": False}
    recent = [(stamp, encoded) for stamp, encoded in frames if stamp >= window["start_s"] - .3]
    prompt = (
        "Inspect the moving primary athlete in this chronological contact sheet, not stationary bystanders. "
        "Classify ONLY the CURRENT cells. What visible phase is occurring? running_step=ordinary running/walking; "
        "push_off=launch of a jump; airborne=body suspended with no surface support; "
        "foot_landing=landing from a jump on the feet; torso_landing=back/trunk first landing on a surface or mat; "
        "torso_resting=back/trunk already resting on a surface. Use unknown for ambiguity. "
        "Separately check whether the ATHLETE'S feet and torso physically touch a surface in CURRENT cells. "
        "Feet hanging above a mat do not touch it. A foot overlapping a background surface in the image is "
        "not sufficient contact evidence. Hidden feet are unknown. Do not infer feet contact from standing "
        "bystanders, the sport name, or torso contact. Return the required JSON."
    )
    payload = {"model": config["model"], "messages": [{"role": "user", "content": prompt,
               "images": [contact_sheet(recent, window["start_s"])]}], "format": schema,
               "stream": False, "think": False, "keep_alive": "5m",
               "options": {"num_ctx": 8192, "num_predict": 160, "temperature": 0, "seed": 42}}
    began = time.perf_counter()
    try:
        raw = requester(payload, config["request_timeout_s"])
        result = json.loads(raw["message"]["content"], object_pairs_hook=_unique_object)
        if raw.get("done_reason") == "length" or not isinstance(result, dict) or result.get("phase") not in phases:
            raise ValueError("Invalid verification response")
        window["phase_verification"] = result
        for name in affected:
            surface = "feet_touch_surface" if name.startswith("foot_") else "torso_touch_surface"
            # Push-off includes separation, so it does not require continuing foot contact.
            if result["phase"] not in JUMP_PHASES[name] or (name != "foot_push_off" and result.get(surface) != "yes"):
                window["labels"][name]["state"] = "unknown"
                window["labels"][name]["abstention_reason"] = "phase_verification_disagrees"
    except (KeyError, TypeError, ValueError, RuntimeError, TimeoutError, error.URLError) as exc:
        window["phase_verification"] = {"phase": "unknown", "error": str(exc)}
        for name in affected:
            window["labels"][name]["state"] = "unknown"
            window["labels"][name]["abstention_reason"] = "phase_verification_unavailable"
    window["verification_latency_s"] = round(time.perf_counter() - began, 3)
    window["status"] = "partial" if any(label["state"] == "unknown" for label in window["labels"].values()) else "ok"


def infer_local_actions(video: Path, output: Path, config_path: Path = ROOT / "configs/local_contacts.yaml",
                        max_seconds: float | None = None, requester=None, start_seconds: float = 0) -> dict:
    config = load_local_config(config_path)
    if config.get("task") == "egocentric_motion":
        from uhm.inference.pov_motion import infer_pov_motion

        return infer_pov_motion(video, output, config, config_path, max_seconds, requester, start_seconds)
    video = Path(video).resolve()
    metadata = video_metadata(video)
    duration = metadata["duration_ms"] / 1000
    if not math.isfinite(start_seconds) or not 0 <= start_seconds < duration:
        raise ValueError("start_seconds must be inside the video")
    analysis_end = duration
    if max_seconds is not None:
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be positive and finite")
        analysis_end = min(duration, start_seconds + max_seconds)
    requester = requester or (lambda payload, timeout: request_local(payload, config["endpoint"], timeout))
    windows = []
    analysis_began = time.perf_counter()
    for offset_start, offset_end in anchors(analysis_end - start_seconds, config["step_s"]):
        start, end = round(start_seconds + offset_start, 6), round(start_seconds + offset_end, 6)
        context_start = max(0, end - config["window_s"])
        times = sample_times(context_start, end, config["sample_fps"])
        labels = {name: {"state": "unknown", "probability": None} for name in config["actions"]}
        window = {"start_s": start, "end_s": end, "context_s": [context_start, end],
                  "status": "error", "labels": labels}
        capture = cv2.VideoCapture(str(video))
        try:
            frames, samples = encode_frames(capture, times, metadata["fps"], metadata["frame_count"],
                                            config["max_image_edge"], config["jpeg_quality"])
        except (RuntimeError, ValueError) as exc:
            window["error"] = str(exc)
            windows.append(window)
            continue
        finally:
            capture.release()
        window["frame_samples"] = samples
        window["sample_times_s"] = [stamp for stamp, _ in frames]
        window["max_sampling_error_s"] = max(abs(sample["sampling_error_s"]) for sample in samples)
        if len(frames) < 2:
            window["status"] = "insufficient_frames"
        else:
            began = time.perf_counter()
            try:
                response = requester(build_local_request(config, frames, start, end), config["request_timeout_s"])
                window["status"], window["labels"] = parse_local_response(
                    response, list(config["actions"]),
                    transition_actions=config.get("transition_actions", []),
                    frame_times=[stamp for stamp, _ in frames], anchor_start=start, anchor_end=end,
                )
                window["runtime"] = {key: response[key] for key in (
                    "total_duration", "load_duration", "prompt_eval_count", "eval_count"
                ) if key in response}
                if config.get("verify_jump_phase"):
                    verify_jump_phase(config, frames, window, requester)
            except (RuntimeError, ValueError, TimeoutError, error.URLError) as exc:
                window["error"] = str(exc)
            window["latency_s"] = round(time.perf_counter() - began, 3)
        windows.append(window)
    report = {
        "video": str(video), "duration_s": duration, "analysis_range_s": [start_seconds, analysis_end],
        "source_fps": metadata["fps"],
        "model": config["model"], "provider": "ollama", "mode": "local_vlm",
        "actions": list(config["actions"]), "step_s": config["step_s"], "window_s": config["window_s"],
        "timestamp_basis": "CFR estimate: decoded frame_index / source_fps; VFR is unsupported",
        "segments": merge_segments(windows, list(config["actions"])), "windows": windows,
        "task": config.get("task", "visible_actions"),
        "analysis_wall_s": round(time.perf_counter() - analysis_began, 3),
        "contact_regions": config.get("contact_regions", {}),
        "evidence_policy": "ordered_transition_frames" if config.get("transition_actions") else "categorical",
        "frame_layout": config.get("frame_layout", "images"),
        "phase_verification_enabled": bool(config.get("verify_jump_phase")),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "notes": "Local categorical judgments of visible motion/contact. Approximate boundaries; force magnitude is not measured.",
    }
    if config.get("task") == "visible_contact":
        report["contact_events_are_candidates"] = True
        report["contact_events"] = [{
            "event_type": segment["label"],
            "contact_region": config["contact_regions"][segment["label"]],
            "start_s": segment["start_s"], "end_s": segment["end_s"],
            "intensity": None, "probability": None,
        } for segment in report["segments"]]
    with video.open("rb") as source:
        report["video_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
