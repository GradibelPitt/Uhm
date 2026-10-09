"""Offline egocentric motion estimates with image-motion evidence and explicit provenance."""

import base64
import hashlib
import json
import math
import time
from pathlib import Path
from urllib import error

import cv2
import numpy as np

from uhm.inference.decisions_actions import anchors, merge_segments, video_metadata
from uhm.inference.local_actions import _unique_object, request_local

PHASES = ["running", "walking", "standing", "takeoff", "airborne", "landing",
          "cushioning", "climbing", "unknown"]
FEET = ["both", "single", "alternating", "none", "unknown"]
CUES = ["forward_flow", "stride_bob", "launch", "flight", "surface_approach",
        "deceleration", "compression_rebound", "visible_contact", "hand_support",
        "camera_turn", "cut", "insufficient"]


def extract_evidence(video: Path) -> tuple[list, list]:
    """Decode a small, time-aligned proxy once; flow is screen motion, never force."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise ValueError("Cannot decode POV evidence video")
    frames, motion = [], []
    previous, previous_t = None, 0.0
    next_sample, next_image, index = 0.0, 0.0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            stamp = index / fps
            index += 1
            if stamp + .0001 < next_sample:
                continue
            next_sample += 1 / 12
            frame = cv2.resize(frame, (640, 360))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if stamp + .0001 >= next_image:
                next_image += .25
                _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frames.append((round(stamp, 4), "data:image/jpeg;base64," + base64.b64encode(jpg).decode()))
            if previous is not None:
                points = cv2.goodFeaturesToTrack(previous, 160, .02, 12)
                row = {"t": round(stamp, 3), "tracked": 0, "inlier_ratio": 0.0,
                       "dx": 0.0, "dy": 0.0, "rotation_deg": 0.0, "scale": 1.0,
                       "cut_candidate": False}
                if points is not None and len(points) >= 8:
                    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
                    good = status.ravel() == 1
                    a, b = points[good], moved[good]
                    row["tracked"] = len(a)
                    if len(a) >= 8:
                        matrix, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                                    ransacReprojThreshold=3)
                        if matrix is not None:
                            row.update(inlier_ratio=round(float(inliers.mean()), 3),
                                       dx=round(float(np.median((b-a)[:, 0, 0])), 2),
                                       dy=round(float(np.median((b-a)[:, 0, 1])), 2),
                                       rotation_deg=round(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])), 2),
                                       scale=round(math.hypot(matrix[0, 0], matrix[1, 0]), 4))
                difference = float(cv2.absdiff(previous, gray).mean())
                # This flags an ambiguous discontinuity for the vision model; it does not prove an edit.
                row["cut_candidate"] = difference > 45 and row["inlier_ratio"] < .3
                row["dt"] = round(stamp - previous_t, 4)
                motion.append(row)
            previous, previous_t = gray, stamp
    finally:
        cap.release()
    return frames, motion


def build_pov_request(config: dict, frames: list, motion: list, slots: list) -> dict:
    current_indices = [i for i, (t, _) in enumerate(frames) if slots[0][0] <= t < slots[0][1]]
    evidence_pairs = [[], *[[i, j] for j in current_indices for i in range(j)]]
    slot_schema = {"type": "object", "properties": {
        "observation": {"type": "string", "maxLength": 220},
        "phase": {"type": "string", "enum": PHASES},
        "feet": {"type": "string", "enum": FEET},
        "basis": {"type": "string", "enum": ["visible", "inferred", "unknown"]},
        "hand_load": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "cues": {"type": "array", "items": {"type": "string", "enum": CUES}, "minItems": 1, "maxItems": 4},
        "evidence_frames": {"type": "array", "items": {"type": "integer"}, "enum": evidence_pairs},
    }, "required": ["observation", "phase", "feet", "basis", "hand_load", "cues", "evidence_frames"],
        "additionalProperties": False}
    schema = {"type": "object", "properties": {
        "slots": {"type": "array", "items": slot_schema, "minItems": len(slots), "maxItems": len(slots)}},
        "required": ["slots"], "additionalProperties": False}
    # One motion summary per visual sample avoids pretending optical flow is camera acceleration.
    features = []
    for stamp, _ in frames:
        rows = [row for row in motion if stamp-.25 < row["t"] <= stamp]
        if rows:
            features.append({"t": stamp, "dy_px": round(float(np.mean([r["dy"] for r in rows])), 1),
                             "turn_deg": round(sum(r["rotation_deg"] for r in rows), 1),
                             "flow_px": round(float(np.mean([math.hypot(r["dx"], r["dy"]) for r in rows])), 1),
                             "coherence": round(float(np.mean([r["inlier_ratio"] for r in rows])), 2)})
    prompt = (
        "These are sequential frames from a first-person parkour camera. Analyze the CAMERA WEARER. "
        f"Frame times: {[t for t, _ in frames]}. Judge only interval {slots[0]}, earlier frames are context. "
        f"CURRENT frame indices are {current_indices}. Other images only explain preceding movement. "
        "Write observation FIRST: describe the last two images and how the wearer moves relative to surfaces. "
        "Then use that observation to classify phase. Do not automatically call approach a landing. "
        "running=forward movement with repeated strides; takeoff=leaving support into a jump; "
        "airborne=crossing a gap or dropping, feet need not be visible; landing=first support after flight; "
        "cushioning=compressing/recovering after landing; climbing=weight-bearing hands negotiating obstacle. "
        "Feet appearing in front of a distant surface are in the air, NOT planted. "
        "Infer hidden-foot support from approach to a surface, stopped descent, compression/rebound. "
        "Ordinary running footfalls are not jump landings. Distinguish camera turn from body translation. "
        "Only use cut when location changes discontinuously, never just because wearer looks down. "
        "feet=both/single/alternating/none/unknown describes support, not visibility. Landing need not be bilateral. "
        "basis=visible only when contact is directly seen, otherwise inferred; use unknown if ambiguous. "
        "hand_load=yes only if hands support weight against a surface, not swinging in air. "
        "Choose cues that actually support the judgment. Inferred landing needs flight or surface_approach "
        "AND deceleration or compression_rebound. Running needs forward_flow AND stride_bob. "
        "evidence_frames: two increasing zero-based indices; second within the target interval. "
        f"Screen motion summary, not metric acceleration: {json.dumps(features[-3:])}. "
        "Return one slot. Use empty evidence_frames only if uncertain."
    )
    return {"model": config["model"], "messages": [{"role": "user", "content": prompt,
            "images": [data.split(',', 1)[1] for _, data in frames]}], "format": schema, "stream": False,
            "think": False, "keep_alive": "10m",
            "options": {"num_ctx": 4096, "num_predict": 500, "temperature": 0, "seed": 42}}


def parse_pov_response(response: dict, times: list, slots: list, actions: list) -> list:
    windows = [{"start_s": start, "end_s": end, "status": "invalid_response",
                "motion": {"phase": "unknown", "feet": "unknown", "basis": "unknown", "cues": []},
                "labels": {name: {"state": "unknown", "probability": None} for name in actions}}
               for start, end in slots]
    try:
        if response.get("done_reason") == "length":
            return windows
        items = json.loads(response["message"]["content"], object_pairs_hook=_unique_object)["slots"]
        if not isinstance(items, list) or len(items) != len(slots):
            return windows
    except (KeyError, ValueError, TypeError):
        return windows
    for window, item in zip(windows, items, strict=True):
        if not isinstance(item, dict):
            continue
        phase, feet, basis = item.get("phase"), item.get("feet"), item.get("basis")
        cues, indices = item.get("cues", []), item.get("evidence_frames", [])
        if (not all(isinstance(v, str) for v in (phase, feet, basis, item.get("hand_load")))
                or phase not in PHASES or feet not in FEET or basis not in {"visible", "inferred", "unknown"}
                or not isinstance(cues, list) or any(cue not in CUES for cue in cues)
                or item.get("hand_load") not in {"yes", "no", "unknown"}):
            continue
        valid = (isinstance(indices, list) and len(indices) == 2
                 and all(type(i) is int for i in indices) and 0 <= indices[0] < indices[1] < len(times))
        evidence = [times[i] for i in indices] if valid else []
        within = lambda t, w=window: w["start_s"] <= t < w["end_s"]
        reason = None
        if "cut" in cues:
            reason = "scene_cut"
        elif not valid or not any(within(t) for t in evidence):
            reason = "missing_current_motion_evidence"
        elif phase in {"takeoff", "landing"} and not within(evidence[-1]):
            reason = "transition_outside_interval"
        elif basis == "inferred" and phase == "landing" and not (
            set(cues) & {"flight", "surface_approach"} and set(cues) & {"deceleration", "compression_rebound"}
        ):
            reason = "landing_sequence_incomplete"
        elif basis == "inferred" and phase == "running" and not {"forward_flow", "stride_bob"} <= set(cues):
            reason = "running_sequence_incomplete"
        elif basis == "unknown":
            reason = "insufficient_evidence"
        elif phase == "airborne" and feet not in {"none", "unknown"}:
            reason = "contradictory_airborne_support"
        elif phase == "landing" and feet == "none":
            reason = "contradictory_landing_support"
        if reason:
            window["status"] = "partial"
            window["motion"]["abstention_reason"] = reason
            window["motion"]["cues"] = cues
            continue
        window["status"] = "partial" if phase == "unknown" or feet == "unknown" else "ok"
        window["motion"] = {"phase": phase, "feet": feet, "basis": basis, "cues": cues,
                            "evidence_times_s": evidence, "observation": item.get("observation", "")}
        states = {
            "foot_support": "positive" if feet in {"both", "single", "alternating"} else
                            "negative" if feet == "none" else "unknown",
            "foot_push_off": "positive" if phase == "takeoff" else "negative",
            "foot_landing": "positive" if phase == "landing" else "negative",
            "cushioning": "positive" if phase == "cushioning" else "negative",
            "hand_load": {"yes": "positive", "no": "negative", "unknown": "unknown"}[item["hand_load"]],
        }
        for name, label in window["labels"].items():
            label.update(state=states.get(name, "unknown"), basis="visible" if name == "hand_load" else basis,
                         evidence_times_s=evidence)
            if phase == "unknown" and name in {"foot_push_off", "foot_landing", "cushioning"}:
                label["state"] = "unknown"
    return windows


def infer_pov_motion(video: Path, output: Path, config: dict, config_path: Path,
                     max_seconds=None, requester=None, start_seconds=0) -> dict:
    video, output = Path(video).resolve(), Path(output)
    metadata = video_metadata(video)
    duration = metadata["duration_ms"] / 1000
    if not math.isfinite(start_seconds) or not 0 <= start_seconds < duration:
        raise ValueError("start_seconds must be inside the video")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive and finite")
    end = min(duration, start_seconds + max_seconds) if max_seconds is not None else duration
    # Proxy is generated from this exact media; validate provenance, not just its duration.
    proxy = video.with_suffix(".analysis.mp4")
    sidecar = proxy.with_suffix(".json")
    with video.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if proxy.exists() and sidecar.exists():
        if json.loads(sidecar.read_text("utf-8")).get("source_sha256") != digest:
            raise ValueError("POV analysis proxy does not match the source")
        if abs(video_metadata(proxy)["duration_ms"] / 1000 - duration) > .1:
            raise ValueError("POV proxy duration mismatch")
    else:
        proxy = video
    began = time.perf_counter()
    frames, motion = extract_evidence(proxy)
    requester = requester or (lambda payload, timeout: request_local(payload, config["endpoint"], timeout))
    report = {"video": str(video), "video_sha256": digest, "duration_s": duration,
              "analysis_range_s": [start_seconds, end], "source_fps": metadata["fps"],
              "model": config["model"], "provider": "ollama", "mode": "local_vlm",
              "task": "egocentric_motion", "step_s": config["step_s"], "window_s": config["window_s"],
              "actions": list(config["actions"]), "contact_regions": config["contact_regions"],
              "windows": [], "segments": [], "contact_events": [], "camera_motion": motion,
              "evidence_policy": "egocentric_temporal_inference", "analysis_complete": False,
              "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
              "notes": "Offline motion inference; 0.5s bins are not event precision. Hidden feet can be inferred. "
                       "Image flow is not metric acceleration. No measured force or calibrated probability."}
    output.parent.mkdir(parents=True, exist_ok=True)
    for offset, stop in anchors(end-start_seconds, config.get("batch_s", 2)):
        start, finish = start_seconds+offset, start_seconds+stop
        slots = [[round(start+a, 6), round(start+b, 6)] for a, b in anchors(finish-start, config["step_s"])]
        selected = [(t, data) for i, (t, data) in enumerate(frames)
                    if (max(0, finish-config["window_s"]) <= t < start)
                    or start <= t < finish]
        call_began = time.perf_counter()
        try:
            raw = requester(build_pov_request(config, selected, motion, slots), config["request_timeout_s"])
            windows = parse_pov_response(raw, [t for t, _ in selected], slots, report["actions"])
            for window in windows:
                window["model_response"] = raw.get("message", {}).get("content", "")
        except (RuntimeError, ValueError, TimeoutError, error.URLError) as exc:
            windows = parse_pov_response({}, [], slots, report["actions"])
            for window in windows:
                window.update(status="error", error=str(exc))
        elapsed = round(time.perf_counter()-call_began, 3)
        for window in windows:
            window.update(latency_s=elapsed, request_batch_s=[start, finish],
                          context_s=[max(0, finish-config["window_s"]), finish])
        report["windows"].extend(windows)
        report["segments"] = merge_segments(report["windows"], report["actions"])
        report["contact_events"] = [
            {"event_type": name, "contact_region": config["contact_regions"][name],
             "start_s": w["start_s"], "end_s": w["end_s"], "basis": label.get("basis", "unknown"),
             "intensity": None, "probability": None}
            for w in report["windows"] for name, label in w["labels"].items() if label["state"] == "positive"]
        report["analysis_wall_s"] = round(time.perf_counter()-began, 3)
        report["analysis_complete"] = finish >= end
        pending = output.with_suffix(".tmp")
        pending.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        pending.replace(output)
        print(f"POV {finish:.1f}/{end:.1f}s: " + ", ".join(w["motion"]["phase"] for w in windows), flush=True)
    return report
