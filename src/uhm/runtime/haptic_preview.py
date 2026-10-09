"""Compile perception results into device-independent, authored envelope previews."""

import hashlib
import json
import math
from collections import Counter
from typing import Literal

from pydantic import Field

from uhm.schemas import BodyForceEvent, Contract

# Durations and shapes are design choices, not learned or calibrated force signals.
PRESETS = {
    "foot_support": ("sustain", None), "hand_load": ("sustain", None),
    "hip_support": ("sustain", None), "torso_support": ("sustain", None),
    "cushioning": ("sustain", None), "foot_push_off": ("pulse", 180),
    "foot_landing": ("pulse", 220), "hand_impact": ("pulse", 220),
    "arm_impact": ("pulse", 220), "leg_impact": ("pulse", 220),
    "torso_impact": ("pulse", 220), "head_impact": ("pulse", 220),
    "impact": ("pulse", 220), "collision": ("pulse", 220), "explosion": ("pulse", 400),
}
CONTACT_REGIONS = {"feet", "hands", "arms", "hands_and_arms", "legs", "torso", "head", "hips"}


class EnvelopePoint(Contract):
    offset_ms: float = Field(ge=0)
    level: float = Field(ge=0, le=1)


class HapticCue(Contract):
    cue_id: str
    source_ids: list[str]
    event_type: str
    region: str
    target: Literal["player", "observed_subject", "camera_wearer"]
    basis: Literal["visible", "inferred", "unspecified", "model_event"]
    start_ms: float
    end_ms: float
    source_end_ms: float
    effect: Literal["pulse", "sustain"]
    envelope: list[EnvelopePoint]
    evidence_times_s: list[float] = Field(default_factory=list)
    perceptual_intensity: float | None = Field(default=None, ge=0, le=1)


class HapticPreview(Contract):
    schema_version: Literal["0.1"] = "0.1"
    mode: Literal["design_preview"] = "design_preview"
    source_kind: Literal["contact_timeline", "body_force_events"]
    source_video: str | None = None
    source_model: str | None = None
    source_digest: str | None = None
    duration_ms: float = Field(gt=0)
    policy: Literal["unit-envelope-v1"] = "unit-envelope-v1"
    amplitude_semantics: Literal["unit_envelope_not_force"] = "unit_envelope_not_force"
    timing_semantics: str
    inferred_included: bool = False
    physical_force_measured: Literal[False] = False
    hardware_output: Literal[False] = False
    cues: list[HapticCue] = Field(default_factory=list)
    omitted: dict[str, int] = Field(default_factory=dict)


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _interval(item: dict, duration_s: float) -> tuple[float, float]:
    start = _number(item.get("start_s"), "start_s")
    end = _number(item.get("end_s"), "end_s")
    if not 0 <= start < end <= duration_s + 1e-6:
        raise ValueError("Preview source interval is outside the video")
    return start, min(end, duration_s)


def _cue(index: int, *, event_type: str, start_ms: float, end_ms: float, **kwargs) -> HapticCue:
    effect, limit = PRESETS[event_type]
    duration = end_ms - start_ms
    if limit is not None:
        duration = min(duration, limit)
    if effect == "pulse":
        points = [(0, 0), (duration * .1, 1), (duration * .4, .45), (duration, 0)]
    else:
        ramp = min(50, duration / 3)
        points = [(0, 0), (ramp, 1), (duration - ramp, 1), (duration, 0)]
    return HapticCue(
        cue_id=f"cue-{index}", event_type=event_type, start_ms=start_ms,
        end_ms=start_ms + duration, source_end_ms=end_ms, effect=effect,
        envelope=[EnvelopePoint(offset_ms=t, level=level) for t, level in points], **kwargs,
    )


def preview_contacts(report: dict, *, include_inferred: bool = False) -> HapticPreview:
    """Intersect candidates with positive evidence; never bridge unknown/error gaps."""
    if not isinstance(report, dict):
        raise TypeError("Contact preview source must be a JSON object")
    duration = _number(report.get("duration_s"), "duration_s")
    preview = HapticPreview(
        source_kind="contact_timeline", duration_ms=duration * 1000,
        source_video=report.get("video"), source_model=report.get("model"),
        source_digest=hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        inferred_included=include_inferred,
        timing_semantics="Offline window boundaries; envelope interpolation adds no timing evidence",
    )
    if report.get("mode") == "dry_run":
        preview.omitted = {"dry_run": len(report.get("contact_events", []))}
        return preview
    if report.get("task") not in {"visible_contact", "egocentric_motion"}:
        preview.omitted = {"unsupported_task": len(report.get("contact_events", []))}
        return preview
    windows = sorted(report.get("windows", []), key=lambda w: _interval(w, duration)[0])
    previous_end = 0.0
    for window in windows:
        start, end = _interval(window, duration)
        if start < previous_end - 1e-9:
            raise ValueError("Preview evidence windows must not overlap")
        previous_end = end
    omitted = Counter()
    spans = []
    for index, event in enumerate(report.get("contact_events", [])):
        start, end = _interval(event, duration)
        name, region = event.get("event_type"), event.get("contact_region")
        if name not in PRESETS or name not in report.get("actions", []):
            omitted["unsupported_event"] += 1
            continue
        if region not in CONTACT_REGIONS:
            omitted["unknown_region"] += 1
            continue
        configured = report.get("contact_regions", {}).get(name)
        if configured is not None and configured != region:
            omitted["region_mismatch"] += 1
            continue
        for window in windows:
            left, right = max(start, window["start_s"]), min(end, window["end_s"])
            if left >= right:
                continue
            label = window.get("labels", {}).get(name, {})
            if (window.get("status") not in {"ok", "partial"}
                    or label.get("state") != "positive" or label.get("abstention_reason")):
                omitted["unconfirmed_interval"] += 1
                continue
            # Either source can carry the stricter evidence basis.
            bases = {label.get("basis"), event.get("basis")}
            if "unknown" in bases:
                omitted["unknown_basis"] += 1
                continue
            basis = "inferred" if "inferred" in bases else (
                "visible" if "visible" in bases else "unspecified")
            if report["task"] == "egocentric_motion" and basis == "unspecified":
                omitted["unknown_basis"] += 1
                continue
            if basis == "inferred" and not include_inferred:
                omitted["inferred_disabled"] += 1
                continue
            evidence = label.get("evidence_times_s", [])
            if not isinstance(evidence, list):
                raise TypeError("evidence_times_s must be a list")
            times = [_number(t, "evidence time") for t in evidence]
            if any(not 0 <= t < window["end_s"] for t in times):
                raise ValueError("Evidence time is outside its causal window")
            spans.append({"event_type": name, "region": region, "basis": basis,
                          "start_ms": left * 1000, "end_ms": right * 1000,
                          "source_ids": [f"contact-{index}"], "evidence_times_s": times})
    # Merge adjacent positives before shaping, so a 500 ms classifier does not
    # turn a continuous landing candidate into a train of invented impacts.
    spans.sort(key=lambda s: (s["event_type"], s["region"], s["basis"], s["start_ms"]))
    merged = []
    for span in spans:
        previous = merged[-1] if merged else None
        if (previous is not None and all(previous[k] == span[k] for k in ("event_type", "region", "basis"))
                and span["start_ms"] <= previous["end_ms"] + 1e-6):
            previous["end_ms"] = max(previous["end_ms"], span["end_ms"])
            previous["source_ids"] = sorted(set(previous["source_ids"] + span["source_ids"]))
            previous["evidence_times_s"] = sorted(set(previous["evidence_times_s"] + span["evidence_times_s"]))
        else:
            merged.append(span)
    target = "camera_wearer" if report["task"] == "egocentric_motion" else "observed_subject"
    preview.cues = [_cue(i, target=target, **span)
                    for i, span in enumerate(sorted(merged, key=lambda s: s["start_ms"]))]
    preview.omitted = dict(omitted)
    return preview


def preview_events(events: list[BodyForceEvent], duration_ms: float) -> HapticPreview:
    preview = HapticPreview(
        source_kind="body_force_events", duration_ms=_number(duration_ms, "duration_ms"),
        source_digest=hashlib.sha256(json.dumps(
            [event.model_dump(mode="json") for event in events], sort_keys=True,
            allow_nan=False,
        ).encode()).hexdigest(),
        timing_semantics="Source event timestamps and display durations; not measured contact duration",
    )
    seen = {}
    omitted = Counter()
    for event in sorted(events, key=lambda e: e.timestamp_ms):
        if event.event_id in seen:
            if seen[event.event_id] != event:
                raise ValueError("Conflicting source event IDs")
            omitted["duplicate_event"] += 1
            continue
        seen[event.event_id] = event
        if event.recipient.value != "player" or event.body_region.value == "unknown":
            omitted["unlocalized_or_nonplayer"] += 1
            continue
        if event.timestamp_ms >= duration_ms:
            raise ValueError("Source event starts outside the video")
        preview.cues.append(_cue(
            len(preview.cues), event_type=event.event_class.value, region=event.body_region.value,
            target="player", basis="model_event", start_ms=event.timestamp_ms,
            end_ms=min(duration_ms, event.timestamp_ms + event.duration_ms),
            source_ids=[event.event_id], perceptual_intensity=event.intensity,
        ))
    preview.omitted = dict(omitted)
    return preview


def envelope_at(cue: HapticCue, timestamp_ms: float) -> float:
    """Pure media-time sampling: pausing/seeking cannot leave stale playback state."""
    if not math.isfinite(timestamp_ms) or not cue.start_ms <= timestamp_ms < cue.end_ms:
        return 0.0
    offset = timestamp_ms - cue.start_ms
    for left, right in zip(cue.envelope, cue.envelope[1:]):
        if left.offset_ms <= offset <= right.offset_ms:
            ratio = (offset - left.offset_ms) / (right.offset_ms - left.offset_ms)
            return left.level + ratio * (right.level - left.level)
    return 0.0
