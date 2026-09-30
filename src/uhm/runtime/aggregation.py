"""Causal confidence gates and one haptic event per observed player episode."""

import math
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import yaml

from uhm.ontology import ROOT
from uhm.schemas import BodyForceEvent, Confidence, Prediction


@dataclass(frozen=True)
class RuntimeConfig:
    event_threshold: float = 0.60
    recipient_threshold: float = 0.60
    region_threshold: float = 0.55
    observable_threshold: float = 0.65
    confirm_samples: int = 2
    release_ms: int = 250
    cooldown_ms: int = 350
    duration_ms: int = 220
    smoothing_alpha: float = 0.5
    max_gap_ms: int = 750

    def __post_init__(self):
        for name in ("event_threshold", "recipient_threshold", "region_threshold", "observable_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number in [0,1]")
        alpha = self.smoothing_alpha
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("smoothing_alpha must be a finite number in (0,1]")
        for name in ("confirm_samples", "release_ms", "cooldown_ms", "duration_ms", "max_gap_ms"):
            value = getattr(self, name)
            minimum = 0 if name in {"release_ms", "cooldown_ms"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.duration_ms > 5000:
            raise ValueError("duration_ms cannot exceed the BodyForceEvent limit of 5000ms")


def load_runtime_config(path: Path | None = None) -> RuntimeConfig:
    """Read YAML policy settings, rejecting unknown keys or invalid ranges."""
    default_path = ROOT / "configs" / "runtime.yaml"
    if path is None and not default_path.exists():
        return RuntimeConfig()
    path = Path(path) if path is not None else default_path
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError("Runtime configuration must be a YAML mapping")  # noqa: TRY004
    try:
        return RuntimeConfig(**values)
    except TypeError as exc:
        raise ValueError(f"Invalid runtime configuration keys: {exc}") from exc


class EventAggregator:
    """Confirm player-only candidates, abstain on location, and latch an episode.

    An episode emits once, irrespective of physical-class or region jitter. Only
    sustained negatives release that latch. Cooldown begins at the confirmed
    release, and candidates collected during cooldown cannot confirm a new event.
    All timers use source presentation timestamps, never wall-clock processing time.
    Duration is an engineering display policy, not a learned physical quantity.
    """

    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config if config is not None else load_runtime_config()
        self.reset()

    def reset(self) -> None:
        """Clear episode, smoothing, confirmation and cooldown after a seek/new video."""
        self._last_timestamp: int | None = None
        self._active = False
        self._negative_since: int | None = None
        self._cooldown_until: int | None = None
        self._candidates: deque[Prediction] = deque(maxlen=self.config.confirm_samples)
        self._smoothed_intensity: float | None = None

    def _clear_candidates(self) -> None:
        self._candidates.clear()
        self._smoothed_intensity = None

    def _eligible(self, prediction: Prediction) -> bool:
        return (
            prediction.event_class.value != "no_event"
            and prediction.recipient.value == "player"
            and prediction.confidence.event >= self.config.event_threshold
            and prediction.confidence.recipient >= self.config.recipient_threshold
        )

    def _observable_region(self, prediction: Prediction) -> str:
        if (
            prediction.body_region.value != "unknown"
            and prediction.confidence.region >= self.config.region_threshold
            and prediction.observable_probability >= self.config.observable_threshold
        ):
            return prediction.body_region.value
        return "unknown"

    def update(self, prediction: Prediction) -> BodyForceEvent | None:
        """Consume one model prediction; return only a newly confirmed player event."""
        timestamp = prediction.timestamp_ms
        if self._last_timestamp is not None:
            gap = timestamp - self._last_timestamp
            if gap < 0 or gap > self.config.max_gap_ms:
                self.reset()
            elif gap == 0:
                # Repeated presentation times are not independent evidence samples.
                return None
        self._last_timestamp = timestamp

        if not self._eligible(prediction):
            # In particular, NPC/environment predictions cannot contaminate player salience.
            self._clear_candidates()
            if self._active:
                if self._negative_since is None:
                    self._negative_since = timestamp
                if timestamp - self._negative_since >= self.config.release_ms:
                    self._active = False
                    self._negative_since = None
                    self._cooldown_until = timestamp + self.config.cooldown_ms
            return None

        self._negative_since = None
        if self._active:
            return None
        if self._cooldown_until is not None and timestamp < self._cooldown_until:
            self._clear_candidates()
            return None

        self._candidates.append(prediction)
        if self._smoothed_intensity is None:
            self._smoothed_intensity = prediction.intensity
        else:
            alpha = self.config.smoothing_alpha
            self._smoothed_intensity = alpha * prediction.intensity + (1 - alpha) * self._smoothed_intensity
        if len(self._candidates) < self.config.confirm_samples:
            return None

        regions = [self._observable_region(candidate) for candidate in self._candidates]
        region = regions[0] if regions[0] != "unknown" and len(set(regions)) == 1 else "unknown"
        confidence = Confidence(
            event=min(candidate.confidence.event for candidate in self._candidates),
            recipient=min(candidate.confidence.recipient for candidate in self._candidates),
            region=min(candidate.confidence.region for candidate in self._candidates) if region != "unknown" else 0.0,
            # V0 does not train or calibrate a separate intensity-confidence estimate.
            intensity=0.0,
        )
        event = BodyForceEvent(
            event_id="evt_" + uuid.uuid4().hex,
            timestamp_ms=timestamp,
            event_class=prediction.event_class.value,
            recipient="player",
            body_region=region,
            intensity=self._smoothed_intensity,
            duration_ms=self.config.duration_ms,
            direction="unknown",
            confidence=confidence,
        )
        self._active = True
        self._clear_candidates()
        return event
