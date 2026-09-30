"""Use the checkpoint's training-time sampling contract during validation too."""

from pathlib import Path

from .dataset import VideoClipDataset
from .streaming import CausalWindowDataset


def build_dataset(annotations: Path, config: dict, split: str, *, augment: bool = False):
    settings = config["data"]
    common = {key: settings[key] for key in ("num_frames", "image_size")}
    # Checkpoints predating causal endpoint supervision used whole-clip labels.
    sampling = settings.get("sampling", "clip")
    if sampling == "causal":
        return CausalWindowDataset(
            annotations, split=split, augment=augment, **common,
            window_ms=settings["window_ms"], step_ms=settings.get("step_ms", 125),
        )
    if sampling == "clip":
        return VideoClipDataset(annotations, split=split, augment=augment, **common)
    raise ValueError(f"Unsupported data sampling mode: {sampling!r}")
