"""Torch dataset whose labels share the runtime's canonical ontology."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from uhm.ontology import BODY_REGIONS, EVENT_CLASSES, RECIPIENTS, ROOT
from uhm.schemas import ClipAnnotation

from .annotations import assert_source_disjoint, load_annotations, validate_video_bounds
from .augmentation import augment_video_clip
from .video import sample_clip


class VideoClipDataset(Dataset):
    def __init__(
        self, annotations: Path | str | list[ClipAnnotation], split: str | None = "train",
        num_frames: int = 6, image_size: int = 160, augment: bool = False,
    ):
        if isinstance(annotations, (Path, str)):
            records = load_annotations(annotations)
        else:
            records = list(annotations)
            validate_video_bounds(records)
        assert_source_disjoint(records)
        self.annotations = [row for row in records if split is None or row.split == split]
        if not self.annotations:
            raise ValueError(f"No annotations found for split {split!r}")
        if num_frames <= 0 or image_size <= 0:
            raise ValueError("num_frames and image_size must be positive")
        self.num_frames, self.image_size, self.augment = num_frames, image_size, augment

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        annotation = self.annotations[index]
        path = Path(annotation.video_path)
        if not path.is_absolute():
            path = ROOT / path
        frames = sample_clip(path, annotation.start_ms, annotation.end_ms,
                             self.num_frames, self.image_size)
        if self.augment:
            frames = augment_video_clip(frames)
        return {
            "frames": frames,
            "event": torch.tensor(EVENT_CLASSES.index(annotation.event_class.value), dtype=torch.long),
            "recipient": torch.tensor(RECIPIENTS.index(annotation.recipient.value), dtype=torch.long),
            "region": torch.tensor(BODY_REGIONS.index(annotation.body_region.value), dtype=torch.long),
            "intensity": torch.tensor(annotation.intensity, dtype=torch.float32),
            "observable": torch.tensor(annotation.observable_region, dtype=torch.bool),
            "ambiguous": torch.tensor(annotation.ambiguous, dtype=torch.bool),
            "annotation_index": torch.tensor(index, dtype=torch.long),
        }
