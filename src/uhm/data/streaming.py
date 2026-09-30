"""Causal source-clock training windows with labels for the current timestamp.

Unlike weak, whole-clip labels, these intervals must exhaustively describe each
recording, including explicit no-event intervals before and after contact.
"""

import os
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from uhm.ontology import BODY_REGIONS, EVENT_CLASSES, RECIPIENTS, ROOT
from uhm.schemas import ClipAnnotation

from .annotations import assert_source_disjoint, load_annotations, validate_video_bounds
from .augmentation import augment_video_clip
from .video import sample_clip, video_metadata


class CausalWindowDataset(Dataset):
    """Sample the same trailing windows as runtime; supervise only their endpoint.

    Annotation endpoints are inclusive integer milliseconds. Every selected
    recording must be covered from 0 through duration_ms - 1 without gaps or
    overlaps, so missing labels can never silently become negative examples.
    The source/file split checks run before filtering, including held-out rows.
    """

    def __init__(
        self, annotations: Path | str | list[ClipAnnotation], split: str | None = "train",
        num_frames: int = 6, image_size: int = 160, window_ms: int = 1000,
        step_ms: int = 125, augment: bool = False,
    ):
        if isinstance(annotations, (Path, str)):
            records = load_annotations(annotations)
        else:
            records = list(annotations)
            assert_source_disjoint(records)
            validate_video_bounds(records)
        assert_source_disjoint(records)
        self.annotations = [row for row in records if split is None or row.split == split]
        if not self.annotations:
            raise ValueError(f"No annotations found for split {split!r}")
        if min(num_frames, image_size, window_ms, step_ms) <= 0:
            raise ValueError("num_frames, image_size, window_ms and step_ms must be positive")
        if not isinstance(window_ms, int) or not isinstance(step_ms, int):
            raise TypeError("window_ms and step_ms must be integer milliseconds")
        self.num_frames, self.image_size, self.augment = num_frames, image_size, augment
        self.window_ms, self.step_ms = window_ms, step_ms

        recordings = defaultdict(list)
        self._paths: dict[int, Path] = {}
        for index, row in enumerate(self.annotations):
            path = Path(row.video_path)
            path = (path if path.is_absolute() else ROOT / path).resolve()
            recordings[os.path.normcase(str(path))].append(index)
            self._paths[index] = path

        # Each entry points to the original interval record, preserving provenance
        # even when many overlapping windows share a supervision interval.
        self.windows: list[tuple[int, int]] = []
        for indices in recordings.values():
            indices.sort(key=lambda index: self.annotations[index].start_ms)
            source_ids = {self.annotations[index].source_id for index in indices}
            if len(source_ids) != 1:
                raise ValueError("One source video cannot have multiple source IDs")
            duration_ms = video_metadata(self._paths[indices[0]])["duration_ms"]
            expected_start = 0
            for index in indices:
                row = self.annotations[index]
                if row.start_ms != expected_start:
                    raise ValueError(
                        f"Source {row.source_id!r} requires complete, non-overlapping labels: "
                        f"expected interval at {expected_start}ms, found {row.start_ms}ms"
                    )
                expected_start = row.end_ms + 1
            if expected_start != duration_ms:
                raise ValueError(
                    f"Source {self.annotations[indices[0]].source_id!r} requires complete labels "
                    f"through {duration_ms - 1}ms; labels end at {expected_start - 1}ms"
                )
            starts = [self.annotations[index].start_ms for index in indices]
            for timestamp_ms in range(0, duration_ms, step_ms):
                annotation_index = indices[bisect_right(starts, timestamp_ms) - 1]
                self.windows.append((annotation_index, timestamp_ms))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        annotation_index, timestamp_ms = self.windows[index]
        annotation = self.annotations[annotation_index]
        frames = sample_clip(
            self._paths[annotation_index], max(0, timestamp_ms - self.window_ms),
            timestamp_ms, self.num_frames, self.image_size,
        )
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
            "annotation_index": torch.tensor(annotation_index, dtype=torch.long),
        }
