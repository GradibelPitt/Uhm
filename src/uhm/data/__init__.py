"""Video-only training data, annotations, and explicit synthetic examples."""

from .annotations import assert_source_disjoint, load_annotations
from .augmentation import augment_video_clip
from .dataset import VideoClipDataset
from .example import (
    create_example_dataset,
    create_example_stream_annotations,
    migrate_example_viewpoints,
)
from .streaming import CausalWindowDataset
from .video import extract_clips, import_video, sample_clip, video_metadata

__all__ = [
    "CausalWindowDataset",
    "VideoClipDataset",
    "assert_source_disjoint",
    "augment_video_clip",
    "create_example_dataset",
    "create_example_stream_annotations",
    "extract_clips",
    "import_video",
    "load_annotations",
    "migrate_example_viewpoints",
    "sample_clip",
    "video_metadata",
]
