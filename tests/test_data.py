import subprocess

import imageio_ffmpeg
import numpy as np
import pytest
import torch

from uhm.data import (
    VideoClipDataset,
    assert_source_disjoint,
    extract_clips,
    import_video,
    load_annotations,
)
from uhm.data.video import IMAGENET_MEAN, IMAGENET_STD, sample_clip, video_metadata, write_video
from uhm.ontology import BODY_REGIONS, EVENT_CLASSES, RECIPIENTS
from uhm.schemas import ClipAnnotation


@pytest.fixture
def tiny_video(tmp_path):
    frames = [np.full((32, 32, 3), index * 30, dtype=np.uint8) for index in range(8)]
    return write_video(tmp_path / "source.mp4", frames, fps=10)


def row(video, **changes):
    values = {
        "video_id": video.stem, "video_path": str(video), "source_id": "original-session", "split": "train",
        "start_ms": 0, "end_ms": 799, "event_class": "no_event", "recipient": "unknown",
        "body_region": "unknown", "intensity": 0, "observable_region": False,
    }
    return ClipAnnotation(**(values | changes))


def test_sampler_uses_preceding_frame_never_nearest_future_frame(tiny_video):
    tensor = sample_clip(tiny_video, 0, 299, num_frames=4, image_size=32)
    rgb = (tensor * IMAGENET_STD + IMAGENET_MEAN) * 255
    luminance = rgb.mean(dim=(1, 2, 3)).tolist()
    assert luminance == pytest.approx([0, 0, 30, 60], abs=2)
    # Frame3 is at300ms and would have value90: it is future evidence at299ms.
    assert luminance[-1] < 70


def test_short_window_repeats_only_already_available_frames(tiny_video):
    tensor = sample_clip(tiny_video, 0, 50, num_frames=6, image_size=24)
    assert tensor.shape == (6, 3, 24, 24)
    assert tensor.dtype == torch.float32
    assert torch.equal(tensor, tensor[0].expand_as(tensor))


def test_source_and_same_file_cannot_leak_between_splits(tiny_video, tmp_path):
    first = row(tiny_video)
    with pytest.raises(ValueError, match="Source.*leaks"):
        assert_source_disjoint([first, row(tmp_path / "another.mp4", split="val")])
    with pytest.raises(ValueError, match="Video.*leaks"):
        assert_source_disjoint([first, row(tiny_video, source_id="renamed", split="val")])


def test_annotations_reject_out_of_video_bounds(tiny_video, tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(row(tiny_video, end_ms=801).model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds video duration"):
        load_annotations(path)


def test_import_extract_annotations_to_tensor_pipeline(tiny_video, tmp_path):
    imported = import_video(tiny_video, tmp_path / "raw")
    clips = extract_clips(imported, tmp_path / "clips", clip_duration_ms=400)
    assert len(clips) == 2
    assert sum(video_metadata(clip)["frame_count"] for clip in clips) == 8
    records = [row(clip, end_ms=399) for clip in clips]
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8")
    sample = VideoClipDataset(path, num_frames=3, image_size=24)[1]
    assert sample["frames"].shape == (3, 3, 24, 24)
    assert sample["event"].item() == EVENT_CLASSES.index("no_event")
    assert sample["recipient"].item() == RECIPIENTS.index("unknown")
    assert sample["region"].item() == BODY_REGIONS.index("unknown")
    assert not sample["observable"].item()


def test_example_encoding_is_silent_browser_h264(tiny_video):
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(tiny_video), "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    )
    streams = [line for line in result.stderr.splitlines() if "Stream #" in line]
    assert any("Video: h264" in line and "yuv420p" in line for line in streams)
    assert not any("Audio:" in line for line in streams)
