import json

import numpy as np
import pytest
import torch

from uhm.data import CausalWindowDataset, VideoClipDataset, create_example_stream_annotations
from uhm.data.example import EXAMPLE_VERSION, FPS, SIZE, _conditions
from uhm.data.video import sample_clip, write_video
from uhm.ontology import BODY_REGIONS, EVENT_CLASSES
from uhm.schemas import ClipAnnotation


def annotation(video, start, end, *, event="no_event", split="train", source="source"):
    return ClipAnnotation(
        video_id=f"{source}_{start}_{end}", video_path=str(video), source_id=source,
        split=split, start_ms=start, end_ms=end, event_class=event,
        recipient="player" if event == "impact" else "unknown",
        body_region="head" if event == "impact" else "unknown",
        intensity=0.5 if event == "impact" else 0.0,
        observable_region=event == "impact",
    )


@pytest.fixture
def video(tmp_path):
    return write_video(
        tmp_path / "source.mp4",
        [np.full((24, 24, 3), frame * 20, dtype=np.uint8) for frame in range(12)], 12,
    )


@pytest.fixture
def contact_intervals(video):
    return [annotation(video, 0, 416), annotation(video, 417, 699, event="impact"),
            annotation(video, 700, 999)]


def test_window_supervision_uses_current_time_even_when_history_contains_contact(contact_intervals):
    dataset = CausalWindowDataset(contact_intervals, num_frames=3, image_size=24)
    assert len(dataset) == 8
    assert len(dataset.annotations) == 3
    # 375ms is before the first rendered contact frame; 500ms is after it.
    assert dataset[3]["event"].item() == EVENT_CLASSES.index("no_event")
    assert dataset[4]["event"].item() == EVENT_CLASSES.index("impact")
    assert dataset[4]["region"].item() == BODY_REGIONS.index("head")
    # At 750ms the trailing window still includes the hit, but the current label
    # is negative. Whole-window presence labels would incorrectly keep it active.
    assert dataset[6]["event"].item() == EVENT_CLASSES.index("no_event")
    assert dataset[6]["annotation_index"].item() == 2


def test_causal_short_prefix_and_runtime_sampling_are_identical(video, contact_intervals):
    dataset = CausalWindowDataset(contact_intervals, num_frames=6, image_size=24)
    first = dataset[0]["frames"]
    assert torch.equal(first, first[0].expand_as(first))
    for index in (1, 4, 7):
        _, timestamp = dataset.windows[index]
        expected = sample_clip(video, max(0, timestamp - 1000), timestamp, 6, 24)
        assert torch.equal(dataset[index]["frames"], expected)


def test_augmentation_receives_only_the_causal_prefix_and_preserves_targets(
    video, contact_intervals, monkeypatch,
):
    expected = sample_clip(video, 0, 125, 6, 24)
    received = []

    def probe(frames):
        received.append(frames.clone())
        return frames + 0.01

    monkeypatch.setattr("uhm.data.streaming.augment_video_clip", probe)
    dataset = CausalWindowDataset(contact_intervals, num_frames=6, image_size=24, augment=True)
    sample = dataset[1]
    assert len(received) == 1
    assert torch.equal(received[0], expected)
    assert sample["event"].item() == EVENT_CLASSES.index("no_event")
    assert sample["annotation_index"].item() == 0


def test_both_datasets_use_common_augmentation_only_when_requested(video, monkeypatch):
    calls = []

    def probe(frames):
        calls.append(frames.shape)
        return frames

    monkeypatch.setattr("uhm.data.streaming.augment_video_clip", probe)
    monkeypatch.setattr("uhm.data.dataset.augment_video_clip", probe)
    train_rows = [annotation(video, 0, 999)]
    validation_rows = [annotation(video, 0, 999, split="val")]
    for dataset_type in (CausalWindowDataset, VideoClipDataset):
        dataset_type(train_rows, image_size=24, augment=True)[0]
        assert len(calls) == 1
        calls.clear()
        dataset_type(validation_rows, split="val", image_size=24, augment=False)[0]
        assert not calls


def test_contact_boundary_is_inclusive_at_integer_millisecond(video, contact_intervals):
    dataset = CausalWindowDataset(contact_intervals, step_ms=1, image_size=24)
    assert dataset.windows[416][0] == 0
    assert dataset.windows[417][0] == 1
    assert dataset.windows[699][0] == 1
    assert dataset.windows[700][0] == 2


@pytest.mark.parametrize("intervals", [
    [(1, 999)],                       # Unlabelled initial timestamp.
    [(0, 415), (417, 999)],          # A missing millisecond is not implicit idle.
    [(0, 416), (416, 999)],          # Inclusive endpoint overlap.
    [(0, 998)],                      # Unlabelled final timestamp.
    [(0, 1000)],                     # Beyond the final valid integer timestamp.
])
def test_incomplete_or_overlapping_labels_are_rejected(video, intervals):
    rows = [annotation(video, start, end) for start, end in intervals]
    with pytest.raises(ValueError, match="complete"):
        CausalWindowDataset(rows)


def test_source_and_file_leakage_rejected_before_split_filter(video, tmp_path):
    training = annotation(video, 0, 999)
    with pytest.raises(ValueError, match="Source.*leaks"):
        CausalWindowDataset([training, annotation(tmp_path / "heldout.mp4", 0, 999, split="test")])
    with pytest.raises(ValueError, match="Video.*leaks"):
        CausalWindowDataset([training, annotation(video, 0, 999, split="test", source="alias")])


def test_train_and_validation_never_include_heldout_windows(video, tmp_path):
    videos = [video]
    for split in ("val", "test"):
        videos.append(write_video(tmp_path / f"{split}.mp4", [np.zeros((24, 24, 3), dtype=np.uint8)] * 12, 12))
    rows = [annotation(path, 0, 999, split=split, source=split)
            for path, split in zip(videos, ("train", "val", "test"), strict=True)]
    for split in ("train", "val"):
        dataset = CausalWindowDataset(rows, split=split)
        assert len(dataset) == 8
        assert {row.split for row in dataset.annotations} == {split}
        assert {row.source_id for row in dataset.annotations} == {split}


@pytest.fixture
def original_source_manifest(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    sources = []
    for index in range(8):
        source = f"synthetic_source_{index:02d}"
        (data / "raw" / f"{source}.mp4").write_bytes(b"unchanged source media")
        sources.append({"source_id": source, "split": "train" if index < 6 else "val",
                        "seed": 4100 + index * 103, "raw_video": f"raw/{source}.mp4",
                        "clip_count": len(_conditions())})
    (data / "example_manifest.json").write_text(json.dumps({
        "example_version": EXAMPLE_VERSION, "sources": sources,
    }), encoding="utf-8")
    metadata = {"fps": FPS, "frame_count": FPS * 36, "duration_ms": 36000,
                "width": SIZE, "height": SIZE}
    monkeypatch.setattr("uhm.data.example.video_metadata", lambda _: dict(metadata))
    return tmp_path, metadata


def test_dense_truth_is_derived_only_from_original_sources_without_changing_media(original_source_manifest):
    root, _ = original_source_manifest
    originals = {path: path.read_bytes() for path in (root / "data").rglob("*") if path.is_file()}
    destination = create_example_stream_annotations(root)
    records = [ClipAnnotation.model_validate_json(line) for line in destination.read_text().splitlines()]
    assert destination.name == "example_stream.jsonl"
    assert len(records) == 8 * (36 + 26)
    assert {row.split for row in records} == {"train", "val"}
    assert len({row.source_id for row in records if row.split == "train"}) == 6
    assert len({row.source_id for row in records if row.split == "val"}) == 2
    assert all("demo" not in row.video_path for row in records)
    for row in records:
        if row.event_class.value == "impact":
            assert row.start_ms % 1000 == 417
            assert row.end_ms % 1000 == 999
        if row.video_id.endswith("precontact"):
            assert row.event_class.value == "no_event"
            assert row.start_ms % 1000 == 0
            assert row.end_ms % 1000 == 416
    for path, original in originals.items():
        assert path.read_bytes() == original


def test_dense_truth_rejects_raw_source_metadata_changes(original_source_manifest):
    root, metadata = original_source_manifest
    metadata["frame_count"] -= 1
    with pytest.raises(ValueError, match="metadata differs"):
        create_example_stream_annotations(root)
    assert not (root / "data" / "annotations" / "example_stream.jsonl").exists()
