import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from uhm.cli import app
from uhm.data import (
    create_example_dataset,
    create_example_stream_annotations,
    migrate_example_viewpoints,
)
from uhm.data.example import EXAMPLE_VERSION, FPS, SIZE, Condition, _annotation, _conditions
from uhm.schemas import BodyForceEvent, ClipAnnotation, export_schemas


def annotation_values(**changes):
    return {
        "video_id": "legacy", "video_path": "legacy.mp4", "source_id": "custom-session",
        "split": "train", "start_ms": 0, "end_ms": 999, "event_class": "no_event",
        "recipient": "unknown", "body_region": "unknown", "intensity": 0,
        "observable_region": False,
    } | changes


def test_legacy_and_custom_annotations_default_to_unknown_viewpoint():
    assert ClipAnnotation(**annotation_values()).viewpoint == "unknown"
    for viewpoint in ("first_person", "third_person", "unknown"):
        assert ClipAnnotation(**annotation_values(viewpoint=viewpoint)).viewpoint == viewpoint
    with pytest.raises(ValidationError):
        ClipAnnotation(**annotation_values(viewpoint="front"))


def test_annotation_schema_exposes_viewpoint_without_changing_runtime_event(tmp_path):
    export_schemas(tmp_path)
    annotation_schema = json.loads((tmp_path / "clip_annotation.schema.json").read_text())
    field = annotation_schema["properties"]["viewpoint"]
    assert field["default"] == "unknown"
    assert set(field["enum"]) == {"first_person", "third_person", "unknown"}
    assert "viewpoint" not in BodyForceEvent.model_fields
    runtime_schema = json.loads((tmp_path / "body_force_event.schema.json").read_text())
    assert "viewpoint" not in runtime_schema["properties"]


@pytest.fixture
def legacy_example(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    (data / "annotations").mkdir()
    records, sources = [], []
    for index in range(8):
        source_id = f"synthetic_source_{index:02d}"
        split, seed = ("train" if index < 6 else "val"), 4100 + index * 103
        (data / "raw" / f"{source_id}.mp4").write_bytes(b"existing raw media")
        clip_dir = data / "clips" / split
        clip_dir.mkdir(parents=True, exist_ok=True)
        ordered = _conditions()
        np.random.default_rng(seed).shuffle(ordered)
        for clip_index, condition in enumerate(ordered):
            video = clip_dir / f"{source_id}_{clip_index:03d}.mp4"
            video.write_bytes(b"existing clip media")
            records.append(_annotation(video, source_id, split, condition).model_dump(
                mode="json", exclude={"viewpoint"},
            ))
        sources.append({"source_id": source_id, "split": split, "seed": seed,
                        "raw_video": f"raw/{source_id}.mp4", "clip_count": 36})
    demo = data / "raw" / "demo.mp4"
    demo.write_bytes(b"existing heldout media")
    manifest = {"example_version": EXAMPLE_VERSION, "synthetic": True, "frame_rate": FPS,
                "resolution": [SIZE, SIZE], "sources": sources,
                "demo": {"video": "raw/demo.mp4", "seed": 95021}}
    (data / "example_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (data / "annotations" / "example.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8",
    )
    demo_row = _annotation(demo, "synthetic_demo_heldout", "test", Condition("idle"), 0, 19999)
    (data / "annotations" / "demo.jsonl").write_text(
        demo_row.model_dump_json(exclude={"viewpoint"}) + "\n", encoding="utf-8",
    )

    def metadata(path):
        path = Path(path)
        seconds = 20 if path.name == "demo.mp4" else 36 if path.parent.name == "raw" else 1
        return {"fps": FPS, "frame_count": FPS * seconds, "duration_ms": seconds * 1000,
                "width": SIZE, "height": SIZE}

    monkeypatch.setattr("uhm.data.example.video_metadata", metadata)
    monkeypatch.setattr("uhm.data.video.video_metadata", metadata)
    stream = create_example_stream_annotations(tmp_path)
    stream_rows = [json.loads(line) for line in stream.read_text().splitlines()]
    for row in stream_rows:
        row.pop("viewpoint")
    stream.write_text("".join(json.dumps(row) + "\n" for row in stream_rows), encoding="utf-8")
    return tmp_path


def test_legacy_example_files_remain_reusable_without_implicit_migration(legacy_example):
    root = legacy_example
    originals = {path: path.read_bytes() for path in (root / "data").rglob("*") if path.is_file()}
    assert create_example_dataset(root).name == "example.jsonl"
    assert create_example_stream_annotations(root).name == "example_stream.jsonl"
    for path, contents in originals.items():
        assert path.read_bytes() == contents


def test_verified_migration_preserves_every_other_field_and_media_and_is_idempotent(legacy_example):
    root = legacy_example
    annotation_dir = root / "data" / "annotations"
    custom = annotation_dir / "custom.jsonl"
    custom.write_text(json.dumps(annotation_values()) + "\n", encoding="utf-8")
    media = {path: path.read_bytes() for path in (root / "data").rglob("*.mp4")}
    paths = [annotation_dir / filename for filename in ("example.jsonl", "demo.jsonl", "example_stream.jsonl")]
    before = {path: [json.loads(line) for line in path.read_text().splitlines()] for path in paths}
    custom_before = custom.read_bytes()
    updated = migrate_example_viewpoints(root)
    assert set(updated) == {*paths, root / "data" / "example_manifest.json"}
    for path in paths:
        after = [json.loads(line) for line in path.read_text().splitlines()]
        assert all(row.pop("viewpoint") == "third_person" for row in after)
        assert after == before[path]
    manifest = json.loads((root / "data" / "example_manifest.json").read_text())
    for entry in [manifest, *manifest["sources"], manifest["demo"]]:
        assert entry["viewpoint"] == "third_person"
        assert entry["camera_orientation"] == "front_facing"
    for path, contents in media.items():
        assert path.read_bytes() == contents
    assert custom.read_bytes() == custom_before
    enriched = {path: path.read_bytes() for path in updated}
    assert migrate_example_viewpoints(root) == []
    assert create_example_stream_annotations(root).name == "example_stream.jsonl"
    for path, contents in enriched.items():
        assert path.read_bytes() == contents


@pytest.mark.parametrize("corruption", ["manifest_seed", "annotation_source", "explicit_first_person"])
def test_migration_rejects_unverified_provenance_before_any_writes(legacy_example, corruption):
    root = legacy_example
    if corruption == "manifest_seed":
        path = root / "data" / "example_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["sources"][0]["seed"] += 1
        path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        path = root / "data" / "annotations" / "demo.jsonl"
        row = json.loads(path.read_text())
        row["source_id" if corruption == "annotation_source" else "viewpoint"] = (
            "custom-session" if corruption == "annotation_source" else "first_person"
        )
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in (root / "data").rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="Unverified"):
        migrate_example_viewpoints(root)
    for path, contents in before.items():
        assert path.read_bytes() == contents


def test_cli_annotates_first_person_and_reports_viewpoint_counts(tmp_path, monkeypatch):
    monkeypatch.setattr("uhm.data.video.video_metadata", lambda _: {"duration_ms": 1000})
    output = tmp_path / "labels.jsonl"
    runner = CliRunner()
    base = ["annotate", str(tmp_path / "capture.mp4"), "user-session", "0", "999", "--output", str(output)]
    result = runner.invoke(app, [*base, "--viewpoint", "first_person"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, base)
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["validate-data", "--annotations", str(output)])
    assert result.exit_code == 0, result.output
    counts = json.loads(result.output)["viewpoint_counts"]
    assert counts == {"first_person": 1, "third_person": 0, "unknown": 1}
