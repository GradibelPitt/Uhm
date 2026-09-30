"""Canonical JSONL loading and checks against source leakage."""

import os
from pathlib import Path

from uhm.ontology import ROOT
from uhm.schemas import ClipAnnotation


def assert_source_disjoint(records: list[ClipAnnotation]) -> None:
    """A recording/session belongs to one split, even after extracting clips."""
    assignments: dict[str, str] = {}
    video_assignments: dict[str, str] = {}
    for record in records:
        previous = assignments.setdefault(record.source_id, record.split)
        if previous != record.split:
            raise ValueError(
                f"Source {record.source_id!r} leaks across {previous!r} and {record.split!r}. "
                "Assign splits by source recording/session before extracting clips."
            )
        path = Path(record.video_path)
        path = path if path.is_absolute() else ROOT / path
        key = os.path.normcase(str(path.resolve()))
        previous_video = video_assignments.setdefault(key, record.split)
        if previous_video != record.split:
            raise ValueError(
                f"Video {record.video_path!r} leaks across {previous_video!r} and {record.split!r}, "
                "even though its source IDs differ."
            )


def load_annotations(path: Path | str) -> list[ClipAnnotation]:
    """Validate every nonempty JSONL line; never silently drop invalid labels."""
    path = Path(path)
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(ClipAnnotation.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"Invalid annotation at {path}:{number}: {exc}") from exc
    if not records:
        raise ValueError(f"Annotation file is empty: {path}")
    assert_source_disjoint(records)
    validate_video_bounds(records)
    return records


def validate_video_bounds(records: list[ClipAnnotation]) -> None:
    """Reject missing recordings and annotation intervals beyond their duration."""
    from .video import video_metadata

    durations = {}
    for record in records:
        video = Path(record.video_path)
        video = video if video.is_absolute() else ROOT / video
        key = os.path.normcase(str(video.resolve()))
        if key not in durations:
            durations[key] = video_metadata(video)["duration_ms"]
        duration = durations[key]
        if record.start_ms >= duration or record.end_ms > duration:
            raise ValueError(
                f"Annotation {record.video_id!r} [{record.start_ms}, {record.end_ms}]ms "
                f"exceeds video duration {duration}ms"
            )


def write_annotations(path: Path | str, records: list[ClipAnnotation]) -> Path:
    path = Path(path)
    assert_source_disjoint(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(row.model_dump_json() + "\n" for row in records), encoding="utf-8")
    return path
