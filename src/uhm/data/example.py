"""Small deterministic raster corpus for testing the pipeline, not gameplay claims.

The rendered contact, avatar identity, and visible anatomical zone are the evidence.
No class names, region IDs, numeric targets, or other label codes are burnt into video.
Ground truth is used only for training/evaluation, never by streaming inference.
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from uhm.ontology import BODY_REGIONS, ROOT
from uhm.schemas import ClipAnnotation

from .annotations import load_annotations, write_annotations
from .video import extract_clips, video_metadata, write_video

FPS = 12
SIZE = 256
EXAMPLE_VERSION = 1
VISIBLE_REGIONS = [region for region in BODY_REGIONS if region not in {"unknown", "whole_body"}]

# Front-facing anatomical left appears on the viewer's right. These are points on
# the visible avatar's body, not labels inferred from the side of the screen.
CONTACT_POINTS = {
    "head": (0, -92), "left_torso": (14, -51), "right_torso": (-14, -51),
    "lower_torso": (0, -5), "left_arm": (53, -26), "right_arm": (-53, -26),
    "left_leg": (21, 69), "right_leg": (-21, 69),
}


@dataclass(frozen=True)
class Condition:
    kind: str
    region: str = "unknown"
    intensity: float = 0.0


def _conditions() -> list[Condition]:
    if set(VISIBLE_REGIONS) != set(CONTACT_POINTS):
        raise ValueError("Example renderer needs contact geometry for the configured visible body regions")
    positives = [Condition("player", region, intensity)
                 for region in VISIBLE_REGIONS for intensity in (0.3, 0.85)]
    positives += [Condition("player_occluded", intensity=0.4), Condition("player_occluded", intensity=0.8)]
    negatives = [Condition(kind) for kind in (
        "idle", "idle", "camera_motion", "camera_motion", "menu", "menu",
        "ui_effect", "passing_object", "player_miss", "distant_motion",
    )]
    negatives += [Condition("npc", intensity=intensity) for intensity in (0.3, 0.45, 0.7, 0.85)]
    negatives += [Condition("environment", intensity=intensity) for intensity in (0.4, 0.8)]
    negatives += [Condition("recipient_occluded", intensity=intensity) for intensity in (0.4, 0.8)]
    return positives + negatives


def _polygon(frame, points, center, scale, color, outline):
    xy = np.rint(np.asarray(points) * scale + center).astype(np.int32)
    cv2.fillPoly(frame, [xy], color, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [xy], True, outline, max(1, round(scale * 2)), cv2.LINE_AA)


def _avatar(frame: np.ndarray, center: tuple, scale: float, player: bool) -> None:
    color = (52, 176, 200) if player else (124, 128, 148)
    edge = (112, 226, 232) if player else (169, 171, 188)
    cx, cy = center
    cv2.ellipse(frame, (round(cx), round(cy - 92 * scale)),
                (max(2, round(21 * scale)), max(2, round(24 * scale))),
                0, 0, 360, color, -1, cv2.LINE_AA)
    cv2.ellipse(frame, (round(cx), round(cy - 92 * scale)),
                (max(2, round(21 * scale)), max(2, round(24 * scale))),
                0, 0, 360, edge, max(1, round(scale * 2)), cv2.LINE_AA)
    # Separate contiguous polygons expose body-relative contact evidence.
    shapes = [
        [(3, -66), (26, -70), (34, -56), (27, -25), (3, -25)],
        [(-3, -66), (-26, -70), (-34, -56), (-27, -25), (-3, -25)],
        [(-27, -20), (27, -20), (32, 17), (15, 27), (0, 18), (-15, 27), (-32, 17)],
        [(39, -66), (49, -60), (58, -22), (69, 6), (58, 13), (45, -14), (33, -45)],
        [(-39, -66), (-49, -60), (-58, -22), (-69, 6), (-58, 13), (-45, -14), (-33, -45)],
        [(5, 28), (17, 32), (30, 22), (28, 67), (25, 101), (34, 112), (7, 114), (9, 67)],
        [(-5, 28), (-17, 32), (-30, 22), (-28, 67), (-25, 101), (-34, 112), (-7, 114), (-9, 67)],
    ]
    for points in shapes:
        _polygon(frame, points, center, scale, color, edge)
    # Face is a visual identity cue, identical across labels.
    eye_y = round(cy - 96 * scale)
    for offset in (-7, 7):
        cv2.circle(frame, (round(cx + offset * scale), eye_y), max(1, round(2 * scale)), (17, 40, 51), -1)


def _contact(frame, point, t: float, intensity: float, side: int, shape: int) -> None:
    px, py = point
    onset = 0.36
    travel = max(0, 1 - t / onset)
    position = (round(px + side * travel * (45 + intensity * 45)), round(py - travel * 12))
    radius = round(5 + intensity * 6)
    if shape % 2:
        cv2.rectangle(frame, (position[0] - radius, position[1] - radius),
                      (position[0] + radius, position[1] + radius), (215, 169, 81), -1, cv2.LINE_AA)
    else:
        cv2.circle(frame, position, radius, (215, 169, 81), -1, cv2.LINE_AA)
    if t >= onset:
        radius = round(7 + intensity * 13)
        p = (round(px), round(py))
        cv2.circle(frame, p, radius, (243, 99, 67), -1, cv2.LINE_AA)
        cv2.circle(frame, p, max(3, radius // 2), (255, 215, 156), -1, cv2.LINE_AA)
        cv2.circle(frame, p, radius + 4, (245, 128, 79), 2, cv2.LINE_AA)
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            inner = np.array(p) + (radius + 7) * np.array([np.cos(angle), np.sin(angle)])
            outer = np.array(p) + (radius + 12) * np.array([np.cos(angle), np.sin(angle)])
            cv2.line(frame, tuple(inner.astype(int)), tuple(outer.astype(int)), (250, 170, 96), 2, cv2.LINE_AA)


def _render(condition: Condition, frame_index: int, scene_seed: int, variation: int) -> np.ndarray:
    scene = np.random.default_rng(scene_seed)
    episode = np.random.default_rng(scene_seed * 1000 + variation)
    background = tuple(int(x) for x in scene.integers([13, 19, 29], [32, 37, 49]))
    frame = np.full((SIZE, SIZE, 3), background, dtype=np.uint8)
    t = frame_index / FPS
    scene_shift = int(scene.integers(-4, 5))
    camera_shift = round(9 * np.sin(t * 7)) if condition.kind == "camera_motion" else 0
    center = (109 + scene_shift + camera_shift, 129 + int(scene.integers(-2, 3)))
    for y in range(25, 256, 37):
        cv2.line(frame, (0, y), (255, y), tuple(min(255, x + 6) for x in background), 1)
    for x in range(0, 256, 42):
        cv2.line(frame, (x + camera_shift, 0), (x + camera_shift, 255), tuple(min(255, c + 4) for c in background), 1)
    cv2.line(frame, (0, 247), (255, 247), (48, 59, 73), 2)
    npc_center = (217 + scene_shift, 177)
    _avatar(frame, center, 1.0, player=True)
    _avatar(frame, npc_center, 0.46, player=False)
    crate_center = (216 + scene_shift, 43)
    cv2.rectangle(frame, (crate_center[0] - 17, 25), (crate_center[0] + 17, 61), (123, 116, 91), -1)
    cv2.line(frame, (crate_center[0] - 17, 25), (crate_center[0] + 17, 61), (165, 153, 112), 2)
    kind = condition.kind
    if kind == "player":
        delta = CONTACT_POINTS[condition.region]
        point = (center[0] + delta[0], center[1] + delta[1])
        _contact(frame, point, t, condition.intensity, int(episode.choice([-1, 1])), variation)
    elif kind == "npc":
        _contact(frame, (npc_center[0], npc_center[1] - 22), t, condition.intensity, -1, variation)
    elif kind == "environment":
        _contact(frame, crate_center, t, condition.intensity, -1, variation)
    elif kind == "player_occluded":
        # Player remains identifiable at the head/feet, but the actual contact is hidden.
        cv2.rectangle(frame, (29, 63), (183, 176), (74, 82, 94), -1)
        cv2.rectangle(frame, (32, 66), (180, 173), (82, 91, 103), 2)
        if t >= 0.36:
            for dx in (-60, -30, 0, 30, 60):
                cv2.line(frame, (center[0] + dx, 59), (center[0] + dx + 7, 47), (245, 160, 100), 2)
        else:
            cv2.circle(frame, (round(10 + t * 90), 117), round(6 + condition.intensity * 5), (215, 169, 81), -1)
    elif kind == "recipient_occluded":
        cv2.rectangle(frame, (12, 10), (247, 243), (88, 91, 100), -1)
        if t >= 0.36:
            # Both identity and contact site are hidden; no body zone is supervised.
            cv2.circle(frame, (129, 129), round(26 + 10 * condition.intensity), (224, 165, 103), 3)
    elif kind == "menu":
        cv2.rectangle(frame, (23, 35), (232, 225), (28, 39, 54), -1)
        for y in (73, 111, 149, 187):
            cv2.rectangle(frame, (54, y - 9), (202, y + 9), (54, 74, 96), -1)
    elif kind == "ui_effect":
        for i in range(4):
            cv2.rectangle(frame, (8 + i * 19, 8), (20 + i * 19, 16), (240, 105, 82), -1)
        cv2.circle(frame, (228, 93), 8 + frame_index % 5, (221, 111, 151), 2)
    elif kind == "passing_object":
        cv2.circle(frame, (round(15 + 225 * t), 253), 5, (214, 172, 96), -1)
    elif kind == "player_miss":
        cv2.line(frame, (175, 126), (204, 119), (170, 177, 189), 5)
        if t >= 0.36:
            cv2.line(frame, (204, 119), (255, 96), (246, 189, 107), 2)
    elif kind == "distant_motion":
        cv2.circle(frame, (round(200 + 25 * np.sin(t * 8)), 83), 5, (214, 172, 96), -1)
    return frame


def _annotation(video_path: Path, source_id: str, split: str, condition: Condition,
                start_ms: int = 0, end_ms: int = 999, video_id: str | None = None) -> ClipAnnotation:
    if condition.kind == "player":
        event, recipient, region, observable, ambiguous = "impact", "player", condition.region, True, False
    elif condition.kind == "player_occluded":
        event, recipient, region, observable, ambiguous = "impact", "player", "unknown", False, False
    elif condition.kind in {"npc", "environment", "recipient_occluded"}:
        event, region, observable = "impact", "unknown", False
        recipient = {"npc": "other_character", "environment": "environment", "recipient_occluded": "unknown"}[condition.kind]
        ambiguous = condition.kind == "recipient_occluded"
    else:
        event, recipient, region, observable, ambiguous = "no_event", "unknown", "unknown", False, False
    return ClipAnnotation(
        video_id=video_id or video_path.stem, video_path=Path(os.path.relpath(video_path, ROOT)).as_posix(),
        source_id=source_id, split=split, start_ms=start_ms, end_ms=end_ms,
        event_class=event, recipient=recipient, body_region=region,
        intensity=condition.intensity if event != "no_event" else 0.0,
        observable_region=observable, ambiguous=ambiguous, synthetic=True, viewpoint="third_person",
    )


def create_example_stream_annotations(root: Path | str = ROOT) -> Path:
    """Add dense causal-training truth to the eight existing raw recordings.

    The legacy example.jsonl labels a whole extracted second by its condition,
    including the projectile's precontact travel. Here the generator's source
    seeds reconstruct contact intervals on the original 36-second source clock.
    Frame 5 is the first contact frame: its 416.666...ms timestamp becomes 417ms
    on our integer clock. Earlier milliseconds are explicitly no_event.

    This function neither reads held-out demo labels nor regenerates any media.
    Existing annotations are reused only when they match the derived truth.
    """
    data_root = Path(root).resolve() / "data"
    manifest_path = data_root / "example_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("example_version") != EXAMPLE_VERSION:
        raise ValueError("Example data version differs; dense labels cannot be reconstructed")
    conditions = _conditions()
    sources = manifest.get("sources", [])
    if len(sources) != 8:
        raise ValueError("Dense example labels require exactly eight original raw sources")
    records = []
    contact_kinds = {"player", "npc", "environment", "player_occluded", "recipient_occluded"}
    onset_ms = math.ceil(math.ceil(0.36 * FPS) * 1000 / FPS)
    for source_index, source in enumerate(sources):
        source_id = f"synthetic_source_{source_index:02d}"
        split = "train" if source_index < 6 else "val"
        seed = 4100 + source_index * 103
        expected = {"source_id": source_id, "split": split, "seed": seed,
                    "raw_video": f"raw/{source_id}.mp4", "clip_count": len(conditions)}
        if any(source.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Original source provenance differs for {source_id}")
        source_path = data_root / source["raw_video"]
        metadata = video_metadata(source_path)
        if (not math.isclose(metadata["fps"], FPS, abs_tol=1e-6)
                or metadata["frame_count"] != len(conditions) * FPS
                or metadata["duration_ms"] != len(conditions) * 1000
                or (metadata["width"], metadata["height"]) != (SIZE, SIZE)):
            raise ValueError(f"Original source metadata differs for {source_id}")
        ordered = list(conditions)
        np.random.default_rng(seed).shuffle(ordered)
        for second, condition in enumerate(ordered):
            start_ms, end_ms = second * 1000, (second + 1) * 1000 - 1
            if condition.kind in contact_kinds:
                records.append(_annotation(
                    source_path, source_id, split, Condition("idle"),
                    start_ms, start_ms + onset_ms - 1,
                    video_id=f"{source_id}_{second:03d}_precontact",
                ))
                start_ms += onset_ms
            records.append(_annotation(
                source_path, source_id, split, condition, start_ms, end_ms,
                video_id=f"{source_id}_{second:03d}",
            ))
    destination = data_root / "annotations" / "example_stream.jsonl"
    if destination.exists():
        existing = load_annotations(destination)
        # Legacy versions predate viewpoint. Reuse those files without rewriting
        # them; only the explicit verified migration below enriches stored rows.
        if len(existing) != len(records) or any(
            old.viewpoint not in {"unknown", "third_person"}
            or old.model_dump(exclude={"viewpoint"}) != new.model_dump(exclude={"viewpoint"})
            for old, new in zip(existing, records, strict=True)
        ):
            raise ValueError("Existing dense example annotations differ; preserve or move them before rebuilding")
        return destination
    return write_annotations(destination, records)


def migrate_example_viewpoints(root: Path | str = ROOT) -> list[Path]:
    """Explicitly enrich only verified built-in example provenance with viewpoint.

    Validate the complete original source manifest, raw-video metadata, and the
    source/file ownership of every existing example annotation before writing.
    Custom annotations are never inspected or modified. All existing annotation
    fields and video files are preserved; already enriched files are left alone.
    Returns the paths actually updated, or an empty list on subsequent calls.
    """
    data_root = Path(root).resolve() / "data"
    manifest_path = data_root / "example_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("example_version") != EXAMPLE_VERSION
            or manifest.get("synthetic") is not True
            or manifest.get("frame_rate") != FPS
            or manifest.get("resolution") != [SIZE, SIZE]
            or manifest.get("viewpoint", "unknown") not in {"unknown", "third_person"}):
        raise ValueError("Unverified example manifest; cannot assign third-person viewpoint")
    sources = manifest.get("sources", [])
    if len(sources) != 8:
        raise ValueError("Viewpoint migration requires eight original example sources")
    expected_ownership = {"example.jsonl": {}, "example_stream.jsonl": {}, "demo.jsonl": {}}
    expected_video_metadata = []
    for index, source in enumerate(sources):
        source_id = f"synthetic_source_{index:02d}"
        split = "train" if index < 6 else "val"
        raw_path = data_root / "raw" / f"{source_id}.mp4"
        expected = {"source_id": source_id, "split": split, "seed": 4100 + index * 103,
                    "raw_video": f"raw/{source_id}.mp4", "clip_count": len(_conditions())}
        if (any(source.get(key) != value for key, value in expected.items())
                or source.get("viewpoint", "unknown") not in {"unknown", "third_person"}):
            raise ValueError(f"Unverified example source provenance: {source_id}")
        expected_ownership["example_stream.jsonl"][raw_path.resolve()] = (source_id, split)
        for clip in range(len(_conditions())):
            clip_path = data_root / "clips" / split / f"{source_id}_{clip:03d}.mp4"
            expected_ownership["example.jsonl"][clip_path.resolve()] = (source_id, split)
        expected_video_metadata.append((raw_path, len(_conditions()) * FPS))
    demo = manifest.get("demo", {})
    if (demo.get("video") != "raw/demo.mp4" or demo.get("seed") != 95021
            or demo.get("viewpoint", "unknown") not in {"unknown", "third_person"}):
        raise ValueError("Unverified held-out example demo provenance")
    demo_path = data_root / "raw" / "demo.mp4"
    expected_ownership["demo.jsonl"][demo_path.resolve()] = ("synthetic_demo_heldout", "test")
    expected_video_metadata.append((demo_path, 20 * FPS))
    for video_path, frame_count in expected_video_metadata:
        metadata = video_metadata(video_path)
        if (not math.isclose(metadata["fps"], FPS, abs_tol=1e-6)
                or metadata["frame_count"] != frame_count
                or metadata["duration_ms"] != round(frame_count * 1000 / FPS)
                or (metadata["width"], metadata["height"]) != (SIZE, SIZE)):
            raise ValueError(f"Unverified example video metadata: {video_path.name}")

    pending = {}
    for filename, ownership in expected_ownership.items():
        path = data_root / "annotations" / filename
        if not path.exists():
            continue
        # Retain the actual JSON fields, including legacy omission of optional
        # defaults. Enrichment must not manufacture changes to unrelated labels.
        original = [json.loads(line) for line in path.read_text("utf-8-sig").splitlines() if line.strip()]
        records = load_annotations(path)
        changed = False
        for values, record in zip(original, records, strict=True):
            video_path = Path(record.video_path)
            video_path = (video_path if video_path.is_absolute() else ROOT / video_path).resolve()
            if (ownership.get(video_path) != (record.source_id, record.split)
                    or not record.synthetic
                    or record.viewpoint not in {"unknown", "third_person"}):
                raise ValueError(f"Unverified example annotation source/viewpoint: {record.video_id}")
            if values.get("viewpoint") != "third_person":
                values["viewpoint"] = "third_person"
                changed = True
        if changed:
            pending[path] = "".join(json.dumps(values, ensure_ascii=False) + "\n" for values in original)

    manifest_changed = False
    for entry in [manifest, *sources, demo]:
        for key, value in {"viewpoint": "third_person", "camera_orientation": "front_facing"}.items():
            if entry.get(key) != value:
                entry[key] = value
                manifest_changed = True
    if manifest_changed:
        pending[manifest_path] = json.dumps(manifest, indent=2) + "\n"
    for path, contents in pending.items():
        path.write_text(contents, encoding="utf-8")
    return list(pending)


def create_example_dataset(root: Path | str) -> Path:
    """Create 288 one-second clips and an independent 20-second browser demo.

    Six raw source recordings are train, two are validation, and the demo is test.
    Existing matching output is reused. A changed/incomplete version is rejected
    rather than replacing recordings or annotations a user may have edited.
    """
    root = Path(root).resolve() / "data"
    root.mkdir(parents=True, exist_ok=True)
    annotations_path = root / "annotations" / "example.jsonl"
    manifest_path = root / "example_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("example_version") != EXAMPLE_VERSION:
            raise ValueError("Example data version differs; choose a new output directory")
        records = load_annotations(annotations_path)
        if len(records) != 288 or not (root / "raw" / "demo.mp4").exists():
            raise ValueError("Example dataset is incomplete; choose a new output directory")
        for row in records:
            if not (ROOT / row.video_path).is_file():
                raise ValueError(f"Example clip is missing: {row.video_id}")
        return annotations_path
    if annotations_path.exists() or list((root / "raw").glob("synthetic_source_*.mp4")):
        raise FileExistsError("Output contains unfinished or unrelated data; choose a new output directory")

    conditions = _conditions()
    records = []
    sources = []
    for source_index in range(8):
        split = "train" if source_index < 6 else "val"
        source_id = f"synthetic_source_{source_index:02d}"
        seed = 4100 + source_index * 103
        ordered = list(conditions)
        np.random.default_rng(seed).shuffle(ordered)
        source_path = root / "raw" / f"{source_id}.mp4"
        frames = (_render(condition, frame, seed, clip_index)
                  for clip_index, condition in enumerate(ordered) for frame in range(FPS))
        write_video(source_path, frames, FPS)
        clips = extract_clips(source_path, root / "clips" / split, clip_duration_ms=1000)
        if len(clips) != len(ordered):
            raise RuntimeError("Rendered source duration did not match its condition count")
        for clip, condition in zip(clips, ordered, strict=True):
            records.append(_annotation(clip, source_id, split, condition))
        sources.append({"source_id": source_id, "split": split, "seed": seed,
                        "viewpoint": "third_person", "camera_orientation": "front_facing",
                        "raw_video": source_path.relative_to(root).as_posix(), "clip_count": len(clips)})
    write_annotations(annotations_path, records)

    demo_conditions = []
    for index, region in enumerate(VISIBLE_REGIONS):
        demo_conditions += [Condition("idle"), Condition("player", region, 0.3 if index % 2 == 0 else 0.85)]
    demo_conditions += [Condition("npc", intensity=0.7), Condition("environment", intensity=0.6),
                        Condition("player_occluded", intensity=0.7), Condition("idle")]
    demo_path = root / "raw" / "demo.mp4"
    write_video(demo_path, (_render(condition, frame, 95021, index)
                           for index, condition in enumerate(demo_conditions) for frame in range(FPS)), FPS)
    demo_records = []
    onset_ms = round(5 * 1000 / FPS)  # frame5 is the first rendered contact after t>=0.36
    for index, condition in enumerate(demo_conditions):
        start_ms, end_ms = index * 1000, (index + 1) * 1000 - 1
        eventful = condition.kind in {"player", "npc", "environment", "player_occluded", "recipient_occluded"}
        if eventful:
            demo_records.append(_annotation(
                demo_path, "synthetic_demo_heldout", "test", Condition("idle"),
                start_ms, start_ms + onset_ms - 1, video_id=f"demo_window_{index:02d}_precontact"))
            start_ms += onset_ms
        demo_records.append(_annotation(
            demo_path, "synthetic_demo_heldout", "test", condition,
            start_ms, end_ms, video_id=f"demo_window_{index:02d}"))
    write_annotations(root / "annotations" / "demo.jsonl", demo_records)
    manifest = {
        "example_version": EXAMPLE_VERSION, "synthetic": True,
        "viewpoint": "third_person", "camera_orientation": "front_facing",
        "purpose": "Pipeline validation only; not evidence of gameplay or cross-game generalization.",
        "source_split_policy": "All clips from each raw source/session share one split; independent test demo seed.",
        "supported_example_event_classes": ["no_event", "impact"],
        "reserved_unsupported_event_classes": ["collision", "explosion"],
        "frame_rate": FPS, "resolution": [SIZE, SIZE], "clip_duration_ms": 1000,
        "train_clips": 216, "validation_clips": 72, "negative_for_player_haptics_fraction": 0.5,
        "sources": sources, "demo": {"video": "raw/demo.mp4", "seed": 95021,
                                      "viewpoint": "third_person", "camera_orientation": "front_facing",
                                      **video_metadata(demo_path)},
        "annotation_timing": "example.jsonl: legacy weak one-second windows. example_stream.jsonl (prepared separately): precise causal endpoint supervision. Dense training and held-out demo contact onset is417ms into each event second; all remaining intervals are explicitly negative.",
        "limitations": ["Procedural raster scenes with fixed front-facing bodies, not recorded gameplay.",
                        "Player identity is represented by cyan body appearance; NPC appearance is slate.",
                        "Ground truth salience is an illustrative synthetic target, not physical force.",
                        "Raw-source holdout tests variation within one renderer, not domain generalization.",
                        "No labels or text are burnt into the videos. Only pixels enter the model."],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "README.md").write_text(
        "# Synthetic video plumbing dataset\n\n"
        "288 real H.264 video clips at 12 FPS, 256×256 pixels; 216 training and 72 validation clips. "
        "Six training and two validation source recordings are disjoint. The 20-second demo is a third, held-out source. "
        "The same procedural renderer is used across splits, so these are not gameplay-generalization results.\n\n"
        "All current examples use `viewpoint: third_person`, with a front-facing player avatar. "
        "This corpus contains no first-person training or evaluation footage. Future first-person data "
        "must be labeled `first_person` and kept in a separate annotation collection for independent training. "
        "Unlabeled legacy or custom recordings default to `unknown`, never inferred third-person.\n\n"
        "Exactly half the clips are negative for player haptics: idle/menu/camera/UI/object motion, NPC impacts, "
        "environment impacts, or an occluded unknown recipient. Positive clips include all eight anatomical "
        "regions plus player impacts whose location is deliberately hidden. Anatomical left is screen-right "
        "for this front-facing actor. Local visible body contact supplies the region labels.\n\n"
        "Only `no_event` and `impact` are populated. `collision` and `explosion` remain reserved classes "
        "without example training support. Synthetic intensity corresponds to rendered contact salience, never Newtons.\n\n"
        "`annotations/example.jsonl` contains train/validation labels. `annotations/demo.jsonl` is held-out evaluation truth "
        "with frame-derived contact onsets and exhaustive negatives; the runtime must never read it. "
        "`example_manifest.json` records seeds, provenance and limitations. "
        "Annotation paths are relative to the repository root. All extracted clips are silent.\n\n"
        "Default training uses `annotations/example_stream.jsonl`: precise contact/negative intervals over "
        "the original sources, with 1728 training and 576 validation trailing windows (1000ms context, "
        "125ms cadence). Targets describe the current timestamp, including pre-contact negatives. "
        "The original weak whole-second labels remain a legacy comparison.\n\n"
        "For real data, import CFR videos, split into clips, annotate using the canonical ClipAnnotation schema, "
        "and assign entire source sessions to one split. Use `unknown` when contact or recipient cannot be observed. "
        "The loader rejects a source ID assigned to multiple splits.\n", encoding="utf-8",
    )
    return annotations_path
