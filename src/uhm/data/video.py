"""Decode recorded video and construct causal, ImageNet-normalized clips."""

import math
import shutil
from collections.abc import Iterable
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _open_video(path: Path | str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {path}")
    return capture


def video_metadata(path: Path | str) -> dict:
    """Metadata used for the CFR baseline; frame timestamps are index / FPS."""
    capture = _open_video(path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError(f"Video has invalid frame count or FPS: {path}")
        return {"fps": fps, "frame_count": count, "width": width, "height": height,
                "duration_ms": round(count * 1000 / fps)}
    finally:
        capture.release()


def write_video(path: Path | str, frames: Iterable[np.ndarray], fps: float) -> Path:
    """Encode RGB frames to browser-playable H.264/yuv420p, without audio."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video FPS must be finite and positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("Cannot encode an empty video") from exc
    if first.ndim != 3 or first.shape[2] != 3 or first.dtype != np.uint8:
        raise ValueError("Video frames must be RGB uint8 arrays [H,W,3]")
    height, width = first.shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(path), (width, height), fps=fps, codec="libx264", pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", quality=7, macro_block_size=2, ffmpeg_log_level="error",
        output_params=["-preset", "ultrafast", "-movflags", "+faststart", "-an"],
    )
    writer.send(None)
    try:
        writer.send(np.ascontiguousarray(first, dtype=np.uint8))
        for frame in iterator:
            if frame.shape != first.shape:
                raise ValueError("All frames in a video must have the same dimensions")
            writer.send(np.ascontiguousarray(frame, dtype=np.uint8))
    finally:
        writer.close()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Video encoding failed: {path}")
    return path


def import_video(source: Path | str, destination_dir: Path | str) -> Path:
    """Copy a readable recording without modifying it or importing any audio features."""
    source = Path(source).resolve()
    video_metadata(source)
    destination = Path(destination_dir).resolve() / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source == destination:
        return destination
    if destination.exists():
        raise FileExistsError(f"Import destination already exists: {destination}")
    shutil.copy2(source, destination)
    return destination


def extract_clips(
    source: Path | str, destination_dir: Path | str, clip_duration_ms: int = 1000,
) -> list[Path]:
    """Split a recording into silent H.264 clips, keeping the final partial clip.

    This baseline assumes constant-frame-rate input. Transcode variable-frame-rate
    recordings to CFR before annotation. The original source/session ID must remain
    identical in the annotations for every extracted clip.
    """
    if clip_duration_ms <= 0:
        raise ValueError("clip_duration_ms must be positive")
    source, destination_dir = Path(source), Path(destination_dir)
    metadata = video_metadata(source)
    frames_per_clip = max(1, round(metadata["fps"] * clip_duration_ms / 1000))
    capture = _open_video(source)
    paths = []
    try:
        while True:
            frames = []
            for _ in range(frames_per_clip):
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not frames:
                break
            target = destination_dir / f"{source.stem}_{len(paths):03d}.mp4"
            if target.exists():
                raise FileExistsError(f"Clip destination already exists: {target}")
            paths.append(write_video(target, frames, metadata["fps"]))
    finally:
        capture.release()
    return paths


def preprocess_frames(frames: list[np.ndarray], image_size: int = 160) -> torch.Tensor:
    """RGB uint8 frames -> [T,3,H,W], using the pretrained encoder normalization."""
    if not frames or image_size <= 0:
        raise ValueError("At least one frame and a positive image_size are required")
    resized = np.stack([
        cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        for frame in frames
    ])
    tensor = torch.from_numpy(resized).permute(0, 3, 1, 2).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def sample_clip(
    video_path: Path | str, start_ms: int, end_ms: int,
    num_frames: int = 6, image_size: int = 160,
) -> torch.Tensor:
    """Sample frames at or before the requested times; never look beyond end_ms.

    Timestamps use the CFR frame clock, with floor rather than nearest rounding.
    Repeated indices pad short windows causally. End-of-file timestamps repeat the
    last available frame; negative times or reversed intervals are rejected.
    """
    if start_ms < 0 or end_ms < start_ms or num_frames <= 0:
        raise ValueError("Expected 0 <= start_ms <= end_ms and num_frames > 0")
    metadata = video_metadata(video_path)
    times = np.linspace(start_ms, end_ms, num_frames)
    indices = np.floor(times * metadata["fps"] / 1000 + 1e-9).astype(np.int64)
    indices = np.clip(indices, 0, metadata["frame_count"] - 1)
    capture = _open_video(video_path)
    decoded = {}
    try:
        for index in sorted(set(indices.tolist())):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not decode frame {index} from {video_path}")
            decoded[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    return preprocess_frames([decoded[int(index)] for index in indices], image_size)
