"""Fetch hash-pinned real videos and make local browser-playable derivatives."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/action-video-sources.json"
ORIGINALS = ROOT / "data/raw/action_validation/original"
DERIVATIVES = ROOT / "data/raw/action_validation/mp4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for frame_index in (0, count // 2, max(0, count - 1)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            if not capture.read()[0]:
                raise ValueError(f"Cannot decode frame {frame_index} of {path}")
        return {"fps": fps, "frames": count, "duration_s": round(count / fps, 3),
                "width": width, "height": height}
    finally:
        capture.release()


def fetch(item: dict) -> dict:
    ORIGINALS.mkdir(parents=True, exist_ok=True)
    DERIVATIVES.mkdir(parents=True, exist_ok=True)
    original = ORIGINALS / item["filename"]
    legacy = ROOT / "data/raw" / item["filename"]
    if not original.exists() and legacy.exists() and sha256(legacy) == item["sha256"]:
        shutil.copy2(legacy, original)
    if not original.exists():
        temporary = original.with_suffix(original.suffix + ".part")
        req = Request(item["url"], headers={"User-Agent": "Mozilla/5.0 (local research validation)"})
        with urlopen(req, timeout=90) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        if sha256(temporary) != item["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"SHA-256 mismatch for {item['id']}")
        temporary.replace(original)
    if sha256(original) != item["sha256"]:
        raise ValueError(f"SHA-256 mismatch for existing {original}")
    source_info = inspect(original)
    mp4 = DERIVATIVES / f"{item['id']}.mp4"
    if not mp4.exists():
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(original), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(mp4)], check=True)
    mp4_info = inspect(mp4)
    return {"id": item["id"], "source": item["source"], "original": str(original),
            "original_sha256": item["sha256"], "original_video": source_info,
            "local_mp4": str(mp4), "local_mp4_sha256": sha256(mp4), "local_mp4_video": mp4_info,
            "attribution": item["attribution"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="Manifest IDs; omit to fetch all")
    args = parser.parse_args()
    items = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wanted = set(args.only) if args.only else {item["id"] for item in items}
    unknown = wanted - {item["id"] for item in items}
    if unknown:
        parser.error(f"Unknown IDs: {sorted(unknown)}")
    results = [fetch(item) for item in items if item["id"] in wanted]
    output = ROOT / "runs/action_video_imports.json"
    existing = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    existing.update({entry["id"]: entry for entry in results})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
