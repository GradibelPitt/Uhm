"""CLI entrypoints keep dataset, learning, inference and serving independently runnable."""

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from .ontology import ROOT

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.command("schema")
def schema():
    """Export canonical Pydantic contracts as JSON Schema for frontend generation."""
    from .schemas import export_schemas

    typer.echo(str(export_schemas()))


@app.command("prepare-data")
def prepare_data():
    """Generate the disclosed synthetic learning/validation/video demonstration corpus."""
    from .data.example import create_example_dataset, create_example_stream_annotations

    create_example_dataset(ROOT)
    typer.echo(str(create_example_stream_annotations(ROOT)))


@app.command("import-video")
def import_video_command(source: Path, destination: Path = ROOT / "data/raw"):
    """Copy a readable source video into the raw video collection."""
    from .data.video import import_video

    typer.echo(str(import_video(source, destination)))


@app.command("split-video")
def split_video(source: Path, destination: Path = ROOT / "data/clips", duration_ms: int = 1000):
    """Extract silent H.264 clips; annotations are created separately."""
    from .data.video import extract_clips

    paths = extract_clips(source, destination, clip_duration_ms=duration_ms)
    typer.echo(f"Extracted {len(paths)} clips to {destination}")


@app.command("annotate")
def annotate(
    video: Path,
    source_id: str,
    start_ms: int,
    end_ms: int,
    event_class: str = "no_event",
    recipient: str = "unknown",
    region: str = "unknown",
    intensity: float = 0.0,
    split: str = "train",
    ambiguous: bool = False,
    viewpoint: str = typer.Option(
        "unknown", help="Camera viewpoint: first_person, third_person, or unknown.",
    ),
    output: Path = ROOT / "data/annotations/custom.jsonl",
):
    """Append one human annotation. Related clips must retain their original source ID."""
    from .data.video import video_metadata
    from .schemas import ClipAnnotation

    video = video.resolve()
    metadata = video_metadata(video)  # Fail before writing a label for an unreadable file.
    if end_ms > metadata["duration_ms"]:
        raise typer.BadParameter("Annotation end exceeds the video duration")
    try:
        video_path = video.relative_to(ROOT).as_posix()
    except ValueError:
        video_path = str(video)
    record = ClipAnnotation(
        video_id=video.stem, video_path=video_path, source_id=source_id, split=split,
        start_ms=start_ms, end_ms=end_ms, event_class=event_class, recipient=recipient,
        body_region=region, intensity=intensity,
        observable_region=region != "unknown", ambiguous=ambiguous, viewpoint=viewpoint,
    )
    if output.exists():
        from .data.annotations import assert_source_disjoint, load_annotations

        assert_source_disjoint([*load_annotations(output), record])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")
    typer.echo(f"Annotation saved: {output}")


@app.command("validate-data")
def validate_data(annotations: Path = ROOT / "data/annotations/example_stream.jsonl"):
    """Validate annotation evidence rules and source-disjoint splits."""
    from .data.annotations import assert_source_disjoint, load_annotations

    records = load_annotations(annotations)
    assert_source_disjoint(records)
    typer.echo(json.dumps({"valid": len(records), "splits": {
        split: sum(r.split == split for r in records) for split in ("train", "val", "test")
    }, "viewpoint_counts": {
        viewpoint: sum(r.viewpoint == viewpoint for r in records)
        for viewpoint in ("first_person", "third_person", "unknown")
    }}, indent=2))


@app.command("train")
def train_command(
    config: Path = ROOT / "configs/baseline.yaml",
    annotations: Path = ROOT / "data/annotations/example_stream.jsonl",
    checkpoint: Path = ROOT / "checkpoints/baseline.pt",
    epochs: int | None = None,
    device: str = "cpu",
):
    """Train temporal layers and heads over genuine pretrained visual features."""
    from .training.train import train

    result = train(config, annotations, checkpoint, epochs=epochs, device=device)
    typer.echo(json.dumps(result, indent=2))


@app.command("validate")
def validate_command(
    checkpoint: Path = ROOT / "checkpoints/baseline.pt",
    annotations: Path = ROOT / "data/annotations/example_stream.jsonl",
    device: str = "cpu",
):
    """Report held-out window metrics; use evaluate-stream for event-level metrics."""
    from .evaluation.validate import validate_checkpoint

    typer.echo(json.dumps(validate_checkpoint(checkpoint, annotations, device), indent=2))


@app.command("export")
def export_command(
    checkpoint: Path = ROOT / "checkpoints/baseline.pt",
    destination: Path = ROOT / "checkpoints/baseline.torchscript.pt",
):
    """Export a self-contained TorchScript model, with metadata alongside it."""
    from .training.export import export_model

    typer.echo(str(export_model(checkpoint, destination)))


@app.command("infer")
def infer_command(
    video: Path = ROOT / "data/raw/demo.mp4",
    checkpoint: Path = ROOT / "checkpoints/baseline.pt",
    output: Path = ROOT / "runs/events.jsonl",
    device: str = "cpu",
    step_ms: int = 125,
    runtime_config: Path | None = None,
    realtime: bool = False,
):
    """Video pixels -> actual model -> temporally gated BodyForceEvent JSONL."""
    from .inference.video import infer_video

    report = infer_video(video, checkpoint, output, device, step_ms, runtime_config, realtime)
    typer.echo(json.dumps(report, indent=2))


@app.command("infer-actions")
def infer_actions_command(
    video: Path = ROOT / "data/raw/demo.mp4",
    output: Path = ROOT / "runs/actions.json",
    config: Path = ROOT / "configs/decisions_actions.yaml",
    max_seconds: float | None = None,
    dry_run: bool = False,
):
    """Label visible actions on an offline video using OpenAI Decisions."""
    from .inference.decisions_actions import infer_actions

    report = infer_actions(video, output, config, max_seconds, dry_run=dry_run)
    typer.echo(json.dumps({"output": str(output), "segments": len(report["segments"]),
                           "windows": len(report["windows"]), "statuses": {
                               status: sum(window["status"] == status for window in report["windows"])
                               for status in sorted({window["status"] for window in report["windows"]})
                           }}, indent=2))

@app.command("infer-local-actions")
@app.command("infer-local-contacts")
def infer_local_actions_command(
    video: Annotated[Path, typer.Option()],
    output: Path = ROOT / "runs/local_actions.json",
    config: Path = ROOT / "configs/local_contacts.yaml",
    max_seconds: float | None = None,
    start_seconds: float = 0,
):
    """Classify video actions using a local vision model, without an API key."""
    from .inference.local_actions import infer_local_actions

    report = infer_local_actions(video, output, config, max_seconds, start_seconds=start_seconds)
    typer.echo(json.dumps({"output": str(output), "model": report["model"],
                           "segments": len(report["segments"]), "windows": len(report["windows"]),
                           "statuses": {status: sum(window["status"] == status for window in report["windows"])
                                        for status in sorted({window["status"] for window in report["windows"]})}},
                          indent=2))


@app.command("evaluate-stream")
def evaluate_stream(
    video: Path = ROOT / "data/raw/demo.mp4",
    checkpoint: Path = ROOT / "checkpoints/baseline.pt",
    annotations: Path = ROOT / "data/annotations/demo.jsonl",
    output: Path = ROOT / "runs/evaluation_events.jsonl",
    device: str = "cpu",
    step_ms: int = 125,
    tolerance_ms: int = 250,
    runtime_config: Path | None = None,
):
    """Infer first, then independently match events to held-out contact annotations."""
    from .data.annotations import load_annotations
    from .evaluation.events import evaluate_events
    from .inference.video import infer_video
    from .schemas import BodyForceEvent

    report = infer_video(video, checkpoint, output, device, step_ms, runtime_config, echo=False)
    events = [BodyForceEvent.model_validate_json(line) for line in output.read_text("utf-8").splitlines() if line.strip()]
    truth = load_annotations(annotations)
    if any((ROOT / a.video_path).resolve() != video.resolve() for a in truth):
        raise typer.BadParameter("All reference annotations must belong to the evaluated video")
    metrics = evaluate_events(events, truth, report["video_duration_ms"], tolerance_ms)
    metrics["profiling"] = report
    destination = output.with_suffix(".metrics.json")
    destination.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(metrics, indent=2))


@app.command("serve-actions")
def serve_actions(
    video: Path,
    timeline: Path,
    port: int = 8001,
):
    """View an action timeline without loading the haptic model or PyTorch."""
    import uvicorn

    from .runtime.actions_server import create_actions_app

    application = create_actions_app(video, timeline)
    typer.echo(f"Open http://127.0.0.1:{port} — standalone action timeline.")
    uvicorn.run(application, host="127.0.0.1", port=port)

@app.command("preview-haptics")
def preview_haptics(
    source: Path,
    output: Path = ROOT / "runs/haptic_preview.json",
    source_kind: str = "contacts",
    duration_ms: float | None = None,
    include_inferred: bool = False,
):
    """Export authored unit envelopes from contacts or BodyForceEvent JSONL."""
    from .runtime.haptic_preview import preview_contacts, preview_events
    from .schemas import BodyForceEvent

    if source.resolve() == output.resolve():
        raise typer.BadParameter("Preview output must not overwrite its source")
    try:
        content = source.read_text(encoding="utf-8")
        if source_kind == "contacts":
            result = preview_contacts(json.loads(content), include_inferred=include_inferred)
        elif source_kind == "events":
            if duration_ms is None:
                raise ValueError("--duration-ms is required for event JSONL")
            events = [BodyForceEvent.model_validate_json(line)
                      for line in content.splitlines() if line.strip()]
            result = preview_events(events, duration_ms)
        else:
            raise ValueError("--source-kind must be contacts or events")
    except (TypeError, ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps({"output": str(output), "cues": len(result.cues),
                           "mode": result.mode, "omitted": result.omitted}, indent=2))


@app.command("serve")
def serve(
    video: Path = ROOT / "data/raw/demo.mp4",
    checkpoint: Path = ROOT / "checkpoints/baseline.pt",
    device: str = "cpu",
    port: int = 8000,
    runtime_config: Path | None = None,
    actions: Path | None = None,
):
    """Serve the built frontend and playback-synchronized inference on localhost."""
    import uvicorn

    from .runtime.server import create_app

    application = create_app(video, checkpoint, device, runtime_config, actions_path=actions)
    typer.echo(f"Open http://127.0.0.1:{port} — video pixels feed the loaded checkpoint.")
    uvicorn.run(application, host="127.0.0.1", port=port)


@app.command("demo")
def demo(
    video: Path | None = None,
    checkpoint: Path = ROOT / "checkpoints/baseline.pt",
    device: str = "cpu",
    port: int = 8000,
    retrain: bool = False,
):
    """One command: prepare synthetic data/checkpoint if needed, build UI, then serve."""
    import os
    import shutil
    import subprocess

    from .schemas import export_schemas

    if video is None:
        from .data.example import create_example_dataset
        create_example_dataset(ROOT)
        video = ROOT / "data/raw/demo.mp4"
    if retrain or not checkpoint.exists():
        if checkpoint.resolve() != (ROOT / "checkpoints/baseline.pt").resolve():
            raise typer.BadParameter("Train the requested custom checkpoint with uhm train first")
        from .data.example import create_example_dataset, create_example_stream_annotations
        from .training.train import train
        create_example_dataset(ROOT)
        annotations = create_example_stream_annotations(ROOT)
        train(ROOT / "configs/baseline.yaml", annotations, checkpoint, device=device)
    export_schemas()
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if npm is None:
        raise typer.BadParameter("Node.js with npm is required to build the frontend")
    frontend = ROOT / "frontend"
    if not (frontend / "node_modules").is_dir():
        subprocess.run([npm, "ci"], cwd=frontend, check=True)
    subprocess.run([npm, "run", "build"], cwd=frontend, check=True)
    serve(video, checkpoint, device, port)
