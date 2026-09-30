"""Recorded-video inference and reproducible event/diagnostic JSONL output."""

import json
from pathlib import Path
from time import perf_counter, sleep

from uhm.data.video import video_metadata
from uhm.runtime.aggregation import EventAggregator, load_runtime_config

from .predictor import ModelPredictor
from .profiling import Profiler


def infer_video(
    video: Path, checkpoint: Path, output: Path, device: str = "cpu",
    step_ms: int = 125, runtime_config: Path | None = None,
    realtime: bool = False, echo: bool = True,
) -> dict:
    if step_ms <= 0:
        raise ValueError("step_ms must be positive")
    metadata = video_metadata(video)
    predictor = ModelPredictor(checkpoint, device)
    aggregator = EventAggregator(load_runtime_config(runtime_config))
    profiler = Profiler(device, predictor.data_config["num_frames"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output.with_suffix(".predictions.jsonl")
    events = 0
    start = perf_counter()
    with output.open("w", encoding="utf-8") as stream, predictions_path.open("w", encoding="utf-8") as diagnostics:
        for timestamp in range(0, metadata["duration_ms"], step_ms):
            if realtime:
                sleep(max(0, start + timestamp / 1000 - perf_counter()))
            prediction, timing = predictor.predict_video(video, timestamp)
            event = aggregator.update(prediction)
            profiler.record(timing, prediction)
            diagnostics.write(json.dumps({"prediction": prediction.model_dump(mode="json"), "timing": timing}) + "\n")
            if event:
                line = event.model_dump_json()
                stream.write(line + "\n")
                stream.flush()
                events += 1
                if echo:
                    print(line, flush=True)
    report = profiler.summary() | {
        "video": str(Path(video).resolve()), "checkpoint": str(Path(checkpoint).resolve()),
        "video_duration_ms": metadata["duration_ms"], "source_fps": metadata["fps"],
        "sample_step_ms": step_ms, "context_window_ms": predictor.data_config["window_ms"],
        "events": events, "event_log": str(output), "prediction_log": str(predictions_path),
        "training_data_synthetic": predictor.metadata.get("all_synthetic", predictor.metadata.get("synthetic")),
    }
    output.with_suffix(".profile.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
