"""Reproduce the opening-head regression through the actual pixel/runtime path.

All model predictions are completed before loading evaluation annotations. Labels
are never supplied to ModelPredictor or EventAggregator. The fixed fresh sources
are held-out fixtures, not extra training data. This script changes no checkpoint,
model configuration, inference threshold, or original demo asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from uhm.data.annotations import load_annotations, write_annotations
from uhm.data.example import FPS, VISIBLE_REGIONS, Condition, _annotation, _render
from uhm.data.video import video_metadata, write_video
from uhm.evaluation.events import evaluate_events
from uhm.inference.predictor import ModelPredictor
from uhm.ontology import ROOT
from uhm.runtime.aggregation import EventAggregator, load_runtime_config


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fresh_fixture(seed: int, destination: Path) -> tuple[Path, Path]:
    """Reuse fixed held-out fixtures, or generate missing ones identically."""
    destination.mkdir(parents=True, exist_ok=True)
    video = destination / f"seed_{seed}.mp4"
    annotations_path = destination / f"seed_{seed}.jsonl"
    if video.exists() and annotations_path.exists():
        return video, annotations_path
    conditions = []
    for index, region in enumerate(VISIBLE_REGIONS):
        conditions.extend([
            Condition("idle"),
            Condition("player", region, 0.3 if index % 2 == 0 else 0.85),
        ])
    conditions.extend([
        Condition("npc", intensity=0.7), Condition("environment", intensity=0.6),
        Condition("player_occluded", intensity=0.7), Condition("idle"),
    ])
    if not video.exists():
        write_video(video, (
            _render(condition, frame, seed, index)
            for index, condition in enumerate(conditions) for frame in range(FPS)
        ), FPS)
    if not annotations_path.exists():
        annotations = []
        for index, condition in enumerate(conditions):
            start_ms, end_ms = index * 1000, (index + 1) * 1000 - 1
            if condition.kind in {"player", "npc", "environment", "player_occluded"}:
                annotations.append(_annotation(
                    video, f"fresh_seed_{seed}", "test", Condition("idle"),
                    start_ms, start_ms + 416, video_id=f"{seed}_{index}_precontact",
                ))
                start_ms += 417
            annotations.append(_annotation(
                video, f"fresh_seed_{seed}", "test", condition, start_ms, end_ms,
                video_id=f"{seed}_{index}",
            ))
        write_annotations(annotations_path, annotations)
    return video, annotations_path


def summarize_case(case: dict, events: list, annotations: list, duration_ms: int) -> dict:
    """Post-inference scoring; strict negative checks complement fuzzy matching."""
    truth = [row for row in annotations if row.end_ms >= case["start_ms"]]
    metrics = evaluate_events(events, truth, duration_ms, tolerance_ms=250)
    head_truth = next(row for row in truth if row.body_region.value == "head")
    head_strict = [event for event in events if (
        event.body_region.value == "head"
        and event.recipient.value == "player"
        and event.event_class.value == head_truth.event_class.value
        and head_truth.start_ms <= event.timestamp_ms <= head_truth.end_ms
    )]
    head_matches = [match for match in metrics["matches"]
                    if match["reference_id"] == head_truth.video_id
                    and match["predicted_region"] == "head"]
    strict_negative_errors, nonplayer_errors, occluded_errors = [], [], []
    contexts = []
    for event in events:
        active = [row for row in truth if row.start_ms <= event.timestamp_ms <= row.end_ms]
        context = {
            "event": event.model_dump(mode="json"),
            "references_at_timestamp": [row.model_dump(mode="json") for row in active],
        }
        contexts.append(context)
        player_active = any(row.recipient.value == "player"
                            and row.event_class.value != "no_event" for row in active)
        if not player_active:
            strict_negative_errors.append(context)
        if any(row.recipient.value in {"other_character", "environment"} for row in active):
            nonplayer_errors.append(context)
        if (any(row.recipient.value == "player" and not row.observable_region for row in active)
                and event.body_region.value != "unknown"):
            occluded_errors.append(context)
    return {
        "case": case,
        "head": {
            "reference_interval_ms": [head_truth.start_ms, head_truth.end_ms],
            "strict_contact_interval_hit": bool(head_strict),
            "strict_trigger_timestamps_ms": [event.timestamp_ms for event in head_strict],
            "strict_delays_ms": [event.timestamp_ms - head_truth.start_ms for event in head_strict],
            "matched_with_250ms_tolerance": bool(head_matches),
            "matched_trigger_timestamps_ms": [match["prediction_ms"] for match in head_matches],
        },
        "metrics": metrics,
        "all_events": [event.model_dump(mode="json") for event in events],
        "event_reference_contexts": contexts,
        "strict_negative_emissions": strict_negative_errors,
        "nonplayer_contact_emissions": nonplayer_errors,
        "occluded_contact_false_localizations": occluded_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/head_debug/regression")
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--suite", choices=("all", "demo", "fresh"), default="all")
    parser.add_argument("--assert-head", action="store_true",
                        help="Exit 1 if any replay misses its first head contact; preserve reports.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    runtime = load_runtime_config(args.runtime_config)
    checkpoint = args.checkpoint.resolve()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    predictor = ModelPredictor(checkpoint, args.device)
    cases = []
    if args.suite in {"all", "demo"}:
        for offset_ms in (0, 40, 80):
            for start_ms in (0, 1000, 1250):
                cases.append({
                    "name": f"demo_start{start_ms}_offset{offset_ms}",
                    "video": str(ROOT / "data/raw/demo.mp4"),
                    "annotations": str(ROOT / "data/annotations/demo.jsonl"),
                    "start_ms": start_ms, "sampling_offset_ms": offset_ms,
                })
    if args.suite in {"all", "fresh"}:
        for seed in (100100, 100101, 100102):
            video, annotations = fresh_fixture(seed, ROOT / "runs/head_debug/fresh_streams")
            cases.append({
                "name": f"fresh_seed{seed}", "video": str(video),
                "annotations": str(annotations), "start_ms": 0, "sampling_offset_ms": 0,
            })

    # Cache only actual stateless model results from the same video/timestamp.
    # Every replay gets its own fresh EventAggregator and sees timestamps in order.
    # Seek/cold-start follows predict_video: preceding pixels remain available in
    # the recorded file, while aggregator confirmation/cooldown state is reset.
    prediction_cache = {}
    case_events, case_predictions, durations = {}, {}, {}
    for case in cases:
        video = Path(case["video"])
        duration_ms = video_metadata(video)["duration_ms"]
        durations[case["name"]] = duration_ms
        aggregator = EventAggregator(runtime)
        events, predictions = [], []
        start = case["start_ms"] + case["sampling_offset_ms"]
        for timestamp_ms in range(start, duration_ms, 125):
            key = (str(video.resolve()), timestamp_ms)
            if key not in prediction_cache:
                prediction_cache[key] = predictor.predict_video(video, timestamp_ms)[0]
            prediction = prediction_cache[key]
            predictions.append(prediction.model_dump(mode="json"))
            event = aggregator.update(prediction)
            if event is not None:
                events.append(event)
        case_events[case["name"]] = events
        case_predictions[case["name"]] = predictions
        print(f"Inferred {case['name']}: {len(predictions)} samples, {len(events)} events", flush=True)

    # Evaluation annotations are loaded only after every inference replay finishes.
    reports = []
    for case in cases:
        annotations = load_annotations(Path(case["annotations"]))
        report = summarize_case(case, case_events[case["name"]], annotations, durations[case["name"]])
        report["predictions"] = case_predictions[case["name"]]
        write_json(args.output / f"{case['name']}.json", report)
        report.pop("predictions")
        reports.append(report)
        print(json.dumps({
            "case": case["name"], "head": report["head"],
            "counts": report["metrics"]["counts"],
            "strict_negative_emissions": len(report["strict_negative_emissions"]),
            "nonplayer_contact_emissions": len(report["nonplayer_contact_emissions"]),
            "occluded_contact_false_localizations": len(report["occluded_contact_false_localizations"]),
        }), flush=True)
    current_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    summary = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_unchanged_during_run": current_hash == checkpoint_hash,
        "runtime_config": asdict(runtime), "sampling_step_ms": 125,
        "prediction_count_actual_inferences": len(prediction_cache),
        "cold_start_semantics": "Fresh aggregator; same causal past-file pixel access as recorded-video runtime.",
        "annotation_use": "Scoring only, after all inference. No annotation supplied to prediction/aggregation.",
        "all_head_contacts_detected_strictly": all(r["head"]["strict_contact_interval_hit"] for r in reports),
        "strict_negative_emission_count": sum(len(r["strict_negative_emissions"]) for r in reports),
        "nonplayer_contact_emission_count": sum(len(r["nonplayer_contact_emissions"]) for r in reports),
        "occluded_false_localization_count": sum(len(r["occluded_contact_false_localizations"]) for r in reports),
        "cases": reports,
        "limitations": [
            "Synthetic procedural fixtures, not evidence of real-gameplay generalization.",
            "Strict negatives count timestamp-exact interval violations; matching separately allows 250ms tolerance.",
            "Cold starts do not emulate unavailable historical capture frames; they reproduce recorded-file seeking.",
        ],
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"report": str((args.output / "summary.json").resolve())}), flush=True)
    if args.assert_head and not summary["all_head_contacts_detected_strictly"]:
        failed = [report["case"]["name"] for report in reports
                  if not report["head"]["strict_contact_interval_hit"]]
        parser.exit(1, "Head regression failed: " + ", ".join(failed) + "\n")


if __name__ == "__main__":
    main()
