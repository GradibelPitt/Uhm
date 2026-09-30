"""Player-event metrics, distinct from clip classification diagnostics.

Matching maximizes one-to-one matches, then minimizes onset distance, with the
same class and recipient. Unknown locations count as abstentions.
"""

from collections import Counter
from statistics import mean, median

import networkx as nx

from uhm.schemas import BodyForceEvent, ClipAnnotation


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_events(
    events: list[BodyForceEvent],
    annotations: list[ClipAnnotation],
    duration_ms: int,
    tolerance_ms: int = 250,
) -> dict:
    """Evaluate one complete source video, including its negative footage.

    A match is eligible between [reference start - tolerance, reference end +
    tolerance]. Detection delay is measured against annotated contact onset in
    media time. It is NOT the end-to-end wall-clock-to-display latency.
    """
    if duration_ms <= 0 or tolerance_ms < 0:
        raise ValueError("duration_ms must be positive and tolerance_ms nonnegative")
    if len({(a.source_id, a.video_path) for a in annotations}) > 1:
        raise ValueError("Evaluate one source video at a time; do not mix media clocks")
    if any(a.end_ms > duration_ms for a in annotations):
        raise ValueError("Annotation exceeds the evaluated video duration")
    if any(e.timestamp_ms > duration_ms for e in events):
        raise ValueError("Prediction exceeds the evaluated video duration")

    # Repeated deliveries of one event ID denote the same semantic event.
    unique = {event.event_id: event for event in reversed(events)}
    predicted = sorted(
        (e for e in unique.values() if e.recipient.value == "player"),
        key=lambda event: event.timestamp_ms,
    )
    reference = sorted(
        (a for a in annotations if a.recipient.value == "player"
         and a.event_class.value != "no_event"), key=lambda a: a.start_ms,
    )

    def eligible(prediction, truth):
        return (prediction.event_class.value == truth.event_class.value
                and truth.start_ms - tolerance_ms <= prediction.timestamp_ms
                <= truth.end_ms + tolerance_ms)

    # Greedy nearest-onset matching can steal the only eligible reference from
    # a later prediction. Match each connected temporal group globally instead.
    graph = nx.Graph()
    for p_index, prediction in enumerate(predicted):
        for a_index, truth in enumerate(reference):
            if eligible(prediction, truth):
                graph.add_edge(("p", p_index), ("a", a_index),
                               weight=-abs(prediction.timestamp_ms - truth.start_ms))
    assignment = {}
    for component in nx.connected_components(graph):
        for left, right in nx.max_weight_matching(graph.subgraph(component), maxcardinality=True):
            p_node, a_node = (left, right) if left[0] == "p" else (right, left)
            assignment[p_node[1]] = a_node[1]
    matches = [(predicted[p], reference[a]) for p, a in sorted(assignment.items())]
    unmatched = [prediction for index, prediction in enumerate(predicted) if index not in assignment]
    remaining = set(range(len(reference))) - set(assignment.values())

    observable = [(p, a) for p, a in matches if a.observable_region and not a.ambiguous]
    localized = [(p, a) for p, a in observable if p.body_region.value != "unknown"]
    unobservable = [(p, a) for p, a in matches if not a.observable_region or a.ambiguous]
    correct = sum(p.body_region.value == a.body_region.value for p, a in localized)
    false_localizations = sum(p.body_region.value != "unknown" for p, _ in unobservable)
    observable_reference = sum(a.observable_region and not a.ambiguous for a in reference)
    # Joint recall includes missed events in each true region's denominator.
    # Conditional localization accuracy above cannot expose a completely missed
    # region. Unknown/ambiguous references have no localization target and are
    # excluded; an unknown or incorrect prediction on a known target still fails.
    def localizable(truth):
        return truth.observable_region and not truth.ambiguous and truth.body_region.value != "unknown"

    region_reference = Counter(a.body_region.value for a in reference if localizable(a))
    region_detected = Counter(a.body_region.value for _, a in matches if localizable(a))
    region_correct = Counter(
        a.body_region.value for p, a in matches
        if localizable(a) and p.body_region.value == a.body_region.value
    )
    joint_recall_by_region = {
        region: {
            "reference_events": total,
            "detected_events": region_detected[region],
            "correctly_localized_events": region_correct[region],
            "joint_recall": region_correct[region] / total,
        }
        for region, total in sorted(region_reference.items())
    }
    duplicate_emissions = sum(any(eligible(p, a) for _, a in matches) for p in unmatched)
    delays = [p.timestamp_ms - a.start_ms for p, a in matches]
    errors = [abs(p.intensity - a.intensity) for p, a in matches]
    counts = {
        "predicted_player_events": len(predicted), "reference_player_events": len(reference),
        "matched_events": len(matches), "false_events": len(unmatched),
        "missed_events": len(remaining), "observable_reference_events": observable_reference,
        "observable_matched_events": len(observable), "localized_observable_matches": len(localized),
        "correct_localizations": correct, "unobservable_matched_events": len(unobservable),
        "false_localizations": false_localizations, "duplicate_emissions": duplicate_emissions,
        "repeated_event_id_deliveries": len(events) - len(unique),
        "non_player_emissions": sum(e.recipient.value != "player" for e in unique.values()),
    }
    return {
        "scope": "player physical events in one full video",
        "synthetic": all(a.synthetic for a in annotations) if annotations else None,
        "duration_ms": duration_ms, "matching_tolerance_ms": tolerance_ms,
        "matching_rule": "Maximum-cardinality one-to-one, then minimum onset distance; same class/recipient; interval +/- tolerance",
        "counts": counts,
        "event_precision": _ratio(len(matches), len(predicted)),
        "event_recall": _ratio(len(matches), len(reference)),
        "false_events_per_minute": len(unmatched) * 60000 / duration_ms,
        "body_region_accuracy_localized_observable": _ratio(correct, len(localized)),
        "body_region_coverage_matched_observable": _ratio(len(localized), len(observable)),
        "body_region_coverage_all_observable": _ratio(len(localized), observable_reference),
        "body_region_accuracy_including_abstentions": _ratio(correct, len(observable)),
        "body_region_joint_recall_by_region": joint_recall_by_region,
        "false_localization_rate_unobservable": _ratio(false_localizations, len(unobservable)),
        "duplicate_events_per_reference": _ratio(duplicate_emissions, len(reference)),
        "intensity_mae_matched": mean(errors) if errors else None,
        "detection_delay_media_ms": {
            "median": median(delays) if delays else None,
            "values": delays,
            "note": "Prediction media timestamp minus contact onset. Negative is early; not display latency.",
        },
        "matches": [{"event_id": p.event_id, "reference_id": a.video_id,
                     "prediction_ms": p.timestamp_ms, "onset_ms": a.start_ms,
                     "predicted_region": p.body_region.value, "true_region": a.body_region.value}
                    for p, a in matches],
        "limitations": ["Synthetic results do not establish gameplay generalization.",
                        "Null ratios have no denominator and do not mean zero error.",
                        "Confidence and salience are not calibrated to human tactile judgments."],
    }
