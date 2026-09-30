"""Window-level diagnostics. These are not temporal event precision or recall."""

import torch

from uhm.ontology import BODY_REGIONS, EVENT_CLASSES, RECIPIENTS
from uhm.training.losses import targets


def _ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def window_metrics(outputs: dict[str, torch.Tensor], batch: dict, thresholds: dict) -> dict:
    event_probs = outputs["event"].softmax(-1)
    recipient_probs = outputs["recipient"].softmax(-1)
    region_probs = outputs["region"].softmax(-1)
    event_conf, event_pred = event_probs.max(-1)
    recipient_conf, recipient_pred = recipient_probs.max(-1)
    region_conf, region_pred = region_probs.max(-1)
    target = targets(batch)
    emitted = ((event_pred != EVENT_CLASSES.index("no_event"))
               & (recipient_pred == RECIPIENTS.index("player"))
               & (event_conf >= thresholds.get("event_threshold", 0.60))
               & (recipient_conf >= thresholds.get("recipient_threshold", 0.60)))
    localized = (emitted & (region_pred != BODY_REGIONS.index("unknown"))
                 & (region_conf >= thresholds.get("region_threshold", 0.55))
                 & (outputs["observable"].sigmoid()
                    >= thresholds.get("observable_threshold", 0.65)))
    true_player = target["player_event"]
    visible_player = target["observable"]
    invisible_player = true_player & ~visible_player
    tp = int((emitted & true_player).sum())
    fp = int((emitted & ~true_player).sum())
    fn = int((~emitted & true_player).sum())
    localized_visible = localized & visible_player
    localized_count = int(localized_visible.sum())
    intensity_mask = true_player & ~batch["ambiguous"].bool()
    return {
        "metric_unit": "annotated video window (not temporal events)",
        "sample_count": len(event_pred),
        "event_class_accuracy": float((event_pred == batch["event"]).float().mean()),
        "recipient_accuracy": float((recipient_pred == batch["recipient"]).float().mean()),
        "player_window_precision": _ratio(tp, tp + fp),
        "player_window_recall": _ratio(tp, tp + fn),
        "player_window_tp": tp, "player_window_fp": fp, "player_window_fn": fn,
        "observable_player_windows": int(visible_player.sum()),
        "unobservable_player_windows": int(invisible_player.sum()),
        "localized_observable_windows": localized_count,
        "body_region_accuracy": _ratio(
            int(((region_pred == target["region"]) & localized_visible).sum()), localized_count
        ),
        "body_region_coverage": _ratio(localized_count, int(visible_player.sum())),
        "false_localization_rate": _ratio(
            int((localized & invisible_player).sum()), int(invisible_player.sum())
        ),
        "intensity_mae_player_windows": (
            float((outputs["intensity"][intensity_mask] - batch["intensity"][intensity_mask]).abs().mean())
            if intensity_mask.any() else None
        ),
        "observable_accuracy": float((
            (outputs["observable"].sigmoid() >= thresholds.get("observable_threshold", 0.65))
            == visible_player
        ).float().mean()),
        "thresholds": thresholds,
        "confidence_calibrated": False,
        "intensity_confidence": "unavailable: runtime confidence.intensity=0",
    }
