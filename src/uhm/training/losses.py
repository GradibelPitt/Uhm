"""Explicit unknown supervision and masked perceptual-intensity regression."""

import torch
from torch.nn import functional as F

from uhm.ontology import BODY_REGIONS, EVENT_CLASSES, RECIPIENTS


def targets(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    player_event = ((batch["event"] != EVENT_CLASSES.index("no_event"))
                    & (batch["recipient"] == RECIPIENTS.index("player")))
    observable = player_event & batch["observable"].bool() & ~batch["ambiguous"].bool()
    region = batch["region"].clone()
    region[~observable] = BODY_REGIONS.index("unknown")
    return {"player_event": player_event, "observable": observable, "region": region}


def multi_task_loss(outputs: dict, batch: dict, weights: dict | None = None) -> tuple:
    target = targets(batch)
    intensity_mask = target["player_event"] & ~batch["ambiguous"].bool()
    intensity = (F.mse_loss(outputs["intensity"][intensity_mask], batch["intensity"][intensity_mask])
                 if intensity_mask.any() else outputs["intensity"].sum() * 0)
    losses = {
        "event": F.cross_entropy(outputs["event"], batch["event"]),
        "recipient": F.cross_entropy(outputs["recipient"], batch["recipient"]),
        "region": F.cross_entropy(outputs["region"], target["region"]),
        "intensity": intensity,
        "observable": F.binary_cross_entropy_with_logits(
            outputs["observable"], target["observable"].float()
        ),
    }
    multipliers = {"event": 1.0, "recipient": 1.0, "region": 2.0,
                   "intensity": 3.0, "observable": 0.5} | (weights or {})
    total = sum(multipliers[name] * loss for name, loss in losses.items())
    return total, losses
