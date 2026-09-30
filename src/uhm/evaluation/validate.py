"""Validation uses held-out annotations and the complete saved pixel model."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from uhm.data.factory import build_dataset
from uhm.models.checkpoint import load_checkpoint
from uhm.training.losses import multi_task_loss

from .metrics import window_metrics


@torch.inference_mode()
def evaluate_loader(model, loader, device: str, config: dict) -> dict:
    model.eval()
    collected_outputs: dict[str, list] = {}
    collected_targets: dict[str, list] = {}
    loss_sum, count = 0.0, 0
    for raw_batch in loader:
        batch = {name: tensor.to(device) for name, tensor in raw_batch.items()}
        outputs = (model.forward_features(batch["features"])
                   if "features" in batch else model(batch["frames"]))
        loss, _ = multi_task_loss(outputs, batch, config["training"].get("loss_weights"))
        size = len(batch["event"])
        loss_sum += float(loss) * size
        count += size
        for name, tensor in outputs.items():
            collected_outputs.setdefault(name, []).append(tensor.cpu())
        for name in ("event", "recipient", "region", "intensity", "observable", "ambiguous"):
            collected_targets.setdefault(name, []).append(batch[name].cpu())
    if not count:
        raise ValueError("Validation loader is empty")
    metrics = window_metrics(
        {name: torch.cat(tensors) for name, tensors in collected_outputs.items()},
        {name: torch.cat(tensors) for name, tensors in collected_targets.items()},
        config.get("evaluation", {}),
    )
    return metrics | {"loss": loss_sum / count}


def validate_checkpoint(checkpoint: Path, annotations: Path, device: str = "cpu") -> dict:
    model, metadata = load_checkpoint(checkpoint, device)
    config = metadata["config"]
    torch.set_num_threads(config["training"].get("num_threads", 4))
    dataset = build_dataset(annotations, config, "val")
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], num_workers=0)
    return evaluate_loader(model, loader, device, config) | {
        "split": "val", "all_synthetic": all(row.synthetic for row in dataset.annotations),
        "viewpoints": sorted({getattr(row, "viewpoint", "unknown") for row in dataset.annotations}),
        "checkpoint": str(Path(checkpoint).resolve()),
        "limitation": "Synthetic validation demonstrates pipeline behavior, not gameplay generalization.",
    }
