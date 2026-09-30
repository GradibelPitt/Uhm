"""Reproducible supervised training with a real pretrained visual backbone."""

import hashlib
import json
import random
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader

from uhm.data.factory import build_dataset
from uhm.evaluation.validate import evaluate_loader
from uhm.models.baseline import PRETRAINED_ID, VisualHapticModel
from uhm.models.checkpoint import load_checkpoint, save_checkpoint
from uhm.ontology import ROOT

from .features import cache_features
from .losses import multi_task_loss


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train(
    config_path: Path, annotations: Path, checkpoint: Path,
    epochs: int | None = None, device: str = "cpu",
) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    settings = config["training"]
    if epochs is not None:
        settings["epochs"] = epochs
    if settings["epochs"] <= 0 or settings["batch_size"] <= 0:
        raise ValueError("epochs and batch_size must be positive")
    seed = settings.get("seed", 1729)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(settings.get("num_threads", 4))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.perf_counter()
    augmentation_views = settings.get("augmentation_views", 1)
    if augmentation_views != 1 and not settings.get("cache_features", True):
        raise ValueError("augmentation_views requires feature caching; use augment for online training")
    training_data = build_dataset(
        annotations, config, "train", augment=settings.get("augment", False),
    )
    validation_data = build_dataset(annotations, config, "val")
    all_records = training_data.annotations + validation_data.annotations
    synthetic = all(row.synthetic for row in all_records)
    print("Training on SYNTHETIC pipeline examples; no gameplay generalization claim."
          if synthetic else "Training on supplied annotated videos; inspect data provenance.", flush=True)
    model = VisualHapticModel(config["model"], pretrained=True).to(device)
    config["model"] = model.config
    initial_parameters = {name: p.detach().cpu().clone()
                          for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=settings["learning_rate"], weight_decay=settings.get("weight_decay", 0.0001),
    )
    batch_size = settings["batch_size"]
    feature_started = time.perf_counter()
    if settings.get("cache_features", True):
        print("Caching pretrained spatial features from training and validation video pixels...", flush=True)
        train_input = cache_features(
            model, training_data, batch_size, device, augmentation_views=augmentation_views,
        )
        val_input = cache_features(model, validation_data, batch_size, device)
    else:
        train_input, val_input = training_data, validation_data
    feature_seconds = time.perf_counter() - feature_started
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_input, batch_size=batch_size, shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(val_input, batch_size=batch_size, shuffle=False, num_workers=0)
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config, "pretrained_weights": PRETRAINED_ID,
        "preprocessing": config["data"] | {
            "color_space": "RGB", "pixel_range_before_normalization": [0, 1],
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "resize": "OpenCV INTER_AREA square", "causal": True,
        },
        "synthetic": synthetic,
        "provenance": {
            "annotations": str(Path(annotations).resolve()),
            "annotations_sha256": hashlib.sha256(Path(annotations).read_bytes()).hexdigest(),
            "train_count": len(training_data), "validation_count": len(validation_data),
            "cached_training_views": augmentation_views,
            "optimization_sample_count": len(train_input),
            "supervision": ("current timestamp of causal trailing window"
                            if config["data"].get("sampling") == "causal"
                            else "weak whole-clip presence"),
            "train_sources": sorted({row.source_id for row in training_data.annotations}),
            "validation_sources": sorted({row.source_id for row in validation_data.annotations}),
            "viewpoints": sorted({getattr(row, "viewpoint", "unknown") for row in all_records}),
            "sources_by_viewpoint": {
                split: {
                    viewpoint: sorted({row.source_id for row in dataset.annotations
                                       if getattr(row, "viewpoint", "unknown") == viewpoint})
                    for viewpoint in sorted({getattr(row, "viewpoint", "unknown")
                                             for row in dataset.annotations})
                } for split, dataset in (("train", training_data), ("val", validation_data))
            },
            "train_event_counts": dict(Counter(row.event_class.value for row in training_data.annotations)),
            "supported_event_classes": sorted({row.event_class.value for row in training_data.annotations}),
            "synthetic_record_count": sum(row.synthetic for row in all_records),
        },
        "versions": {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__)},
        "confidence_calibrated": False,
        "intensity_confidence": "Unavailable; the runtime uses 0.0 rather than invented certainty.",
        "limitations": [
            "Synthetic examples exercise the software pipeline, not general gameplay understanding.",
            "Train/validation sources differ but share a procedural renderer and visual conventions.",
            ("Synthetic frame-derived contact labels supervise causal window endpoints."
             if config["data"].get("sampling") == "causal"
             else "Weak clip labels supervise window presence, not precise contact onset."),
            "Cached augmentation uses a fixed finite set of views, not fresh draws every epoch.",
            "Perceptual intensity is an illustrative annotation target, not physical force.",
        ] if synthetic else ["Performance applies only to the supplied annotation distribution."],
    }
    best_loss = float("inf")
    history = []
    best_metrics = None
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total_loss, samples = 0.0, 0
        for raw_batch in train_loader:
            batch = {name: tensor.to(device) for name, tensor in raw_batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = (model.forward_features(batch["features"])
                       if "features" in batch else model(batch["frames"]))
            loss, _ = multi_task_loss(outputs, batch, settings.get("loss_weights"))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch["event"])
            samples += len(batch["event"])
        validation = evaluate_loader(model, val_loader, device, config)
        history.append({"epoch": epoch, "train_loss": total_loss / samples, "validation": validation})
        if validation["loss"] < best_loss:
            best_loss, best_metrics = validation["loss"], validation
            metadata["best_epoch"] = epoch
            metadata["validation"] = validation
            save_checkpoint(checkpoint, model, metadata)
        if epoch == 1 or epoch % 5 == 0 or epoch == settings["epochs"]:
            print(f"epoch {epoch:02d}/{settings['epochs']} "
                  f"train_loss={total_loss / samples:.4f} val_loss={validation['loss']:.4f} "
                  f"player_window_recall={validation['player_window_recall']} "
                  f"region_coverage={validation['body_region_coverage']}", flush=True)
        _write_json(ROOT / "runs/training_history.json", history)
    reloaded, saved_metadata = load_checkpoint(checkpoint, device)
    squared_change = sum(float((p.detach().cpu() - initial_parameters[name]).square().sum())
                         for name, p in reloaded.named_parameters() if name in initial_parameters)
    if squared_change <= 0:
        raise RuntimeError("Training did not change any trainable parameter")
    # Exercise pixel decoding and the complete reloaded model, not just cached heads.
    sample = validation_data[0]["frames"].unsqueeze(0).to(device)
    with torch.inference_mode():
        prediction = reloaded.predict(sample)
        if not all(torch.isfinite(tensor).all() for tensor in prediction.values()):
            raise FloatingPointError("Reloaded checkpoint produced non-finite predictions")
    result = {
        "checkpoint": str(Path(checkpoint).resolve()), "synthetic": synthetic,
        "epochs_run": settings["epochs"], "best_epoch": saved_metadata["best_epoch"],
        "training_seconds": time.perf_counter() - started,
        "feature_cache_seconds": feature_seconds,
        "trainable_parameter_change_l2": squared_change ** 0.5,
        "trainable_parameter_count": sum(p.numel() for p in reloaded.parameters() if p.requires_grad),
        "full_checkpoint_reload_inference_passed": True,
        "validation": best_metrics,
        "note": "Window metrics only; event matching and streaming latency are evaluated separately.",
    }
    saved_metadata["training_summary"] = result
    save_checkpoint(checkpoint, reloaded, saved_metadata)
    _write_json(ROOT / "runs/training_metrics.json", result)
    return result
