"""Portable full-state checkpoints: inference never needs a weight download."""

from pathlib import Path

import torch

from uhm.ontology import BODY_REGIONS, EVENT_CLASSES, ONTOLOGY, ONTOLOGY_HASH, RECIPIENTS

from .baseline import VisualHapticModel


def save_checkpoint(path: Path, model: VisualHapticModel, metadata: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "metadata": metadata | {
            "model_config": model.config, "ontology": ONTOLOGY, "ontology_hash": ONTOLOGY_HASH,
            "labels": {"event": EVENT_CLASSES, "recipient": RECIPIENTS, "region": BODY_REGIONS},
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(checkpoint: Path, device: str = "cpu") -> tuple[VisualHapticModel, dict]:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    metadata = payload["metadata"]
    expected = {"event": EVENT_CLASSES, "recipient": RECIPIENTS, "region": BODY_REGIONS}
    if metadata.get("ontology_hash") != ONTOLOGY_HASH or metadata.get("labels") != expected:
        raise ValueError("Checkpoint ontology or label ordering does not match this checkout")
    model = VisualHapticModel(metadata["model_config"], pretrained=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, metadata
