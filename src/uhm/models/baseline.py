"""Small pretrained spatial encoder with trainable temporal, multi-task heads."""

from collections.abc import Callable

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from uhm.ontology import BODY_REGIONS, EVENT_CLASSES, RECIPIENTS

BACKBONES: dict[str, Callable] = {}
TEMPORAL_MODELS: dict[str, Callable] = {}
PRETRAINED_ID = "torchvision.MobileNet_V3_Small_Weights.IMAGENET1K_V1"


def register_backbone(name: str, factory: Callable) -> None:
    """Factory(pretrained, spatial_size) -> encoder with output_dim attribute."""
    if name in BACKBONES:
        raise ValueError(f"Backbone already registered: {name}")
    BACKBONES[name] = factory


def register_temporal(name: str, factory: Callable) -> None:
    """Factory(hidden_size) -> module mapping [B,T,H] to [B,H]."""
    if name in TEMPORAL_MODELS:
        raise ValueError(f"Temporal model already registered: {name}")
    TEMPORAL_MODELS[name] = factory


class MobileNetSpatial(nn.Module):
    def __init__(self, pretrained: bool, spatial_size: int = 4, final_features: bool = False):
        super().__init__()
        # Training intentionally raises on missing/download-failed weights. Checkpoint
        # loading uses pretrained=False then restores every saved encoder parameter.
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        network = mobilenet_v3_small(weights=weights)
        self.features = network.features if final_features else network.features[:9]
        channels = 576 if final_features else 48
        self.pool = nn.AdaptiveAvgPool2d((spatial_size, spatial_size))
        self.output_dim = channels * spatial_size * spatial_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Retain coarse spatial location; global pooling discards useful contact cues.
        return self.pool(self.features(images)).flatten(1)


class GRUAggregation(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.gru(features)[0][:, -1]


class MeanAggregation(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=1)


class ConvAggregation(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.pad = nn.ConstantPad1d((2, 0), 0)
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(self.pad(features.transpose(1, 2))))[:, :, -1]


register_backbone("mobilenet_v3_small_spatial", MobileNetSpatial)
register_backbone(
    "mobilenet_v3_small_final",
    lambda pretrained, spatial_size: MobileNetSpatial(pretrained, spatial_size, True),
)
register_temporal("gru", GRUAggregation)
register_temporal("mean", MeanAggregation)
register_temporal("conv", ConvAggregation)


class VisualHapticModel(nn.Module):
    def __init__(self, config: dict | None = None, *, pretrained: bool = True):
        super().__init__()
        self.config = {
            "backbone": "mobilenet_v3_small_spatial", "spatial_size": 4,
            "temporal": "gru", "hidden_size": 128, "dropout": 0.05,
            "freeze_encoder": True,
        } | (config or {})
        self.encoder = BACKBONES[self.config["backbone"]](
            pretrained, self.config["spatial_size"]
        )
        hidden = self.config["hidden_size"]
        self.project = nn.Sequential(
            nn.LayerNorm(self.encoder.output_dim),
            nn.Linear(self.encoder.output_dim, hidden), nn.GELU(),
        )
        self.temporal = TEMPORAL_MODELS[self.config["temporal"]](hidden)
        self.shared = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(self.config["dropout"]))
        self.heads = nn.ModuleDict({
            "event": nn.Linear(hidden, len(EVENT_CLASSES)),
            "recipient": nn.Linear(hidden, len(RECIPIENTS)),
            "region": nn.Linear(hidden, len(BODY_REGIONS)),
            "intensity": nn.Linear(hidden, 1),
            "observable": nn.Linear(hidden, 1),
        })
        if self.config["freeze_encoder"]:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config["freeze_encoder"]:
            self.encoder.eval()
        return self

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        batch, length, channels, height, width = frames.shape
        encoded = self.encoder(frames.reshape(batch * length, channels, height, width))
        return encoded.reshape(batch, length, -1)

    def forward_features(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(self.temporal(self.project(features)))
        outputs = {name: head(shared) for name, head in self.heads.items()}
        outputs["intensity"] = outputs["intensity"].squeeze(-1).sigmoid()
        outputs["observable"] = outputs["observable"].squeeze(-1)
        return outputs

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_features(self.encode_frames(frames))

    @torch.inference_mode()
    def predict(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        outputs = self(frames)
        return {
            "event_probs": outputs["event"].softmax(-1),
            "recipient_probs": outputs["recipient"].softmax(-1),
            "region_probs": outputs["region"].softmax(-1),
            "intensity": outputs["intensity"],
            "observable_prob": outputs["observable"].sigmoid(),
        }
