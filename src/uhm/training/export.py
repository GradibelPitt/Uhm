"""Full visual model export, including frozen pretrained encoder parameters."""

import json
from pathlib import Path

import torch
from torch import nn

from uhm.models.checkpoint import load_checkpoint


class _ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frames):
        outputs = self.model(frames)
        return (outputs["event"].softmax(-1), outputs["recipient"].softmax(-1),
                outputs["region"].softmax(-1), outputs["intensity"],
                outputs["observable"].sigmoid())


def export_model(checkpoint: Path, destination: Path) -> Path:
    model, metadata = load_checkpoint(checkpoint, "cpu")
    torch.set_num_threads(metadata["config"]["training"].get("num_threads", 4))
    data = metadata["config"]["data"]
    sample = torch.zeros(1, data["num_frames"], 3, data["image_size"], data["image_size"])
    wrapper = _ExportWrapper(model).eval()
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, sample, strict=True)
        probe = torch.randn_like(sample)
        expected, actual = wrapper(probe), traced(probe)
        for reference, exported in zip(expected, actual, strict=True):
            torch.testing.assert_close(reference, exported, rtol=1e-4, atol=1e-5)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, destination)
    sidecar = metadata | {
        "export_format": "TorchScript", "input": "ImageNet-normalized float32 [B,T,3,H,W]",
        "traced_input_shape": list(sample.shape),
        "outputs": ["event_probs", "recipient_probs", "region_probs", "intensity", "observable_prob"],
        "export_numerical_parity_passed": True,
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(sidecar, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return destination
