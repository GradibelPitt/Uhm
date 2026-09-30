"""Small appearance variations that preserve time and anatomical left/right."""

import torch

from .video import IMAGENET_MEAN, IMAGENET_STD


def augment_video_clip(
    frames: torch.Tensor, *, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Transform normalized [T,3,H,W] frames with one shared set of parameters.

    Translate by at most six percent along either axis, replicating edge pixels;
    apply mild brightness, contrast, and per-channel offsets. There is no spatial
    reflection, temporal resampling, or access to any frames outside this input.
    A CPU generator can be supplied for repeatable views without changing global
    random state. Labels and anatomical side assignments remain unchanged.
    """
    if frames.ndim != 4 or frames.shape[1] != 3 or min(frames.shape) <= 0:
        raise ValueError("Expected a nonempty [T,3,H,W] video clip")
    if not frames.is_floating_point():
        raise ValueError("Expected ImageNet-normalized floating-point frames")
    _, _, height, width = frames.shape
    # Draw once per clip, never independently per frame. Pixel-index translation
    # avoids interpolation and replicates borders instead of wrapping content.
    limits = (round(width * 0.06), round(height * 0.06))
    shift_x = int(torch.randint(-limits[0], limits[0] + 1, (), generator=generator))
    shift_y = int(torch.randint(-limits[1], limits[1] + 1, (), generator=generator))
    parameters = torch.rand(5, generator=generator).to(device=frames.device, dtype=frames.dtype)
    brightness, contrast = 0.9 + parameters[:2] * 0.2
    offsets = ((parameters[2:] - 0.5) * 0.05).view(1, 3, 1, 1)
    x = (torch.arange(width, device=frames.device) - shift_x).clamp(0, width - 1)
    y = (torch.arange(height, device=frames.device) - shift_y).clamp(0, height - 1)
    mean = IMAGENET_MEAN.to(device=frames.device, dtype=frames.dtype)
    std = IMAGENET_STD.to(device=frames.device, dtype=frames.dtype)
    rgb = frames.index_select(-2, y).index_select(-1, x) * std + mean
    rgb = (((rgb - 0.5) * contrast + 0.5) * brightness + offsets).clamp(0, 1)
    return (rgb - mean) / std
