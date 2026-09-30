"""In-memory frozen feature cache, rebuilt from the current video pixels per run."""

import torch
from torch.utils.data import DataLoader, Dataset

from uhm.data.augmentation import augment_video_clip


class CachedFeatures(Dataset):
    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors["features"])

    def __getitem__(self, index):
        return {name: tensor[index] for name, tensor in self.tensors.items()}


@torch.inference_mode()
def cache_features(
    model, dataset, batch_size: int, device: str, *, augmentation_views: int = 1,
) -> CachedFeatures:
    """Cache original features plus explicitly requested fixed augmented views.

    augmentation_views is the total number of views: 1 means original pixels
    only; 4 means the original and three random augmented views per sample. The
    views are sampled once while building this cache and reused on every epoch,
    not resampled online. Seed torch before calling to reproduce the views.
    Dataset augmentation must be disabled to keep this distinction explicit.
    Validation/test annotations may only use the unchanged one-view path.
    """
    if not model.config["freeze_encoder"]:
        raise ValueError("Feature caching requires a frozen encoder")
    if dataset.augment:
        raise ValueError(
            "Feature caching would freeze online augmentation; set dataset augment=False "
            "and request fixed cached augmentation_views explicitly"
        )
    if (not isinstance(augmentation_views, int) or isinstance(augmentation_views, bool)
            or augmentation_views < 1):
        raise ValueError("augmentation_views must be an integer >= 1")
    if augmentation_views > 1 and any(
        row.split != "train" for row in getattr(dataset, "annotations", [])
    ):
        raise ValueError("Validation/test feature caches must use augmentation_views=1")
    model.eval()
    chunks: dict[str, list[torch.Tensor]] = {}
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        frames = batch.pop("frames")
        for view in range(augmentation_views):
            view_frames = frames if view == 0 else torch.stack([
                augment_video_clip(clip) for clip in frames
            ])
            features = model.encode_frames(view_frames.to(device)).cpu()
            chunks.setdefault("features", []).append(features)
            for name, tensor in batch.items():
                chunks.setdefault(name, []).append(tensor)
    # clone outside inference mode is done by DataLoader collation before training.
    return CachedFeatures({name: torch.cat(values) for name, values in chunks.items()})
