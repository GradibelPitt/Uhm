from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from uhm.data.augmentation import augment_video_clip
from uhm.data.video import IMAGENET_MEAN, IMAGENET_STD
from uhm.training.features import cache_features


def normalized_gradient():
    coordinates = torch.linspace(0.25, 0.55, 32)
    frame = torch.stack((coordinates[None, :].expand(32, 32),
                         coordinates[:, None].expand(32, 32), torch.full((32, 32), 0.4)))
    rgb = torch.stack((frame, frame, frame + 0.05))
    return (rgb - IMAGENET_MEAN) / IMAGENET_STD


def test_clip_transform_keeps_time_and_space_synchronous_without_reflecting_sides():
    original = normalized_gradient()
    transformed = augment_video_clip(original, generator=torch.Generator().manual_seed(7))
    rgb = transformed * IMAGENET_STD + IMAGENET_MEAN
    assert transformed.shape == original.shape
    assert not torch.equal(original, transformed)
    # Identical observations remain identical instead of independent frame jitter.
    assert torch.equal(transformed[0], transformed[1])
    # A uniform temporal brightness change stays uniform across all pixels. No
    # frame gets a different translation, color transform, or temporal neighbor.
    difference = rgb[2] - rgb[1]
    torch.testing.assert_close(difference, difference.mean().expand_as(difference))
    assert difference.min() > 0
    # The coordinate ramps retain their direction: anatomical sides never flip.
    assert (rgb[0, 0, :, 1:] - rgb[0, 0, :, :-1]).min() >= 0
    assert (rgb[0, 1, 1:, :] - rgb[0, 1, :-1, :]).min() >= 0
    # This seeded horizontal translation repeats the border rather than wrapping
    # the opposite side into view or introducing black padding.
    assert torch.equal(rgb[0, 0, :, -1], rgb[0, 0, :, -2])
    assert rgb.min() > 0.1
    torch.testing.assert_close(original, normalized_gradient())


def test_seeded_transform_is_reproducible():
    first = augment_video_clip(normalized_gradient(), generator=torch.Generator().manual_seed(31))
    second = augment_video_clip(normalized_gradient(), generator=torch.Generator().manual_seed(31))
    assert torch.equal(first, second)


class TwoClips(Dataset):
    augment = False

    def __init__(self, split="train"):
        self.annotations = [SimpleNamespace(split=split), SimpleNamespace(split=split)]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"frames": normalized_gradient() + index * 0.03,
                "event": torch.tensor(index), "region": torch.tensor(2 + index),
                "intensity": torch.tensor(index * 0.25), "annotation_index": torch.tensor(index)}


class IdentityEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {"freeze_encoder": True}

    def encode_frames(self, frames):
        return frames.flatten(2)


def test_cache_contains_original_and_multiple_fixed_views_with_identical_targets():
    torch.manual_seed(102)
    dataset = TwoClips()
    cached = cache_features(IdentityEncoder(), dataset, 2, "cpu", augmentation_views=4)
    assert len(cached) == 8
    for source_index in range(2):
        rows = [cached[index] for index in range(len(cached))
                if cached[index]["annotation_index"].item() == source_index]
        assert len(rows) == 4
        sample = dataset[source_index]
        for key in ("event", "region", "intensity", "annotation_index"):
            assert all(torch.equal(row[key], sample[key]) for row in rows)
        assert torch.equal(rows[0]["features"], sample["frames"].flatten(1))
        assert all(not torch.equal(rows[left]["features"], rows[right]["features"])
                   for left in range(4) for right in range(left + 1, 4))
    # Cached augmentations are explicit fixed views, not re-drawn per access.
    assert torch.equal(cached[4]["features"], cached[4]["features"])


def test_validation_cache_is_unaugmented_and_rejects_requested_augmented_views(monkeypatch):
    def forbidden(_):
        raise AssertionError("Validation pixels must not be augmented")

    monkeypatch.setattr("uhm.training.features.augment_video_clip", forbidden)
    dataset = TwoClips(split="val")
    cached = cache_features(IdentityEncoder(), dataset, 2, "cpu")
    assert len(cached) == len(dataset)
    for index in range(len(dataset)):
        assert torch.equal(cached[index]["features"], dataset[index]["frames"].flatten(1))
    with pytest.raises(ValueError, match="Validation/test"):
        cache_features(IdentityEncoder(), dataset, 2, "cpu", augmentation_views=4)


@pytest.mark.parametrize("views", [0, -1, 1.5, True])
def test_cache_requires_positive_integer_view_count(views):
    with pytest.raises(ValueError, match="integer >= 1"):
        cache_features(IdentityEncoder(), TwoClips(), 2, "cpu", augmentation_views=views)
