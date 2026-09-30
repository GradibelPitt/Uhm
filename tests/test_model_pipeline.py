"""Structural unit fixtures use random encoders; real pretrained training is separate.

These tests never describe random unit-fixture weights as a trained checkpoint.
The actual supplied pretrained run is recorded in runs/training_metrics.json.
"""

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from uhm.models.baseline import VisualHapticModel
from uhm.models.checkpoint import load_checkpoint, save_checkpoint
from uhm.ontology import BODY_REGIONS, EVENT_CLASSES, RECIPIENTS
from uhm.training.export import export_model
from uhm.training.features import cache_features
from uhm.training.losses import multi_task_loss, targets


def _model(temporal="gru", freeze=True):
    torch.set_num_threads(4)
    torch.manual_seed(71)
    return VisualHapticModel({"hidden_size": 16, "spatial_size": 2,
                              "dropout": 0.0, "temporal": temporal,
                              "freeze_encoder": freeze}, pretrained=False)


def _batch():
    return {
        "frames": torch.randn(2, 3, 3, 64, 64),
        "event": torch.tensor([EVENT_CLASSES.index("impact"), EVENT_CLASSES.index("no_event")]),
        "recipient": torch.tensor([RECIPIENTS.index("player"), RECIPIENTS.index("unknown")]),
        "region": torch.tensor([BODY_REGIONS.index("left_torso"), BODY_REGIONS.index("unknown")]),
        "intensity": torch.tensor([0.7, 0.0]),
        "observable": torch.tensor([True, False]), "ambiguous": torch.tensor([False, False]),
    }


def _metadata(model):
    return {"test_fixture_random_encoder": True, "config": {
        "model": model.config,
        "data": {"num_frames": 3, "image_size": 64, "window_ms": 1000},
        "training": {"num_threads": 4},
    }}


def test_gradient_step_preserves_frozen_encoder_and_checkpoint_predictions(tmp_path):
    model, batch = _model(), _batch()
    encoder_before = {name: value.clone() for name, value in model.encoder.state_dict().items()}
    head_before = model.heads["event"].weight.detach().clone()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    model.train()
    assert not model.encoder.training
    loss, _ = multi_task_loss(model(batch["frames"]), batch)
    loss.backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    optimizer.step()
    assert not torch.equal(head_before, model.heads["event"].weight)
    for name, value in model.encoder.state_dict().items():
        torch.testing.assert_close(value, encoder_before[name], rtol=0, atol=0)
    model.eval()
    expected = model.predict(batch["frames"])
    checkpoint = save_checkpoint(tmp_path / "random-unit-fixture.pt", model, _metadata(model))
    restored, metadata = load_checkpoint(checkpoint)
    assert metadata["test_fixture_random_encoder"] is True
    for name, tensor in restored.predict(batch["frames"]).items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


def test_unknown_is_supervised_for_unobservable_ambiguous_and_non_player():
    region = BODY_REGIONS.index("left_arm")
    batch = {
        "event": torch.tensor([EVENT_CLASSES.index("impact")] * 4),
        "recipient": torch.tensor([RECIPIENTS.index("player"), RECIPIENTS.index("player"),
                                   RECIPIENTS.index("other_character"), RECIPIENTS.index("player")]),
        "region": torch.tensor([region] * 4),
        "intensity": torch.tensor([0.5] * 4),
        "observable": torch.tensor([False, True, True, True]),
        "ambiguous": torch.tensor([False, True, False, False]),
    }
    target = targets(batch)
    assert target["region"].tolist() == [BODY_REGIONS.index("unknown")] * 3 + [region]
    assert target["observable"].tolist() == [False, False, False, True]
    logits = torch.zeros(4, len(BODY_REGIONS), requires_grad=True)
    outputs = {
        "event": torch.zeros(4, len(EVENT_CLASSES), requires_grad=True),
        "recipient": torch.zeros(4, len(RECIPIENTS), requires_grad=True),
        "region": logits, "intensity": torch.full((4,), 0.5, requires_grad=True),
        "observable": torch.zeros(4, requires_grad=True),
    }
    loss, _ = multi_task_loss(outputs, batch)
    loss.backward()
    # Unknown examples actively push the unknown logit up, rather than being masked.
    assert (logits.grad[:3, BODY_REGIONS.index("unknown")] < 0).all()
    assert logits.grad[3, region] < 0


def test_feature_cache_rejects_trainable_encoder_or_frozen_augmentation():
    with pytest.raises(ValueError, match="frozen encoder"):
        cache_features(_model(freeze=False), SimpleNamespace(augment=False), 2, "cpu")
    with pytest.raises(ValueError, match="augmentation"):
        cache_features(_model(), SimpleNamespace(augment=True), 2, "cpu")


@pytest.mark.parametrize("temporal", ["gru", "mean", "conv"])
def test_temporal_replacement_preserves_prediction_contract(temporal):
    output = _model(temporal).predict(torch.randn(2, 3, 3, 64, 64))
    for key, count in (("event_probs", len(EVENT_CLASSES)),
                       ("recipient_probs", len(RECIPIENTS)), ("region_probs", len(BODY_REGIONS))):
        assert output[key].shape == (2, count)
        torch.testing.assert_close(output[key].sum(-1), torch.ones(2))
    assert output["intensity"].shape == output["observable_prob"].shape == (2,)


def test_checkpoint_rejects_ontology_mismatch(tmp_path):
    model = _model()
    checkpoint = save_checkpoint(tmp_path / "fixture.pt", model, _metadata(model))
    payload = torch.load(checkpoint, weights_only=True)
    payload["metadata"]["ontology_hash"] = "different-ontology"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="ontology"):
        load_checkpoint(checkpoint)


def test_exported_full_model_matches_reloaded_checkpoint(tmp_path):
    model = _model()
    checkpoint = save_checkpoint(tmp_path / "fixture.pt", model, _metadata(model))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        exported = export_model(checkpoint, tmp_path / "fixture.torchscript.pt")
        executable = torch.jit.load(exported)
    probe = torch.randn(2, 3, 3, 64, 64)
    expected = model.predict(probe)
    with torch.inference_mode():
        actual = executable(probe)
    for name, result in zip(("event_probs", "recipient_probs", "region_probs", "intensity", "observable_prob"), actual, strict=True):
        torch.testing.assert_close(expected[name], result, rtol=1e-4, atol=1e-5)
    sidecar = json.loads(Path(str(exported) + ".json").read_text())
    assert sidecar["test_fixture_random_encoder"] is True
    assert sidecar["export_numerical_parity_passed"] is True
