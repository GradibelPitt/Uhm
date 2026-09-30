import pytest
from pydantic import ValidationError

from uhm.schemas import BodyForceEvent, ClipAnnotation


def force_event(**changes):
    values = {
        "event_id": "evt-test", "timestamp_ms": 100, "event_class": "impact", "recipient": "player",
        "body_region": "unknown", "intensity": 0.5, "duration_ms": 220, "direction": "unknown",
        "confidence": {"event": 0.9, "recipient": 0.8, "region": 0.0, "intensity": 0.0},
    }
    return BodyForceEvent(**(values | changes))


def annotation(**changes):
    values = {
        "video_id": "clip", "video_path": "sample.mp4", "source_id": "source", "split": "train",
        "start_ms": 0, "end_ms": 1000, "event_class": "impact", "recipient": "player",
        "body_region": "unknown", "intensity": 0.5, "observable_region": False, "ambiguous": False,
    }
    return ClipAnnotation(**(values | changes))


def test_unknown_is_a_valid_first_class_event_and_annotation():
    event = force_event()
    assert BodyForceEvent.model_validate_json(event.model_dump_json()) == event
    assert annotation(ambiguous=True).body_region.value == "unknown"
    assert force_event(recipient="unknown").recipient.value == "unknown"


@pytest.mark.parametrize("intensity", [-0.01, 1.01, float("nan"), float("inf")])
def test_salience_cannot_be_out_of_range_or_nonfinite(intensity):
    with pytest.raises(ValidationError):
        force_event(intensity=intensity)


@pytest.mark.parametrize("changes", [
    {"body_region": "head", "observable_region": False},
    {"body_region": "head", "observable_region": True, "ambiguous": True},
    {"body_region": "unknown", "observable_region": True},
    {"event_class": "no_event", "recipient": "player", "intensity": 0},
    {"start_ms": 1000, "end_ms": 1000},
])
def test_annotations_reject_fabricated_locations_and_inconsistent_labels(changes):
    with pytest.raises(ValidationError):
        annotation(**changes)


def test_no_event_is_a_training_label_not_an_emittable_force_event():
    assert annotation(event_class="no_event", recipient="unknown", intensity=0).intensity == 0
    with pytest.raises(ValidationError):
        force_event(event_class="no_event")
    with pytest.raises(ValidationError):
        force_event(duration_ms=0)
