import pytest

from uhm.runtime import EventAggregator, RuntimeConfig
from uhm.schemas import Prediction


def prediction(time, **changes):
    values = {
        "timestamp_ms": time, "event_class": "impact", "recipient": "player", "body_region": "left_torso",
        "intensity": 0.6, "confidence": {"event": 0.9, "recipient": 0.9, "region": 0.9, "intensity": 0},
        "observable_probability": 0.9,
    }
    return Prediction(**(values | changes))


def negative(time):
    return prediction(time, event_class="no_event", recipient="unknown", body_region="unknown", intensity=0)


def test_one_episode_emits_once_despite_repeated_class_and_region_jitter():
    aggregator = EventAggregator(RuntimeConfig())
    events = []
    for time in range(0, 1100, 100):
        event = aggregator.update(prediction(time, event_class="collision" if time > 300 else "impact",
                                              body_region="head" if time > 300 else "left_torso"))
        if event is not None:
            events.append(event)
    assert len(events) == 1
    assert events[0].body_region.value == "left_torso"
    assert events[0].direction.value == "unknown"
    assert events[0].confidence.intensity == 0


@pytest.mark.parametrize("recipient", ["other_character", "environment", "unknown"])
def test_nonplayer_impacts_never_create_haptic_events(recipient):
    aggregator = EventAggregator(RuntimeConfig())
    assert all(aggregator.update(prediction(time, recipient=recipient)) is None for time in range(0, 1000, 100))


@pytest.mark.parametrize("second", [
    {"body_region": "right_torso"},
    {"body_region": "unknown"},
    {"observable_probability": 0.2},
    {"confidence": {"event": 0.9, "recipient": 0.9, "region": 0.2, "intensity": 0}},
])
def test_location_disagreement_or_weak_evidence_abstains_without_suppressing_event(second):
    aggregator = EventAggregator(RuntimeConfig())
    assert aggregator.update(prediction(0)) is None
    event = aggregator.update(prediction(100, **second))
    assert event is not None
    assert event.body_region.value == "unknown"


def test_sustained_negative_release_and_cooldown_require_fresh_confirmation():
    aggregator = EventAggregator(RuntimeConfig())
    aggregator.update(prediction(0))
    assert aggregator.update(prediction(100)) is not None
    # A short dropout does not re-arm a continuous impact.
    assert aggregator.update(negative(200)) is None
    assert aggregator.update(prediction(300)) is None
    for time in (400, 600, 700):
        assert aggregator.update(negative(time)) is None
    # Release at700ms, then cooldown through1050ms.
    assert aggregator.update(prediction(800)) is None
    assert aggregator.update(prediction(1000)) is None
    assert aggregator.update(prediction(1050)) is None
    assert aggregator.update(prediction(1150)) is not None


@pytest.mark.parametrize("next_time", [50, 1000])
def test_backward_seek_or_large_gap_clears_latch_and_cooldown(next_time):
    aggregator = EventAggregator(RuntimeConfig())
    aggregator.update(prediction(0))
    assert aggregator.update(prediction(100)) is not None
    assert aggregator.update(prediction(next_time)) is None
    assert aggregator.update(prediction(next_time + 100)) is not None


def test_repeated_media_timestamp_does_not_count_as_confirmation():
    aggregator = EventAggregator(RuntimeConfig())
    assert aggregator.update(prediction(100)) is None
    assert aggregator.update(prediction(100)) is None
    assert aggregator.update(prediction(200)) is not None


def test_nonplayer_breaks_confirmation_and_intensity_history():
    aggregator = EventAggregator(RuntimeConfig())
    aggregator.update(prediction(0, intensity=0.9))
    aggregator.update(prediction(50, recipient="other_character", intensity=1))
    assert aggregator.update(prediction(100, intensity=0.2)) is None
    event = aggregator.update(prediction(150, intensity=0.4))
    assert event.intensity == pytest.approx(0.3)


@pytest.mark.parametrize("settings", [
    {"confirm_samples": True}, {"release_ms": -1}, {"duration_ms": 5001},
    {"smoothing_alpha": float("nan")}, {"recipient_threshold": 1.1},
])
def test_runtime_configuration_rejects_invalid_policy(settings):
    with pytest.raises(ValueError):
        RuntimeConfig(**settings)
