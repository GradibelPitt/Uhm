import pytest

from uhm.evaluation.events import evaluate_events
from uhm.schemas import BodyForceEvent, ClipAnnotation


def reference(start=1000, end=1400, region="head", **changes):
    values = {
        "video_id": f"ref-{start}", "video_path": "recording.mp4", "source_id": "session", "split": "test",
        "start_ms": start, "end_ms": end, "event_class": "impact", "recipient": "player", "body_region": region,
        "intensity": 0.5, "observable_region": region != "unknown", "synthetic": True,
    }
    return ClipAnnotation(**(values | changes))


def event(time=1100, region="head", event_id="event-1", **changes):
    values = {
        "event_id": event_id, "timestamp_ms": time, "event_class": "impact", "recipient": "player",
        "body_region": region, "intensity": 0.5, "duration_ms": 220, "direction": "unknown",
        "confidence": {"event": 0.9, "recipient": 0.9, "region": 0.9, "intensity": 0},
    }
    return BodyForceEvent(**(values | changes))


def test_matching_is_one_to_one_and_duplicate_emissions_are_false_events():
    result = evaluate_events(
        [event(1100), event(1200, event_id="duplicate-impact")],
        [reference(), reference(3000, 3400)], duration_ms=4000,
    )
    assert result["counts"]["matched_events"] == 1
    assert result["counts"]["false_events"] == 1
    assert result["counts"]["missed_events"] == 1
    assert result["counts"]["duplicate_emissions"] == 1
    assert result["event_precision"] == 0.5
    assert result["event_recall"] == 0.5


def test_adjacent_events_maximize_valid_matches_before_minimizing_onset_distance():
    # The first prediction can match either event, but the second can only match
    # the later event. Greedy nearest-onset assignment would lose a valid match.
    result = evaluate_events(
        [event(1190), event(1500, event_id="event-2")],
        [reference(1000, 1100), reference(1200, 1300)],
        duration_ms=2000,
        tolerance_ms=250,
    )
    assert result["counts"]["matched_events"] == 2
    assert result["counts"]["false_events"] == 0
    assert result["counts"]["missed_events"] == 0
    assert {(match["event_id"], match["reference_id"]) for match in result["matches"]} == {
        ("event-1", "ref-1000"), ("event-2", "ref-1200"),
    }


def test_repeated_delivery_of_same_event_id_is_not_a_second_emission():
    delivery = event()
    result = evaluate_events([delivery, delivery], [reference()], duration_ms=2000)
    assert result["counts"]["predicted_player_events"] == 1
    assert result["counts"]["repeated_event_id_deliveries"] == 1
    assert result["counts"]["false_events"] == 0


def test_always_abstaining_has_zero_coverage_and_undefined_localized_accuracy():
    result = evaluate_events([event(region="unknown")], [reference()], duration_ms=2000)
    assert result["body_region_coverage_matched_observable"] == 0
    assert result["body_region_coverage_all_observable"] == 0
    assert result["body_region_accuracy_localized_observable"] is None
    assert result["body_region_accuracy_including_abstentions"] == 0


def test_known_region_on_unobservable_event_counts_as_false_localization():
    result = evaluate_events([event(region="head")], [reference(region="unknown")], duration_ms=2000)
    assert result["false_localization_rate_unobservable"] == 1
    assert result["counts"]["false_localizations"] == 1
    assert result["body_region_accuracy_localized_observable"] is None


def test_negative_only_video_uses_full_duration_for_false_events_per_minute():
    negative = reference(0, 30000, region="unknown", event_class="no_event", recipient="unknown", intensity=0)
    result = evaluate_events([event(5000), event(20000, event_id="event-2")], [negative], duration_ms=30000)
    assert result["false_events_per_minute"] == 4
    assert result["event_precision"] == 0
    assert result["event_recall"] is None


@pytest.mark.parametrize("changes", [{"source_id": "another-session"}, {"video_path": "another-video.mp4"}])
def test_different_source_media_clocks_cannot_be_pooled(changes):
    with pytest.raises(ValueError, match="do not mix media clocks"):
        evaluate_events([], [reference(), reference(3000, 3400, **changes)], duration_ms=4000)


def test_temporal_match_requires_same_event_class():
    result = evaluate_events([event(event_class="collision")], [reference()], duration_ms=2000)
    assert result["counts"]["matched_events"] == 0
    assert result["counts"]["false_events"] == 1
    assert result["counts"]["missed_events"] == 1


def test_joint_region_recall_exposes_missed_head_despite_perfect_localized_accuracy():
    result = evaluate_events(
        [event(3100, region="left_torso")],
        [reference(), reference(3000, 3400, region="left_torso")],
        duration_ms=4000,
    )
    assert result["body_region_accuracy_localized_observable"] == 1
    assert result["body_region_joint_recall_by_region"] == {
        "head": {
            "reference_events": 1, "detected_events": 0,
            "correctly_localized_events": 0, "joint_recall": 0,
        },
        "left_torso": {
            "reference_events": 1, "detected_events": 1,
            "correctly_localized_events": 1, "joint_recall": 1,
        },
    }


def test_joint_region_recall_excludes_unknown_and_non_player_reference_locations():
    result = evaluate_events(
        [event(region="unknown"), event(3100, event_id="head-hit")],
        [reference(region="unknown"), reference(3000, 3400),
         reference(5000, 5400, region="left_arm", recipient="other_character")],
        duration_ms=6000,
    )
    assert result["body_region_joint_recall_by_region"] == {
        "head": {
            "reference_events": 1, "detected_events": 1,
            "correctly_localized_events": 1, "joint_recall": 1,
        },
    }
    assert evaluate_events([], [reference(region="unknown")], duration_ms=2000)[
        "body_region_joint_recall_by_region"
    ] == {}


@pytest.mark.parametrize("predicted_region", ["left_arm", "unknown"])
def test_joint_region_recall_counts_wrong_location_and_abstention_as_failure(predicted_region):
    result = evaluate_events([event(region=predicted_region)], [reference()], duration_ms=2000)
    assert result["event_recall"] == 1
    assert result["body_region_joint_recall_by_region"]["head"] == {
        "reference_events": 1, "detected_events": 1,
        "correctly_localized_events": 0, "joint_recall": 0,
    }
