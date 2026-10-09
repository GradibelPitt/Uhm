import json

import pytest

from uhm.inference.pov_motion import parse_pov_response

ACTIONS = ['foot_support', 'foot_push_off', 'foot_landing', 'cushioning', 'hand_load']


def estimate(**changes):
    item = {'phase': 'landing', 'feet': 'both', 'basis': 'inferred', 'hand_load': 'no',
            'cues': ['surface_approach', 'deceleration'], 'evidence_frames': [0, 2]}
    item.update(changes)
    response = {'message': {'content': json.dumps({'slots': [item]})}}
    return parse_pov_response(response, [.25, .5, .75], [[.5, 1]], ACTIONS)[0]


def test_hidden_feet_can_be_inferred_from_a_landing_sequence():
    window = estimate()
    assert window['labels']['foot_landing']['state'] == 'positive'
    assert window['labels']['foot_support']['basis'] == 'inferred'
    assert window['motion']['feet'] == 'both'
    assert window['labels']['foot_support']['probability'] is None


def test_landing_does_not_imply_both_feet():
    window = estimate(feet='unknown')
    assert window['motion']['phase'] == 'landing'
    assert window['motion']['feet'] == 'unknown'
    assert window['labels']['foot_landing']['state'] == 'positive'
    assert window['labels']['foot_support']['state'] == 'unknown'


@pytest.mark.parametrize('changes', [
    {'cues': ['camera_turn']},
    {'cues': ['cut', 'surface_approach', 'deceleration']},
    {'evidence_frames': [0, 0]},
    {'evidence_frames': [True, 2]},
    {'evidence_frames': [0, 3]},
    {'feet': 'none'},
])
def test_camera_turns_cuts_invalid_evidence_and_contradictions_do_not_trigger_landing(changes):
    window = estimate(**changes)
    assert window['motion']['phase'] == 'unknown'
    assert window['labels']['foot_landing']['state'] == 'unknown'


def test_airborne_clears_ground_support():
    window = estimate(phase='airborne', feet='none', cues=['flight'])
    assert window['labels']['foot_support']['state'] == 'negative'
    assert window['labels']['foot_landing']['state'] == 'negative'


def test_running_stride_is_not_a_jump_landing():
    window = estimate(phase='running', feet='alternating', cues=['forward_flow', 'stride_bob'])
    assert window['motion']['phase'] == 'running'
    assert window['labels']['foot_landing']['state'] == 'negative'


def test_missing_or_truncated_results_abstain():
    for response in ({}, {'done_reason': 'length'}, {'message': {'content': '{"slots":[]}'}}):
        assert parse_pov_response(response, [], [[0, .5]], ACTIONS)[0]['status'] == 'invalid_response'


@pytest.mark.parametrize('field', ['phase', 'feet', 'basis', 'hand_load'])
def test_malformed_categories_do_not_crash_analysis(field):
    assert estimate(**{field: []})['status'] == 'invalid_response'
