import pytest

from catanzero.online_linear import LinearActionValues, shaped_transition


def test_linear_values_normalize_clip_and_have_one_bias():
    model = {"weights": [2, -1], "intercept": .3, "feature_clip": 4}
    scaler = {"feature_names": ["a", "b"], "fit_split": "train", "offset": [1, 2], "scale": [2, 1]}
    q = LinearActionValues(model, scaler)
    assert q.normalize([3, 100]) == [1, 4]
    assert q.predict([3, 100]) == pytest.approx(-1.7)
    with pytest.raises(ValueError):
        q.predict([float("nan"), 1])


def test_potential_shaping_telescopes_and_does_not_reward_points_in_a_loss():
    a = {"action_index": 0, "learner_color": "RED", "vp": 2}
    b = {"action_index": 3, "learner_color": "RED", "vp": 5}
    c = {"action_index": 7, "learner_color": "RED", "vp": 9}
    r1, d1 = shaped_transition(a, b, terminal=False, won=None, gamma=.9)
    r2, d2 = shaped_transition(b, c, terminal=True, won=False, gamma=.9)
    assert d1 == pytest.approx(.9**3) and d2 == 0
    assert r1+d1*r2 == pytest.approx(-.05*a["vp"])
    win_r, _ = shaped_transition(b, c, terminal=True, won=True, gamma=.9)
    assert win_r-r2 == pytest.approx(.9**3)


def test_unknown_outcome_and_cross_player_transition_are_rejected():
    a = {"action_index": 0, "learner_color": "RED", "vp": 0}
    b = {"action_index": 1, "learner_color": "BLUE", "vp": 0}
    with pytest.raises(ValueError, match="crossed"):
        shaped_transition(a, b, terminal=False, won=None)
    b["learner_color"] = "RED"
    with pytest.raises(ValueError, match="known"):
        shaped_transition(a, b, terminal=True, won=None)
