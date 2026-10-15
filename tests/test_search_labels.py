import json
import random

import pytest
from catanatron.models.enums import RESOURCES, ActionType
from catanatron.serialization import action_to_json, state_to_json
from catanatron.state_functions import player_key

from catanzero.dataset import observation, ordered_actions
from catanzero.search_labels import SearchConfig, evaluate_actions, label_job, paired_preferences, rollout, wilson_interval
from catanzero.simulation import SimulationConfig, create_game


def test_counterfactual_search_is_reproducible_and_preserves_environment():
    game = create_game(SimulationConfig(), 52)
    before = state_to_json(game)
    action = ordered_actions(game)[0]
    config = SearchConfig(rollouts=2, candidates=2, max_actions=2, screening_beliefs=1)
    first = evaluate_actions(game, action, config, 991)
    second = evaluate_actions(game, action, config, 991)
    first.pop("elapsed_seconds")
    second.pop("elapsed_seconds")
    assert first == second
    assert state_to_json(game) == before
    assert first["selected_win_rate"] is None
    assert first["quality_score"] is None
    assert not first["ranking_resolved"]
    assert all(c["truncated"] == 2 for c in first["candidates"])
    assert all(c["win_rate_bounds"] == [0, 1] for c in first["candidates"])


def test_actual_hidden_hands_and_future_rng_do_not_change_search():
    game = create_game(SimulationConfig(), 81)
    actor = game.state.current_color()
    enemies = [c for c in game.state.colors if c != actor]
    key, other_key = (player_key(game.state, c) for c in enemies[:2])
    game.state.player_state[f"{key}_WOOD_IN_HAND"] = 2
    game.state.player_state[f"{other_key}_ORE_IN_HAND"] = 2
    game.state.resource_freqdeck[RESOURCES.index("WOOD")] -= 2
    game.state.resource_freqdeck[RESOURCES.index("ORE")] -= 2
    changed = game.copy()
    changed.state.player_state[f"{key}_WOOD_IN_HAND"] = 0
    changed.state.player_state[f"{key}_ORE_IN_HAND"] = 2
    changed.state.player_state[f"{other_key}_ORE_IN_HAND"] = 0
    changed.state.player_state[f"{other_key}_WOOD_IN_HAND"] = 2
    changed.state.development_listdeck.reverse()
    changed.random = changed.state.random = random.Random(98912)
    assert observation(game, actor) == observation(changed, actor)
    config = SearchConfig(rollouts=2, candidates=2, max_actions=3, screening_beliefs=1)
    selected = ordered_actions(game)[0]
    first = evaluate_actions(game, selected, config, 123)
    second = evaluate_actions(changed, selected, config, 123)
    first.pop("elapsed_seconds")
    second.pop("elapsed_seconds")
    assert first == second


def test_immediate_winning_move_gets_terminal_win_not_censoring():
    game = create_game(SimulationConfig(), 42)
    action = ordered_actions(game)[0]
    assert action.action_type == ActionType.BUILD_SETTLEMENT
    key = player_key(game.state, action.color)
    game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] = 9
    game.state.player_state[f"{key}_VICTORY_POINTS"] = 9
    outcome, steps = rollout(game, action, 20, SearchConfig(max_actions=1))
    assert (outcome, steps) == (1, 1)
    assert game.winning_color() is None


def test_output_keeps_old_targets_and_does_not_promote_censored_estimates():
    game = create_game(SimulationConfig(), 42)
    actions = ordered_actions(game)
    row = {"inputs": {"action": action_to_json(actions[0])},
           "targets": {"terminal_outcome": 1.0, "weighted_score": 0.72}}
    output = label_job((state_to_json(game), row, SearchConfig(rollouts=1, candidates=2, max_actions=1, screening_beliefs=1), 20, "test"))
    assert output["inputs"] == row["inputs"]
    assert output["targets"]["weighted_score"] == 0.72
    assert output["targets"]["search_weighted_score"] is None
    assert output["source_split"] == "test"
    assert not output["search_ranking_eligible"]


def test_intervals_and_invalid_search_budgets():
    low, high = wilson_interval(16, 32)
    assert low < 0.5 < high
    assert high - low > 0.2
    assert wilson_interval(0, 0) == [0, 1]
    with pytest.raises(ValueError):
        SearchConfig(rollouts=0)
    with pytest.raises(ValueError):
        SearchConfig(candidates=1)
    with pytest.raises(ValueError):
        SearchConfig(max_seconds=float("inf"))


def test_time_budget_keeps_unknown_trials_unknown():
    game = create_game(SimulationConfig(), 52)
    action = ordered_actions(game)[0]
    config = SearchConfig(rollouts=3, candidates=2, max_seconds=1e-9, screening_beliefs=1)
    result = evaluate_actions(game, action, config, 22)
    assert result["time_budget_exhausted"]
    assert result["quality_score"] is None
    assert not result["ranking_resolved"]
    assert all(c["not_attempted"] == c["truncated"] == 3 for c in result["candidates"])
    assert all(c["losses"] == c["wins"] == c["rollouts_attempted"] == 0 for c in result["candidates"])


def test_paired_preferences_detect_a_bad_move_without_a_unique_best():
    actions = ordered_actions(create_game(SimulationConfig(), 42))[:3]
    good = [1] * 20 + [0] * 44
    bad = [0] * 20 + [1] * 5 + [0] * 39
    pairs = paired_preferences(actions, [good, good, bad])
    assert sum(p["resolved"] for p in pairs) == 2
    assert not pairs[0]["resolved"]
    assert all(p["paired_one_sided_p_value"] < p["corrected_threshold"] for p in pairs if p["resolved"])
    censored = paired_preferences(actions[:2], [good, [None] * 64])
    assert censored == []
