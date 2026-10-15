from copy import deepcopy

import pytest
from catanatron.serialization import action_to_json, state_to_json

from catanzero.dataset import observation, ordered_actions
from catanzero.tournament import make_match
from catanzero.tournament_dataset import convert_match, multibelief_label, seed_splits


def recorded_prefix(seed=90100, steps=4):
    game, candidate = make_match("AB:2", seed, 0)
    for _ in range(steps):
        game.execute(ordered_actions(game)[0])
    metadata = {"bot": "AB:2", "seed": seed, "candidate_seat": 0,
                "candidate_color": candidate.value, "game_index": 0,
                "winner": None, "termination": "action_limit", "actions": steps,
                "match_directory": "fixture"}
    return state_to_json(game), metadata


def test_recorded_rows_have_preaction_inputs_and_verified_replay():
    document, metadata = recorded_prefix()
    rows, result, _ = convert_match(document, metadata, 8, "test", 2)
    initial, _ = make_match("AB:2", metadata["seed"], 0)
    assert rows[0]["inputs"]["observation"] == observation(initial, initial.state.current_color())
    assert result["replay_verified"]
    assert result["actions"] == 4
    assert all(r["game_index"] == r["inputs"]["map_ref"] == 8 for r in rows)
    assert all(r["split"] == "test" for r in rows)
    assert all(r["inputs"]["action"] in r["inputs"]["legal_actions"] for r in rows)
    assert all(r["targets"]["terminal_outcome"] is None and not r["train_eligible"] for r in rows)


def test_corrupt_authoritative_game_is_rejected():
    document, metadata = recorded_prefix(steps=1)
    damaged = deepcopy(document)
    damaged["num_turns"] += 1
    with pytest.raises(ValueError, match="replay mismatch"):
        convert_match(damaged, metadata, 0, "train", 1)


def test_same_seed_never_crosses_splits_between_bot_groups():
    seeds = list(range(90000, 90012)) * 3
    splits = seed_splits(reversed(seeds))
    assert len(splits) == 12
    assert sum(v == "train" for v in splits.values()) == 8
    assert sum(v == "validation" for v in splits.values()) == 2
    assert sum(v == "test" for v in splits.values()) == 2
    assert splits == seed_splits(seeds)


def test_multibelief_label_averages_before_computing_regret(monkeypatch):
    game, _ = make_match("AB:2", 90, 0)
    actions = ordered_actions(game)[:2]
    # Different sampled beliefs prefer opposite actions. Averaging their
    # separate regrets would penalize action 0 despite its higher mean value.
    values = iter([10.0, 0.0, 0.0, 8.0])
    monkeypatch.setattr("catanzero.tournament_dataset.PilotHeuristic.expected_value",
                        lambda *_: next(values))
    before = state_to_json(game)
    rng = game.state.random.getstate()
    label = multibelief_label(game, actions, actions[0], 2, 11)
    assert label["selected_value"] == 5.0
    assert label["best_value"] == 5.0
    assert label["heuristic_score"] == 1.0
    assert label["candidate_values"][1]["mean"] == 4.0
    assert label["best_action"] == action_to_json(actions[0])
    assert state_to_json(game) == before
    assert game.state.random.getstate() == rng
