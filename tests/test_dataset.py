import gzip
import json

import pytest

from catanzero.dataset import (
    PilotHeuristic, belief_game, collect_game, main, observation, ordered_actions,
)
from catanzero.scoring import ScoreConfig
from catanzero.simulation import SimulationConfig, create_game
from catanatron.models.enums import RESOURCES
from catanatron.state_functions import player_key


def test_private_opponent_hands_and_future_stream_do_not_affect_inputs_or_labels():
    game = create_game(SimulationConfig(), 81)
    actor = game.state.current_color()
    enemy = next(c for c in game.state.colors if c != actor)
    other = next(c for c in game.state.colors if c not in (actor, enemy))
    key = player_key(game.state, enemy)
    other_key = player_key(game.state, other)
    game.state.player_state[f"{key}_WOOD_IN_HAND"] = 3
    game.state.player_state[f"{other_key}_ORE_IN_HAND"] = 3
    game.state.resource_freqdeck[0] -= 3
    game.state.resource_freqdeck[4] -= 3
    changed = game.copy()
    changed.state.player_state[f"{key}_WOOD_IN_HAND"] = 0
    changed.state.player_state[f"{key}_ORE_IN_HAND"] = 3
    changed.state.player_state[f"{other_key}_ORE_IN_HAND"] = 0
    changed.state.player_state[f"{other_key}_WOOD_IN_HAND"] = 3
    changed.state.development_listdeck.reverse()
    changed.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] = 5
    assert observation(game, actor) == observation(changed, actor)
    proxy1 = belief_game(game, actor, 123)
    proxy2 = belief_game(changed, actor, 123)
    assert proxy1.state.player_state == proxy2.state.player_state
    assert proxy1.state.development_listdeck == proxy2.state.development_listdeck
    assert proxy1.state.resource_freqdeck == game.state.resource_freqdeck
    evaluator = PilotHeuristic(game)
    actions = ordered_actions(game)
    assert evaluator.label(proxy1, actions, actions[0]) == evaluator.label(proxy2, actions, actions[0])
    serialized = json.dumps(observation(game, actor))
    assert "random_state" not in serialized
    for player in observation(game, actor)["players"]:
        if player["color"] != actor.value:
            assert "resources" not in player and "development_cards" not in player
            assert "actual_vp" not in player


def test_counterfactual_labels_never_advance_actual_rng_or_mutate_game():
    game = create_game(SimulationConfig(), 17)
    before_rng = game.state.random.getstate()
    before = observation(game, game.state.current_color())
    actions = ordered_actions(game)
    proxy = belief_game(game, game.state.current_color(), 999)
    evaluator = PilotHeuristic(game)
    scores = [evaluator.expected_value(proxy, a) for a in actions]
    best = actions[max(range(len(scores)), key=scores.__getitem__)]
    assert evaluator.label(proxy, actions, best)["heuristic_score"] == pytest.approx(1.0)
    assert game.state.random.getstate() == before_rng
    assert observation(game, game.state.current_color()) == before
    assert len(game.state.action_records) == 0


def test_completed_game_has_actor_correct_labels_and_legal_choices():
    game, rows, result = collect_game(42, 0, ["W", "W"], 1000, 100_000, ScoreConfig())
    assert result["termination"] == "victory"
    assert game.winning_color().value == result["winner"]
    assert result["eligible_rows"] > 0
    for row in rows:
        assert row["inputs"]["action"] in row["inputs"]["legal_actions"]
        assert row["targets"]["terminal_outcome"] == float(row["inputs"]["observation"]["actor"] == result["winner"])
        assert row["has_choice"] == (len(row["inputs"]["legal_actions"]) > 1)
        if row["has_choice"]:
            assert 0 <= row["targets"]["weighted_score"] <= 1
            assert row["targets"]["weighted_score"] == pytest.approx(
                0.5 * row["heuristic"]["heuristic_score"] + 0.5 * row["targets"]["terminal_outcome"]
            )
        else:
            assert not row["train_eligible"]
            assert row["targets"]["weighted_score"] is None


def test_collector_round_trip_truncation_and_overwrite(tmp_path):
    output = tmp_path / "pilot"
    assert main(["--games", "2", "--max-actions", "3", "--output", str(output)]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["games"] == summary["truncated_games"] == 2
    assert summary["eligible_rows"] == 0
    with gzip.open(output / "games/game_0000.moves.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 3
    assert all(row["targets"]["weighted_score"] is None for row in rows)
    assert all(row["targets"]["terminal_outcome"] is None for row in rows)
    with pytest.raises(SystemExit):
        main(["--output", str(output)])


def test_scarce_bank_mixed_game_regression():
    _, rows, result = collect_game(20053, 53, ["R", "W", "VP", "F"], 1000, 100_000, ScoreConfig())
    assert result["termination"] == "victory"
    assert any(row["inputs"]["action"][1] == "MARITIME_TRADE" for row in rows)


def test_parallel_collection_matches_serial_completed_games(tmp_path):
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    shared = ["--games", "2", "--seed", "20053", "--save-games"]
    for output, workers in ((serial, "1"), (parallel, "2")):
        assert main([*shared, "--workers", workers, "--output", str(output)]) == 0
    for index in range(2):
        name = f"games/game_{index:04d}.moves.jsonl.gz"
        with gzip.open(serial / name, "rt", encoding="utf-8") as stream:
            serial_rows = stream.read()
        with gzip.open(parallel / name, "rt", encoding="utf-8") as stream:
            assert stream.read() == serial_rows
    for name in ("maps.jsonl", "audit_sample.jsonl"):
        assert (serial / name).read_text(encoding="utf-8") == (parallel / name).read_text(encoding="utf-8")
    for name in ("results.jsonl",):
        documents = []
        for output in (serial, parallel):
            rows = [json.loads(line) for line in (output / name).read_text(encoding="utf-8").splitlines()]
            for row in rows:
                row.pop("elapsed_seconds")
                row.pop("label_seconds")
                assert row["termination"] == "victory"
                assert (output / "games" / f"game_{row['game_index']:04d}.json").is_file()
            documents.append(rows)
        assert documents[0] == documents[1]
    summaries = [json.loads((output / "summary.json").read_text(encoding="utf-8")) for output in (serial, parallel)]
    for summary in summaries:
        for key in ("elapsed_seconds", "label_seconds", "workers"):
            summary.pop(key)
    assert summaries[0] == summaries[1]


def test_workers_reject_zero_before_creating_output(tmp_path):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit):
        main(["--workers", "0", "--output", str(output)])
    assert not output.exists()
