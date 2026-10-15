import gzip
import json

from catanatron.serialization import map_to_json, state_to_json

from catanzero.dataset import ordered_actions
from catanzero.features import feature_schema
from catanzero.learned_match import LearnedPlayer, make_learned_match, run
from catanzero.viewer import replay_data


def manifest_fixture(tmp_path):
    schema = feature_schema()
    scaler = {"feature_names": schema["feature_names"], "fit_split": "train",
              "offset": [0.0] * schema["input_columns"], "scale": [1.0] * schema["input_columns"]}
    normalizer = tmp_path / "normalizer.json"
    normalizer.write_text(json.dumps(scaler))
    result = {"normalizer_path": str(normalizer), "source_board_hashes": []}
    for policy in ("A_mixed", "B_terminal"):
        path = tmp_path / f"{policy}.json"
        path.write_text(json.dumps({"feature_names": schema["feature_names"],
                                    "input_columns": schema["input_columns"], "baseline": 0.5, "trees": []}))
        result[policy] = {"portable_path": str(path)}
    return result


def test_swapped_pair_preserves_initial_board_future_stream_and_balance(tmp_path):
    manifest = manifest_fixture(tmp_path)
    counts = {p: [0] * 4 for p in ("A_mixed", "B_terminal")}
    for board in range(4):
        first, _, seats0 = make_learned_match(manifest, 120000 + board, board, 0)
        other, _, seats1 = make_learned_match(manifest, 120000 + board, board, 1)
        assert map_to_json(first.state.board.map) == map_to_json(other.state.board.map)
        assert first.state.random.getstate() == other.state.random.getstate()
        assert seats0["A_mixed"] == seats1["B_terminal"]
        assert seats0["B_terminal"] == seats1["A_mixed"]
        for seats in (seats0, seats1):
            for policy, seat in seats.items():
                counts[policy][seat] += 1
    assert counts == {"A_mixed": [2, 2, 2, 2], "B_terminal": [2, 2, 2, 2]}


def test_policy_choice_does_not_mutate_real_state_or_random_stream(tmp_path):
    manifest = manifest_fixture(tmp_path)
    game, _, _ = make_learned_match(manifest, 120000, 0, 0)
    player = game.state.current_player()
    assert isinstance(player, LearnedPlayer)
    state_before = state_to_json(game)
    rng_before = game.state.random.getstate()
    actions = ordered_actions(game)
    choice = player.decide(game, actions)
    assert choice == actions[0]
    assert len(player.last_scores) == len(actions)
    assert state_to_json(game) == state_before
    assert game.state.random.getstate() == rng_before


def test_capped_match_keeps_unknown_outcome_and_predecision_trace(tmp_path):
    manifest = manifest_fixture(tmp_path)
    output = tmp_path / "match"
    output.mkdir()
    result = run(manifest, 120000, 0, 0, output, max_actions=4)
    assert result["termination"] == "action_limit"
    assert result["winner"] is result["winner_policy"] is None
    document = json.loads((output / "game.json").read_text())
    frames = replay_data(document)["frames"]
    assert len(frames) == 5
    with gzip.open(output / "decisions.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 4
    assert rows[0]["inputs"]["observation"]["buildings"] == []
    assert all(r["inputs"]["action"] in r["inputs"]["legal_actions"] for r in rows)
    assert rows[0]["raw_scores"] is not None


def test_direct_lineup_keeps_four_players_and_equal_policy_representation(tmp_path):
    manifest = manifest_fixture(tmp_path)
    manifest["lineup"] = "AABB"
    game, policies, seats = make_learned_match(manifest, 130000, 0, 0)
    swapped, other, other_seats = make_learned_match(manifest, 130000, 0, 1)
    assert list(policies.values()).count("A_mixed") == list(policies.values()).count("B_terminal") == 2
    assert all(isinstance(p, LearnedPlayer) for p in game.state.players)
    assert set(seats["A_mixed"]) == set(other_seats["B_terminal"])
    assert game.state.random.getstate() == swapped.state.random.getstate()
