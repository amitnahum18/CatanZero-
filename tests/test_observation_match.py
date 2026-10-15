from copy import deepcopy
import gzip
import json

import pytest
from catanatron import Game
from catanatron.players.value import ValueFunctionPlayer
from catanatron.serialization import action_to_json, state_to_json

from catanzero.dataset import observation, ordered_actions
from catanzero.observation_match import make_match, run
from catanzero.static_policies import forbid_forward_simulation
from catanzero.viewer import replay_data
from test_learned_match import manifest_fixture


def test_actual_search_baseline_is_rejected_and_caught_calls_also_fail(tmp_path):
    game, _, _, _ = make_match(manifest_fixture(tmp_path), 140000, 0, 0)
    player = ValueFunctionPlayer(game.state.current_color())
    with pytest.raises(RuntimeError, match="Game.copy"):
        with forbid_forward_simulation():
            player.decide(game, ordered_actions(game))
    with pytest.raises(RuntimeError, match="attempted"):
        with forbid_forward_simulation():
            try:
                game.copy()
            except RuntimeError:
                pass


@pytest.mark.parametrize("method", ["execute", "__init__"])
def test_transition_and_new_game_are_forbidden(tmp_path, method):
    game, _, _, _ = make_match(manifest_fixture(tmp_path), 140000, 0, 0)
    with pytest.raises(RuntimeError, match="forbidden"):
        with forbid_forward_simulation():
            game.execute(ordered_actions(game)[0]) if method == "execute" else Game(game.state.players, seed=1)


def test_all_policies_choose_from_observation_and_ignore_unknown_private_fields(tmp_path):
    game, _, policies, _ = make_match(manifest_fixture(tmp_path), 140000, 0, 0)
    inputs = {"observation": observation(game, game.state.current_color()),
              "legal_actions": [action_to_json(a) for a in ordered_actions(game)]}
    changed = deepcopy(inputs)
    changed.update(seed=99, winner="RED", future_roll=[6, 6])
    changed["observation"].update(random_state="future", development_listdeck=["VICTORY_POINT"] * 99)
    for p in changed["observation"]["players"]:
        if p["color"] != changed["observation"]["actor"]:
            p.update(resources={"ORE": 999}, actual_vp=999, development_cards={"KNIGHT": 999})
    before, rng = state_to_json(game), game.state.random.getstate()
    for policy in policies.values():
        with forbid_forward_simulation() as attempts:
            index, scores = policy.choose(inputs)
            assert policy.choose(changed) == (index, scores)
        assert attempts == []
        assert index == max(range(len(scores)), key=scores.__getitem__)
    assert state_to_json(game) == before and game.state.random.getstate() == rng
    # Patches are restored: the real selected move can now execute.
    game.execute(ordered_actions(game)[index])


def test_balanced_seats_and_equal_initial_conditions(tmp_path):
    manifest = manifest_fixture(tmp_path)
    counts = {p: [0] * 4 for p in ("A_mixed", "B_terminal")}
    for board in range(4):
        g0, c0, _, s0 = make_match(manifest, 140000 + board, board, 0)
        g1, c1, _, s1 = make_match(manifest, 140000 + board, board, 1)
        doc0, doc1 = state_to_json(g0), state_to_json(g1)
        # Engine assigns a fresh UUID; it has no effect on board or randomness.
        doc0["game"].pop("id")
        doc1["game"].pop("id")
        assert doc0 == doc1
        assert g0.state.random.getstate() == g1.state.random.getstate()
        assert s0["A_mixed"] == s1["B_terminal"]
        assert list(c0.values()).count("F_static") == list(c1.values()).count("F_static") == 2
        for seats in (s0, s1):
            for p, seat in seats.items():
                counts[p][seat] += 1
    assert counts == {"A_mixed": [2]*4, "B_terminal": [2]*4}


def test_guarded_capped_match_and_authoritative_replay(tmp_path):
    output = tmp_path / "match"
    result = run(manifest_fixture(tmp_path), 140000, 0, 0, output, max_actions=12)
    assert result["termination"] == "action_limit", result
    assert result["winner"] is result["winner_policy"] is None
    assert sum(result["guarded_decisions"].values()) == result["actions"] == 12
    assert result["forbidden_engine_calls"] == []
    document = json.loads((output / "game.json").read_text())
    assert len(replay_data(document)["frames"]) == 13
    with gzip.open(output / "decisions.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert all(r["forward_simulation_guard"] and not r["forbidden_engine_calls"] for r in rows)
    assert {r["policy"] for r in rows} == {"A_mixed", "B_terminal", "F_static"}
