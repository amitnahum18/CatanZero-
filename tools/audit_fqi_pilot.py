"""Audit compact same-actor experience and replay a fixed sample of episodes."""

import argparse
import gzip
import json
from pathlib import Path
import pickle

from catanatron.models.enums import ActionType
from catanatron.serialization import action_record_from_json, action_to_json, map_to_json, state_to_json

from catanzero.dataset import observation, ordered_actions
from catanzero.features import FeatureEncoder
from catanzero.simulation import SimulationConfig, create_game
from catanzero.tournament import atomic_json


def audit(root, phases):
    report = {}
    for phase in phases:
        directory = root / phase
        results = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        ids, boards, states, replay_actions = set(), set(), 0, 0
        by_seat = {i: 0 for i in range(4)}
        for result in results:
            assert result["game_index"] not in ids
            ids.add(result["game_index"])
            boards.add(result["initial_map_hash"])
            by_seat[result["learner_seat"]] += 1
            assert result["forbidden_engine_calls"] == 0
            assert result["actions"] == result["guarded_decisions"]
            with gzip.open(directory / f"game_{result['game_index']:04d}/experience.pkl.gz", "rb") as stream:
                data = pickle.load(stream)
            assert data["result"] == result and len(data["trace"]) == result["actions"]
            previous = -1
            for s in data["states"]:
                assert s["actor"] == result["learner_color"] and s["action_index"] > previous
                previous = s["action_index"]
                assert data["trace"][s["action_index"]]["actor"] == s["actor"]
                assert len(s["features"]) == 164*s["candidate_count"]
                assert 0 <= s["selected_index"] < s["candidate_count"]
                states += 1
            if result["game_index"] not in {0, 1, 2, 3, 100, 101, 102, 103, 400, 401, 402, 403}:
                continue
            document = json.loads((directory / f"game_{result['game_index']:04d}/game.json").read_text())
            game = create_game(SimulationConfig(), result["seed"])
            encoder = FeatureEncoder(map_to_json(game.state.board.map))
            states_by_index = {s["action_index"]: s for s in data["states"]}
            for index, payload in enumerate(document["action_records"]):
                record = action_record_from_json(payload)
                action = record.action
                if action.action_type in (ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD):
                    action = action._replace(value=None)
                assert action_to_json(action) == data["trace"][index]["selected_intent"]
                if index in states_by_index:
                    s = states_by_index[index]
                    legal = [action_to_json(a) for a in ordered_actions(game)]
                    inputs = {"observation": observation(game, game.state.current_color()), "legal_actions": legal,
                              "action": action_to_json(action)}
                    expected = encoder.encode(inputs)
                    selected = list(s["features"])[164*s["selected_index"]:164*(s["selected_index"]+1)]
                    assert legal[s["selected_index"]] == inputs["action"]
                    assert all(abs(a-b) <= 1e-5 * max(1, abs(a)) for a, b in zip(expected, selected))
                game.execute(action, action_record=record)
                replay_actions += 1
            final = state_to_json(game)
            for field in ("board", "player_state", "num_turns", "action_records", "resource_freqdeck", "development_listdeck"):
                assert final[field] == document[field]
            winner = game.winning_color()
            assert result["winner"] == (winner.value if winner else None)
        report[phase] = {"passed": True, "games": len(results), "unique_ids": len(ids), "unique_boards": len(boards),
                         "same_actor_states_checked": states, "sampled_replay_actions_verified": replay_actions,
                         "seat_counts": by_seat, "forbidden_engine_calls": 0, "board_hashes": sorted(boards)}
    if "collection" in report and "test" in report:
        assert not set(report["collection"]["board_hashes"]).intersection(report["test"]["board_hashes"])
    atomic_json(root / "experience_audit.json", report)
    print(json.dumps({p: {k: v for k, v in r.items() if k != 'board_hashes'} for p, r in report.items()}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--phases", nargs="+", default=["collection", "test"])
    args = parser.parse_args()
    audit(args.root, args.phases)
