"""Replay every duel; independently re-score first/last games per condition."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from catanatron.serialization import action_to_json, map_to_json, state_to_json
from catanzero.board_exclusions import board_hash_variants
from catanzero.dataset import observation, ordered_actions
from catanzero.duel_protocol import CONDITIONS
from catanzero.regression_search import RegressionSearch, public_f_choice
from catanzero.search_labels import live_belief
from catanzero.simulation import create_game, SimulationConfig
from catanzero.static_policies import forbid_forward_simulation
from catanzero.tournament import atomic_json, canonical_hash


def audit(root):
    start = perf_counter()
    config = json.loads((root / "config.json").read_text())
    rows = [json.loads(s) for s in (root / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 4 * config["games_per_condition"]
    assert len({(r["condition"], r["game_index"]) for r in rows}) == len(rows)
    assert len({r["initial_map_hash"] for r in rows}) == config["games_per_condition"]
    model = json.loads(Path(config["model"]).read_text())
    norm = json.loads(Path(config["normalizer"]).read_text())
    assert len(model["weights"]) == len(norm["feature_names"]) == 264
    for name, digest in config["source_hashes"].items():
        assert hashlib.sha256((Path(config["source"]) / name).read_bytes()).hexdigest() == digest
        if name != "model.joblib":
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    totals = {"games_replayed": 0, "actions_replayed": 0, "observations_verified": 0,
              "policy_rescore_games": 0, "policy_decisions_rescored": 0,
              "privacy_checks": 0, "max_base_score_error": 0.}
    paired = {}
    for result in sorted(rows, key=lambda r: (r["game_index"], r["condition"])):
        condition, index = result["condition"], result["game_index"]
        rescore = index in (0, 28, config["games_per_condition"]-1)
        output = root / condition / f"game_{index:04d}"
        game = create_game(SimulationConfig(players=2), result["seed"])
        document = map_to_json(game.state.board.map)
        assert len(game.state.colors) == 2 and game.vps_to_win == 10
        assert canonical_hash(document) == result["initial_map_hash"]
        assert canonical_hash(game.state.random.getstate()) == result["initial_stream_hash"]
        assert not (board_hash_variants(document) & set(config["excluded_board_hashes"]))
        pair = (result["seed"], result["candidate_seat"], result["candidate_color"],
                result["initial_map_hash"], result["initial_stream_hash"])
        if index in paired:
            assert paired[index] == pair
        paired[index] = pair
        candidate = game.state.colors[result["candidate_seat"]]
        assert candidate.value == result["candidate_color"]
        depth, opponent = CONDITIONS[condition]
        policy = RegressionSearch(model, norm, document, depth=depth, beam=config["beam"],
                                  beliefs=config["beliefs"], max_branch_actions=config["max_branch_actions"],
                                  allow_fewer_players=True, opponent_policy=opponent)
        privacy_kinds = set()
        with gzip.open(output / "decisions.jsonl.gz", "rt", encoding="utf-8") as trace:
            for line in trace:
                row = json.loads(line)
                actions = ordered_actions(game)
                actor = game.state.current_color()
                actual_inputs = {"observation": observation(game, actor), "map_ref": 0,
                                 "legal_actions": [action_to_json(a) for a in actions]}
                assert row["action_index"] == len(game.state.action_records)
                assert row["inputs"] == actual_inputs
                assert row["policy"] == ("v4" if actor == candidate else "F")
                assert row["decision_seed"] == result["seed"]*1000003 + row["action_index"]
                assert 0 <= row["selected_index"] < len(actions)
                totals["observations_verified"] += 1
                if rescore and len(actions) > 1:
                    stream, before = game.state.random.getstate(), state_to_json(game)
                    if actor == candidate:
                        if depth == 1:
                            with forbid_forward_simulation():
                                chosen, detail = policy.choose(game, actions, row["decision_seed"])
                        else:
                            chosen, detail = policy.choose(game, actions, row["decision_seed"])
                        assert detail == row["detail"]
                    elif opponent == "F":
                        chosen = actions.index(public_f_choice(game, actions, row["decision_seed"]))
                    else:
                        with forbid_forward_simulation():
                            chosen, scores = policy.teacher.choose(actual_inputs)
                        assert {"static_scores": scores} == row["detail"]
                    assert chosen == row["selected_index"]
                    assert stream == game.state.random.getstate() and before == state_to_json(game)
                    totals["policy_decisions_rescored"] += 1
                    kind = actions[chosen].action_type.value
                    key = (row["policy"], kind)
                    if index == 0 and key not in privacy_kinds:
                        changed = live_belief(game, actor, 989878 + row["action_index"])
                        assert observation(changed, actor) == actual_inputs["observation"]
                        other = ordered_actions(changed)
                        assert other == actions
                        if actor == candidate:
                            assert policy.choose(changed, other, row["decision_seed"]) == (chosen, detail)
                        elif opponent == "F":
                            assert other.index(public_f_choice(changed, other, row["decision_seed"])) == chosen
                        else:
                            with forbid_forward_simulation():
                                assert policy.teacher.choose({**actual_inputs, "observation": observation(changed, actor)}) == (chosen, scores)
                        privacy_kinds.add(key)
                        totals["privacy_checks"] += 1
                game.execute(actions[row["selected_index"]])
                totals["actions_replayed"] += 1
        assert len(game.state.action_records) == result["actions"]
        winner = game.winning_color()
        assert (winner.value if winner else None) == result["winner"]
        assert (winner == candidate if winner else None) == result["candidate_won"]
        final = state_to_json(game)
        saved = json.loads((output / "game.json").read_text())
        # A game UUID is generated independently of the seeded simulator.
        final["game"].pop("id")
        saved["game"].pop("id")
        assert final == saved
        totals["games_replayed"] += 1
        totals["policy_rescore_games"] += int(rescore)
        if totals["games_replayed"] % 50 == 0:
            print(f"AUDIT {totals['games_replayed']}/{len(rows)}, {perf_counter()-start:.1f}s", flush=True)
    report = {"status": "passed", **totals, "paired_unique_boards": len(paired),
              "source_weights_and_normalizer_unchanged": True, "input_columns": 264,
              "scores_exact_on_rescored_games": True,
              "scope": "all games full engine replay/whitelisted observations/legals/final outcomes; first, last and recovered index 28 in each condition fully re-scored",
              "elapsed_seconds": perf_counter()-start}
    atomic_json(root / "audit.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    audit(parser.parse_args().root.resolve())
