"""Replay every learned-policy match and verify public inputs, scores and seats."""

import argparse
from collections import Counter
import gzip
import json
from math import isclose, isfinite
from pathlib import Path

from catanatron.models.enums import ActionType
from catanatron.serialization import action_record_from_json, action_to_json, state_to_json

from catanzero.dataset import observation, ordered_actions
from catanzero.features import FeatureEncoder
from catanzero.portable_trees import PortableTrees
from catanzero.regression_data import privacy_invariance_check
from catanzero.simulation import SimulationConfig, create_game
from catanzero.tournament import canonical_hash


def audit_phase(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    results = [json.loads(line) for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    scorers = {p: PortableTrees.load(manifest[p]["portable_path"], manifest["normalizer_path"])
               for p in ("A_mixed", "B_terminal")}
    assert manifest["models_saved_before_gameplay"]
    wins, seats, checked, score_checks, all_candidate_checks = Counter(), {p: Counter() for p in scorers}, 0, 0, 0
    for board in sorted({r["board_index"] for r in results}):
        pair = [r for r in results if r["board_index"] == board]
        assert len(pair) == 2 and {r["leg"] for r in pair} == {0, 1}
        assert pair[0]["seed"] == pair[1]["seed"]
        for field in ("initial_map_hash", "initial_stream_hash", "seating_order"):
            assert pair[0][field] == pair[1][field]
        def seat_set(record, policy):
            value = record["policy_seats"][policy]
            return set(value if isinstance(value, list) else [value])
        assert seat_set(pair[0], "A_mixed") == seat_set(pair[1], "B_terminal")
        assert seat_set(pair[0], "B_terminal") == seat_set(pair[1], "A_mixed")
    for result in results:
        directory = Path(result["match_directory"])
        document = json.loads((directory / "game.json").read_text(encoding="utf-8"))
        game = create_game(SimulationConfig(players=4), result["seed"])
        assert canonical_hash(document["map"]) == result["initial_map_hash"]
        assert result["initial_map_hash"] not in manifest["source_board_hashes"]
        assert canonical_hash(game.state.random.getstate()) == result["initial_stream_hash"]
        encoder = FeatureEncoder(document["map"])
        with gzip.open(directory / "decisions.jsonl.gz", "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == len(document["action_records"]) == result["actions"]
        type_counts = {policy: Counter() for policy in ("A_mixed", "B_terminal", "F")}
        checked_types = set()
        for index, (row, payload) in enumerate(zip(rows, document["action_records"])):
            record = action_record_from_json(payload)
            selected = record.action
            if selected.action_type in (ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD):
                selected = selected._replace(value=None)
            inputs = row["inputs"]
            assert row["action_index"] == index
            assert inputs["observation"] == observation(game, game.state.current_color())
            assert inputs["legal_actions"] == [action_to_json(a) for a in ordered_actions(game)]
            assert inputs["action"] == action_to_json(selected)
            assert row["policy"] == result["policies_by_color"][selected.color.value]
            assert row["has_choice"] == (len(inputs["legal_actions"]) > 1)
            if row["has_choice"]:
                type_counts[row["policy"]][selected.action_type.value] += 1
            if row["has_choice"] and row["policy"] in scorers:
                scores = row["raw_scores"]
                assert len(scores) == len(inputs["legal_actions"]) and all(isfinite(s) for s in scores)
                choice_index = max(range(len(scores)), key=scores.__getitem__)
                assert inputs["legal_actions"][choice_index] == inputs["action"]
                scorer = scorers[row["policy"]]
                actual = scorer.predict_raw_features(encoder.encode(inputs))
                assert isclose(actual, scores[choice_index], rel_tol=1e-12, abs_tol=1e-12)
                score_checks += 1
                stratum = (row["policy"], selected.action_type.value)
                if stratum not in checked_types:
                    for action, expected in zip(inputs["legal_actions"], scores):
                        actual = scorer.predict_raw_features(encoder.encode({**inputs, "action": action}))
                        assert isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
                        all_candidate_checks += 1
                    assert privacy_invariance_check(encoder, {"inputs": inputs, "targets": {}, "heuristic": {}})
                    checked_types.add(stratum)
            game.execute(selected, action_record=record)
            checked += 1
        final = state_to_json(game)
        for field in ("board", "player_state", "num_turns", "action_records", "resource_freqdeck", "development_listdeck"):
            assert final[field] == document[field]
        winner = game.winning_color()
        assert result["winner"] == (winner.value if winner else None)
        if result["termination"] == "victory":
            assert winner is not None
            assert result["winner_policy"] == result["policies_by_color"][winner.value]
            wins[result["winner_policy"]] += 1
        else:
            assert result["winner"] is result["winner_policy"] is None
        for policy, values in type_counts.items():
            assert dict(values) == result["decision_action_counts"][policy]
        for policy in scorers:
            value = result["policy_seats"][policy]
            seats[policy].update(value if isinstance(value, list) else [value])
    assert summary["games"] == len(results)
    assert summary["completed_games"] == sum(r["termination"] == "victory" for r in results)
    assert summary["unfinished_games"] == len(results) - summary["completed_games"]
    for policy, values in summary["groups"].items():
        assert values["wins"] == wins[policy]
    assert all(len(counts) == 4 and len(set(counts.values())) == 1 for counts in seats.values())
    return {"passed": True, "games": len(results), "actions_verified": checked,
            "selected_predictions_verified": score_checks, "candidate_predictions_verified": all_candidate_checks,
            "swapped_initial_conditions_verified": True, "fresh_boards_verified": True,
            "all_predecision_inputs_verified": True, "censoring_verified": True,
            "seat_counts": {p: dict(c) for p, c in seats.items()}, "wins": dict(wins)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    study = args.study
    report = {"anchor": audit_phase(study), "direct": audit_phase(study / "direct")}
    with (study / "game_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
