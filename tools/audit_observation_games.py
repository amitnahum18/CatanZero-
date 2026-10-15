"""Replay observation-only games; check decisions, guard coverage and pairing."""

import argparse
from collections import Counter
import gzip
import json
from math import isclose
from pathlib import Path

from catanatron.models.enums import ActionType
from catanatron.serialization import action_record_from_json, action_to_json, state_to_json

from catanzero.dataset import observation, ordered_actions
from catanzero.observation_match import make_match
from catanzero.regression_data import privacy_invariance_check
from catanzero.static_policies import forbid_forward_simulation
from catanzero.tournament import atomic_json


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    results = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    summary = json.loads((root / "summary.json").read_text())
    counts, candidates, choices, private_checks = Counter(), 0, 0, 0
    seat_counts = {p: Counter() for p in ("A_mixed", "B_terminal")}
    for board in sorted({r["board_index"] for r in results}):
        pair = [r for r in results if r["board_index"] == board]
        assert len(pair) == 2 and {r["leg"] for r in pair} == {0, 1}
        for field in ("seed", "initial_map_hash", "initial_stream_hash", "seating_order"):
            assert pair[0][field] == pair[1][field]
        assert pair[0]["policy_seats"]["A_mixed"] == pair[1]["policy_seats"]["B_terminal"]
    for r in results:
        assert r["error"] is None and r["forbidden_engine_calls"] == []
        directory = Path(r["match_directory"])
        document = json.loads((directory / "game.json").read_text())
        game, by_color, policies, seats = make_match(manifest, r["seed"], r["board_index"], r["leg"])
        assert by_color == r["policies_by_color"] and seats == r["policy_seats"]
        assert r["initial_map_hash"] not in manifest["source_board_hashes"]
        with gzip.open(directory / "decisions.jsonl.gz", "rt") as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == len(document["action_records"]) == r["actions"]
        guards, action_counts, strata = Counter(), {p: Counter() for p in policies}, set()
        for i, (row, payload) in enumerate(zip(rows, document["action_records"])):
            record = action_record_from_json(payload)
            selected = record.action
            if selected.action_type in (ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD):
                selected = selected._replace(value=None)
            inputs, policy = row["inputs"], policies[row["policy"]]
            assert inputs["observation"] == observation(game, game.state.current_color())
            assert inputs["legal_actions"] == [action_to_json(a) for a in ordered_actions(game)]
            assert inputs["action"] == action_to_json(selected)
            assert row["action_index"] == i and row["policy"] == by_color[selected.color.value]
            assert row["forward_simulation_guard"] and not row["forbidden_engine_calls"]
            assert row["has_choice"] == (len(inputs["legal_actions"]) > 1)
            guards[row["policy"]] += 1
            if row["has_choice"]:
                scores = row["raw_scores"]
                index = max(range(len(scores)), key=scores.__getitem__)
                assert inputs["legal_actions"][index] == inputs["action"]
                vector = policy.encoder.encode(inputs)
                actual = policy.heuristic(inputs, vector) if policy.scorer is None else policy.scorer.predict_raw_features(vector)
                assert isclose(actual, scores[index], rel_tol=1e-12, abs_tol=1e-12)
                choices += 1
                action_counts[row["policy"]][selected.action_type.value] += 1
                stratum = (row["policy"], selected.action_type.value)
                if stratum not in strata:
                    with forbid_forward_simulation():
                        got_index, got_scores = policy.choose(inputs)
                    assert got_index == index
                    assert len(got_scores) == len(scores)
                    assert all(isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(got_scores, scores))
                    candidates += len(scores)
                    assert privacy_invariance_check(policy.encoder, {"inputs": inputs, "targets": {}, "heuristic": {}})
                    private_checks += 1
                    strata.add(stratum)
            else:
                assert row["raw_scores"] is None
            game.execute(selected, action_record=record)
        final = state_to_json(game)
        for field in ("board", "player_state", "num_turns", "action_records", "resource_freqdeck", "development_listdeck"):
            assert final[field] == document[field]
        assert dict(guards) == {p: n for p, n in r["guarded_decisions"].items() if n}
        assert {p: dict(c) for p, c in action_counts.items()} == r["decision_action_counts"]
        winner = game.winning_color()
        assert r["winner"] == (winner.value if winner else None)
        if winner:
            assert r["termination"] == "victory" and r["winner_policy"] == by_color[winner.value]
            counts[r["winner_policy"]] += 1
        else:
            assert r["winner_policy"] is None and r["termination"] in ("action_limit", "time_limit", "turn_limit")
        for p, seat in seats.items():
            seat_counts[p][seat] += 1
    assert summary["games"] == len(results)
    for p, g in summary["groups"].items():
        assert g["wins"] == counts[p]
    assert summary["completed_games"] == sum(counts.values())
    assert all(len(c) == 4 and len(set(c.values())) == 1 for c in seat_counts.values())
    return {"passed": True, "games": len(results), "actions_verified": sum(r["actions"] for r in results),
            "guarded_decisions_verified": sum(sum(r["guarded_decisions"].values()) for r in results),
            "selected_scores_verified": choices, "candidate_scores_verified": candidates,
            "private_information_invariance_checks": private_checks,
            "forbidden_engine_calls": 0, "seat_counts": {p: dict(c) for p, c in seat_counts.items()},
            "paired_boards_and_streams_verified": True, "wins": dict(counts)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    report = audit(args.study)
    atomic_json(args.study / "game_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)
