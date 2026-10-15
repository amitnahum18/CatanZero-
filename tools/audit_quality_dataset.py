"""Audit replay-derived multi-belief rows, candidate export, and grouped splits."""

import argparse
from collections import Counter
import gzip
import json
from math import isclose, isfinite
from pathlib import Path

from catanatron.models.enums import ActionType
from catanatron.serialization import action_record_from_json, action_to_json, state_to_json

from audit_dataset import audit
from catanzero.dataset import observation, ordered_actions
from catanzero.tournament import make_match
from catanzero.tournament_dataset import VERIFY_FIELDS, seed_splits


def audit_quality(root):
    assert not (root / "FAILED.json").exists()
    report = audit(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    results = [json.loads(line) for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    source = Path(config["source_tournament"])
    original = [json.loads(line) for line in (source / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = seed_splits(r["seed"] for r in original)
    expected_candidates = []
    for result in results:
        source_result = next(r for r in original if r["bot"] == config["candidate_bot"]
                             and r["game_index"] == result["source_game_index"])
        assert result["split"] == splits[result["seed"]]
        assert source_result["seed"] == result["seed"]
        assert source_result["winner"] == result["winner"]
        assert source_result["candidate_color"] == result["candidate_color"]
        game, candidate = make_match(source_result["bot"], result["seed"], source_result["candidate_seat"])
        document = json.loads((root / "games" / f"game_{result['game_index']:04d}.json").read_text(encoding="utf-8"))
        with gzip.open(root / result["moves_file"], "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == len(document["action_records"])
        for row, payload in zip(rows, document["action_records"]):
            record = action_record_from_json(payload)
            selected = record.action
            if selected.action_type in (ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD):
                selected = selected._replace(value=None)
            assert row["inputs"]["observation"] == observation(game, game.state.current_color())
            assert row["inputs"]["legal_actions"] == [action_to_json(a) for a in ordered_actions(game)]
            assert row["inputs"]["action"] == action_to_json(selected)
            assert row["split"] == result["split"]
            assert row["candidate_action"] == (selected.color == candidate)
            assert row["imitation_eligible"] == (row["train_eligible"] and row["candidate_action"])
            if row["has_choice"]:
                label = row["heuristic"]
                assert label["belief_samples"] == config["belief_samples_per_decision"]
                candidates = label["candidate_values"]
                assert [c["action"] for c in candidates] == row["inputs"]["legal_actions"]
                assert all(isfinite(c["mean"]) and isfinite(c["std"]) and c["std"] >= 0 for c in candidates)
                assert isclose(label["best_value"], max(c["mean"] for c in candidates))
                selected_value = next(c["mean"] for c in candidates if c["action"] == row["inputs"]["action"])
                assert isclose(label["selected_value"], selected_value)
            if row["imitation_eligible"]:
                expected_candidates.append(row)
            game.execute(selected, action_record=record)
        final = state_to_json(game)
        for field in VERIFY_FIELDS:
            assert final[field] == document[field]
        assert sum(r["imitation_eligible"] for r in rows) == len(source_result["decision_times"]["candidate"])
    with gzip.open(root / "candidate.moves.jsonl.gz", "rt", encoding="utf-8") as stream:
        actual_candidates = [json.loads(line) for line in stream]
    assert actual_candidates == expected_candidates
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["candidate_rows"] == len(actual_candidates)
    assert summary["candidate_split_rows"] == dict(Counter(r["split"] for r in actual_candidates))
    report.update(authoritative_replays_verified=len(results),
                  all_preaction_observations_verified=True,
                  candidate_rows=len(actual_candidates), candidate_export_verified=True,
                  initial_seed_split_groups_verified=True, candidate_value_statistics_verified=True,
                  note="Integrity and label arithmetic verified; not proof of expert play or calibrated confidence.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = audit_quality(args.directory)
    with (args.directory / "audit_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
