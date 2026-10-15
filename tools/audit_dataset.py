"""Check every pilot row without fitting or selecting a model."""

import argparse
from collections import Counter
import gzip
import json
from math import exp, isclose
from pathlib import Path


def audit(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    results = [json.loads(line) for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    maps = [json.loads(line) for line in (directory / "maps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(results) == config["games"] == len(maps)
    map_ids = {m["map_ref"] for m in maps}
    seen_ids = set()
    counts, action_counts, wins = Counter(), Counter(), Counter()
    for result in results:
        if result["winner"] is not None:
            wins[result["bot_seats"][result["winner"]]] += 1
        with gzip.open(directory / result["moves_file"], "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == result["actions"]
        assert sum(row["has_choice"] for row in rows) == result["decisions"]
        assert sum(row["train_eligible"] for row in rows) == result["eligible_rows"]
        for index, row in enumerate(rows):
            row_id = (row["game_index"], row["action_index"])
            assert row_id not in seen_ids
            seen_ids.add(row_id)
            assert row_id == (result["game_index"], index)
            inputs = row["inputs"]
            assert set(inputs) == {"observation", "action", "legal_actions", "map_ref"}
            assert inputs["map_ref"] in map_ids
            assert inputs["action"] in inputs["legal_actions"]
            obs = inputs["observation"]
            actor = obs["actor"]
            assert inputs["action"][0] == actor
            assert row["actor_bot"] == result["bot_seats"][actor]
            assert not {"seed", "winner", "random_state", "development_listdeck", "targets"}.intersection(obs)
            for player in obs["players"]:
                if player["color"] != actor:
                    assert not {"actual_vp", "resources", "development_cards", "playable_at_turn_start"}.intersection(player)
                else:
                    assert all(count >= 0 for count in player["resources"].values())
            assert row["has_choice"] == (len(inputs["legal_actions"]) > 1)
            assert row["train_eligible"] == (row["has_choice"] and result["winner"] is not None)
            targets = row["targets"]
            outcome = float(actor == result["winner"]) if result["winner"] is not None else None
            assert targets["terminal_outcome"] == outcome
            counts["actions"] += 1
            if row["has_choice"]:
                counts["decisions"] += 1
                action_counts[inputs["action"][1]] += 1
                heuristic = row["heuristic"]
                assert heuristic["version"] == config["heuristic_version"]
                assert heuristic["alternatives_evaluated"] == len(inputs["legal_actions"])
                assert 0 <= heuristic["heuristic_score"] <= 1
                assert isclose(heuristic["raw_regret"], max(heuristic["best_value"] - heuristic["selected_value"], 0))
                assert isclose(heuristic["heuristic_score"], exp(-heuristic["raw_regret"] / config["regret_scale"]))
                if outcome is not None:
                    expected = config["weights"]["heuristic"] * heuristic["heuristic_score"] + config["weights"]["outcome"] * outcome
                    assert isclose(targets["weighted_score"], expected)
                    counts["eligible_rows"] += 1
                    counts["winning_actor_rows" if outcome else "losing_actor_rows"] += 1
                    counts["heuristic_best_rows"] += heuristic["raw_regret"] < 1e-9
                else:
                    if config["weights"]["outcome"] == 0:
                        assert isclose(targets["weighted_score"], heuristic["heuristic_score"])
                    else:
                        assert targets["weighted_score"] is None
            else:
                assert row["heuristic"] is None and targets["weighted_score"] is None
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    for key in ("actions", "decisions", "eligible_rows"):
        assert counts[key] == summary[key]
    assert dict(action_counts) == summary["decision_action_counts"]
    return {"all_rows_validated": True, "games": len(results), **counts,
            "wins_by_bot": dict(wins), "decision_action_counts": dict(action_counts),
            "note": "Structural and numerical checks only; not an expert audit of move quality."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = audit(args.directory)
    with (args.directory / "audit_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
