"""Independent tournament counts, paired starts, seat balance, and replay QA."""
import argparse
from collections import Counter
import json
from pathlib import Path

from catanzero.viewer import replay_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    root = args.dataset
    assert not (root / "FAILED.json").exists()
    config = json.loads((root / "config.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    results = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    assert len(results) == len(config["bots"]) * config["games_per_bot"]
    assert len({(r["bot"], r["game_index"]) for r in results}) == len(results)
    checked_replays = 0
    for index in range(config["games_per_bot"]):
        paired = [r for r in results if r["game_index"] == index]
        assert {r["bot"] for r in paired} == set(config["bots"])
        assert all(r["seed"] == config["base_seed"] + index for r in paired)
        assert all(r["candidate_seat"] == index % 4 for r in paired)
        for field in ("initial_map_hash", "initial_stream_hash", "candidate_color", "seating_order"):
            assert all(r[field] == paired[0][field] for r in paired), field
    for bot in config["bots"]:
        matches = [r for r in results if r["bot"] == bot]
        group = summary["groups"][bot]
        completed = [r for r in matches if r["termination"] == "victory"]
        wins = sum(r["candidate_won"] is True for r in matches)
        assert group["games"] == len(matches)
        assert group["completed_games"] == len(completed)
        assert group["wins"] == wins
        assert group["losses"] == len(completed) - wins
        assert group["unfinished_games"] == len(matches) - len(completed)
        assert Counter(r["candidate_seat"] for r in matches) == {s: len(matches) // 4 for s in range(4)}
        assert group["win_rate_completed"] == (wins / len(completed) if completed else None)
        for result in matches:
            complete = result["termination"] == "victory"
            assert (result["winner"] is not None) == complete
            assert result["candidate_won"] == (result["winner"] == result["candidate_color"] if complete else None)
            for group_times in result["decision_times"].values():
                assert all(t["legal_actions"] > 1 and t["wall_seconds"] >= 0 and t["cpu_seconds"] >= 0 for t in group_times)
            game_path = Path(result["match_directory"]) / "game.json"
            if game_path.exists():
                game = json.loads(game_path.read_text(encoding="utf-8"))
                replay = replay_data(game)
                assert len(replay["frames"]) - 1 == result["actions"]
                assert replay["frames"][-1]["winner"] == result["winner"]
                checked_replays += 1
                print(f"Audited {bot} game={result['game_index']} {result['termination']}", flush=True)
    report = {"passed": True, "games": len(results), "paired_initial_conditions": True,
              "balanced_seats": True, "counts_and_censoring": True,
              "authoritative_replays_verified": checked_replays,
              "note": "Integrity verification; not proof of statistical superiority or expert-level play."}
    (root / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
