"""Check counterfactual-label arithmetic and exact linkage to original inputs."""
import argparse
from collections import Counter
from itertools import combinations
from math import comb
import gzip
import json
from pathlib import Path


def key(action):
    return json.dumps(action, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    assert not (args.dataset / "FAILED.json").exists(), "partial run is not a completed dataset"
    config = json.loads((args.dataset / "config.json").read_text(encoding="utf-8"))
    source = Path(config["source_dataset"])
    rows = [json.loads(line) for line in (args.dataset / "search.moves.jsonl").read_text(encoding="utf-8").splitlines()]
    originals = {}
    wanted = {(r["game_index"], r["action_index"]) for r in rows}
    assert len(wanted) == len(rows), "duplicate decisions"
    source_results = {r["game_index"]: r for r in (json.loads(line) for line in (source / "results.jsonl").read_text().splitlines())}
    for game_index in {r["game_index"] for r in rows}:
        with gzip.open(source / source_results[game_index]["moves_file"], "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                identity = (row["game_index"], row["action_index"])
                if identity in wanted:
                    originals[identity] = row
    assert (args.dataset / "maps.jsonl").read_bytes() == (source / "maps.jsonl").read_bytes()
    totals = Counter()
    for row in rows:
        original = originals[row["game_index"], row["action_index"]]
        assert row["inputs"] == original["inputs"]
        assert row["source_split"] == source_results[row["game_index"]]["split"]
        for field, value in original["targets"].items():
            assert row["targets"][field] == value
        search = row["search"]
        legal = {key(a) for a in row["inputs"]["legal_actions"]}
        compared = {key(c["action"]) for c in search["candidates"]}
        assert len(compared) == len(search["candidates"])
        assert compared <= legal
        assert key(row["inputs"]["action"]) in compared
        assert search["all_legal_actions_evaluated"] == (compared == legal)
        assert search["legal_action_count"] == len(legal)
        for candidate in search["candidates"]:
            trials = candidate["rollouts"]
            assert trials == config["search"]["rollouts"]
            assert candidate["wins"] + candidate["losses"] + candidate["truncated"] == trials
            assert all(candidate[f] >= 0 for f in ("wins", "losses", "truncated"))
            attempted = candidate.get("rollouts_attempted", trials)
            skipped = candidate.get("not_attempted", 0)
            assert attempted + skipped == trials
            assert 0 <= skipped <= candidate["truncated"]
            if "trial_outcomes" in candidate:
                outcomes = candidate["trial_outcomes"]
                assert len(outcomes) == trials
                assert outcomes.count(1) == candidate["wins"]
                assert outcomes.count(0) == candidate["losses"]
                assert outcomes.count(None) == candidate["truncated"]
            expected = candidate["wins"] / trials if candidate["truncated"] == 0 else None
            assert candidate["win_rate"] == expected
            assert candidate["win_rate_bounds"] == [candidate["wins"] / trials, (candidate["wins"] + candidate["truncated"]) / trials]
            low, high = candidate["interval_approx_95_familywise"]
            assert 0 <= low <= candidate["win_rate_bounds"][0] <= candidate["win_rate_bounds"][1] <= high <= 1
            totals["rollouts"] += trials
            totals["truncated_rollouts"] += candidate["truncated"]
        best = next(c for c in search["candidates"] if c["action"] == search["best_estimated_action"])
        selected = next(c for c in search["candidates"] if c["action"] == row["inputs"]["action"])
        complete = all(c["win_rate"] is not None for c in search["candidates"])
        assert search["search_estimate_available"] == complete
        resolved = all(best["interval_approx_95_familywise"][0] > c["interval_approx_95_familywise"][1] for c in search["candidates"] if c is not best)
        assert search["ranking_resolved"] == resolved
        if "pairwise_comparisons" in search:
            expected_pairs = []
            candidates = search["candidates"]
            threshold = 0.05 / (len(candidates) * (len(candidates) - 1))
            for first, second in combinations(candidates, 2):
                x, y = first["trial_outcomes"], second["trial_outcomes"]
                assert len(x) == len(y) == config["search"]["rollouts"]
                assert all(v in (0, 1, None) for v in x + y)
                if None in x + y:
                    continue
                plus, minus = sum(a > b for a, b in zip(x, y)), sum(a < b for a, b in zip(x, y))
                favorite, rejected = (first, second) if plus >= minus else (second, first)
                n = plus + minus
                p = sum(comb(n, k) for k in range(min(plus, minus) + 1)) / 2 ** n if n else 1.0
                expected_pairs.append((favorite["action"], rejected["action"], p, p < threshold and plus != minus))
            actual = search["pairwise_comparisons"]
            assert len(actual) == len(expected_pairs)
            for pair, (favorite, rejected, p, separated) in zip(actual, expected_pairs):
                assert pair["preferred_action"] == favorite and pair["rejected_action"] == rejected
                assert pair["paired_one_sided_p_value"] == p
                assert pair["corrected_threshold"] == threshold
                assert pair["resolved"] == separated
            count = sum(p["resolved"] for p in actual)
            assert count == search["resolved_pairwise_count"]
            assert row["search_ranking_eligible"] == (count > 0)
            totals["resolved_pairs"] += count
            totals["positions_with_resolved_pairs"] += int(count > 0)
        else:
            assert row["search_ranking_eligible"] == resolved
        assert search["selected_win_rate"] == row["targets"]["search_win_rate"] == selected["win_rate"]
        regret = max(c["win_rate"] for c in search["candidates"]) - selected["win_rate"] if complete else None
        quality = 1 - regret if regret is not None else None
        assert search["regret_among_candidates"] == regret
        assert search["quality_score"] == row["targets"]["search_quality_score"] == quality
        expected = 0.5 * quality + 0.5 * row["targets"]["terminal_outcome"] if quality is not None else None
        assert row["targets"]["search_weighted_score"] == expected
        totals["positions"] += 1
        totals["ranking_resolved"] += int(resolved)
        totals["fully_evaluated_positions"] += int(search["all_legal_actions_evaluated"])
        totals["simulated_actions"] += search["simulated_actions"]
    summary = json.loads((args.dataset / "summary.json").read_text(encoding="utf-8"))
    for field, value in totals.items():
        assert summary[field] == value
    report = {"passed": True, **totals, "scope": "structural, arithmetic, source-input and original-split checks; not strategic validation"}
    (args.dataset / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
