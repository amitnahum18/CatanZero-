"""Measure game-level processes on identical observation-only trajectories."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

REPO = Path(__file__).resolve().parents[1]


def write(path, document):
    path.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def play(manifest, directory, index, seed, seconds):
    board, leg = divmod(index, 2)
    cmd = [sys.executable, "-u", "-m", "catanzero.observation_match", str(manifest),
           "--seed", str(seed + board), "--board-index", str(board), "--leg", str(leg),
           "--output", str(directory), "--max-seconds", str(seconds)]
    with (directory / "worker.log").open("x", encoding="utf-8") as stream:
        subprocess.run(cmd, cwd=REPO, env={**os.environ, "PYTHONHASHSEED": "0"},
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=seconds + 20)
    result = json.loads((directory / "result.json").read_text())
    if result["termination"] != "victory":
        raise RuntimeError(f"benchmark requires complete games: {directory}: {result['termination']}")
    if result["forbidden_engine_calls"]:
        raise RuntimeError("decision attempted forward simulation")
    return {**result, "match_directory": str(directory)}


def trajectory_digest(result):
    directory = Path(result["match_directory"])
    hasher = hashlib.sha256()
    with gzip.open(directory / "decisions.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = 0
        for line in stream:
            row = json.loads(line)
            row.pop("decision_seconds")
            assert row["forward_simulation_guard"] and not row["forbidden_engine_calls"]
            hasher.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
            rows += 1
    assert rows == result["actions"] == sum(result["guarded_decisions"].values())
    document = json.loads((directory / "game.json").read_text())
    assert len(document["action_records"]) == rows
    stable_result = {k: result[k] for k in (
        "seed", "board_index", "leg", "winner", "winner_policy", "termination", "actual_vp",
        "actions", "turns", "initial_map_hash", "initial_stream_hash", "policies_by_color",
        "seating_order", "policy_seats", "decision_action_counts", "guarded_decisions")}
    fields = ("map", "board", "player_state", "num_turns", "action_records", "resource_freqdeck", "development_listdeck")
    return {"decisions": hasher.hexdigest(), "result": digest(stable_result),
            "authoritative_game": digest({k: document[k] for k in fields})}


def benchmark(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write(root / "config.json", {
        "games_per_setting": args.games, "base_seed": args.seed, "workers": args.workers,
        "logical_cpus": os.cpu_count(), "max_seconds_per_game": args.max_seconds,
        "source_sha256": {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in (
            "tools/benchmark_no_lookahead_workers.py", "src/catanzero/observation_match.py",
            "src/catanzero/static_policies.py", "src/catanzero/tournament.py")},
    })
    source = args.source.resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    for policy in ("A_mixed", "B_terminal"):
        assert hashlib.sha256(Path(manifest[policy]["portable_path"]).read_bytes()).hexdigest() == manifest[policy]["portable_sha256"]
    write(root / "manifest.json", manifest)
    measurements, reference = [], None
    for workers in args.workers:
        phase = root / f"workers-{workers}"
        phase.mkdir()
        (phase / "matches").mkdir()
        write(phase / "manifest.json", manifest)
        jobs = []
        for index in range(args.games):
            directory = phase / "matches" / f"board_{index//2:04d}_leg_{index%2}"
            directory.mkdir()
            jobs.append((phase / "manifest.json", directory, index, args.seed, args.max_seconds))
        started = perf_counter()
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(play, *job) for job in jobs]):
                results.append(future.result())
                if len(results) % 8 == 0:
                    print(f"workers={workers}: {len(results)}/{args.games} games", flush=True)
        seconds = perf_counter() - started
        results.sort(key=lambda r: (r["board_index"], r["leg"]))
        (phase / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
        hashes = [trajectory_digest(r) for r in results]
        if reference is None:
            reference = hashes
        assert hashes == reference, "parallelism changed observations, scores, actions or outcomes"
        write(phase / "trajectory_hashes.json", hashes)
        wins = Counter(r["winner_policy"] for r in results)
        write(phase / "summary.json", {"games": args.games, "completed_games": args.games,
              "unfinished_games": 0, "groups": {p: {"wins": wins[p]} for p in ("A_mixed", "B_terminal", "F_static")}})
        # Outside timing: validate recorded observations and sampled outcomes.
        with (phase / "audit.log").open("x", encoding="utf-8") as log:
            subprocess.run([sys.executable, str(REPO / "tools/audit_observation_games.py"), str(phase)],
                           cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        measurement = {"workers": workers, "games": args.games, "completed_games": args.games,
                       "wall_seconds": seconds, "games_per_second": args.games/seconds,
                       "estimated_500_game_seconds": seconds*500/args.games,
                       "actions": sum(r["actions"] for r in results),
                       "same_gameplay_inputs_scores_and_outcomes": True, "replay_audit_passed": True}
        measurements.append(measurement)
        write(root / "measurements.json", measurements)
        print(json.dumps(measurement), flush=True)
    for m in measurements:
        m["speedup_vs_one_worker"] = measurements[0]["wall_seconds"]/m["wall_seconds"]
    best = min(measurements, key=lambda m: m["wall_seconds"])
    report = {"games_per_worker_setting": args.games, "base_seed": args.seed,
              "total_games_run": args.games*len(args.workers), "measurements": measurements,
              "fastest_tested_workers": best["workers"],
              "method": "Sequential measurements of identical seeds and seats; independent engine subprocess per game; audit outside timing.",
              "limitations": ["single pass per setting, affected by machine load",
                              "500-game times extrapolated, not a measured 500-game run",
                              "current A/B plus two F_static; FQI candidate has not been trained",
                              "times include game process startup and trace saving, exclude replay audits, training and MLflow upload"]}
    write(root / "benchmark_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO / "runs/mlflow-label-study-v2/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--seed", type=int, default=150000)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--max-seconds", type=float, default=120)
    args = parser.parse_args()
    if args.games < 8 or args.games % 8 or args.workers[0] != 1 or len(set(args.workers)) != len(args.workers) or any(w < 1 for w in args.workers):
        parser.error("games must be a positive multiple of eight; distinct positive workers starting with 1")
    benchmark(args)
