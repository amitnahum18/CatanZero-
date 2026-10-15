"""Run frozen models against observation-only heuristic; track in MLflow."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
from mlflow.tracking import MlflowClient

REPO = Path(__file__).resolve().parents[1]
ENGINE = REPO / ".venv/Scripts/python.exe"


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def job(manifest_path, directory, board, leg, seed, seconds):
    command = [str(ENGINE), "-u", "-m", "catanzero.observation_match", str(manifest_path),
               "--seed", str(seed), "--board-index", str(board), "--leg", str(leg),
               "--output", str(directory), "--max-seconds", str(seconds)]
    with (directory / "worker.log").open("x", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=REPO, env={**os.environ, "PYTHONHASHSEED": "0"},
                                   stdout=stream, stderr=subprocess.STDOUT, timeout=seconds + 20)
    if completed.returncode:
        raise RuntimeError(f"worker failed: {directory}")
    result = json.loads((directory / "result.json").read_text())
    if result["termination"] == "error":
        raise RuntimeError(f"match failed: {result['error']}")
    return {**result, "worker_exit_code": completed.returncode, "match_directory": str(directory)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO / "runs/mlflow-label-study-v2/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boards", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=140000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=120)
    args = parser.parse_args()
    if args.boards < 4 or args.boards % 4 or args.workers < 1 or args.max_seconds <= 0:
        parser.error("boards must be a positive multiple of four; positive workers/time")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "matches").mkdir()
    manifest = json.loads(args.source.read_text(encoding="utf-8"))
    assert manifest["models_saved_before_gameplay"]
    original_hashes = {}
    for policy in ("A_mixed", "B_terminal"):
        path = Path(manifest[policy]["portable_path"])
        original_hashes[str(path)] = sha(path)
        assert original_hashes[str(path)] == manifest[policy]["portable_sha256"]
    normalizer = Path(manifest["normalizer_path"])
    original_hashes[str(normalizer)] = sha(normalizer)
    manifest.update(lineup="A,B,F_static,F_static", decision_protocol="observation_only_no_forward_simulation")
    write(root / "manifest.json", manifest)
    static = subprocess.run([str(ENGINE), "-c", "import json; from catanzero.static_policies import STATIC_WEIGHTS,STATIC_VERSION; print(json.dumps({'version':STATIC_VERSION,'weights':STATIC_WEIGHTS}))"],
                            cwd=REPO, capture_output=True, text=True, check=True)
    config = {"boards": args.boards, "legs_per_board": 2, "scheduled_games": args.boards * 2,
              "base_seed": args.base_seed, "workers": args.workers, "max_seconds": args.max_seconds,
              "max_turns": 1000, "max_actions": 3000, "lineup": manifest["lineup"],
              "models_retrained": False, "model_and_normalizer_sha256": original_hashes,
              "static_heuristic": json.loads(static.stdout), "guarded_methods": ["Game.copy", "Game.execute", "Game.__init__"],
              "policy_inputs": ["static map", "actor whitelist observation", "legal action intents"],
              "source_sha256": {name: sha(REPO / name) for name in (
                  "src/catanzero/static_policies.py", "src/catanzero/observation_match.py", "tools/no_lookahead_experiment.py")},
              "limitations": ["F_static is a new fixed heuristic, not original F with identical decisions",
                              "two F_static seats versus one of each learned model; baseline wins pooled",
                              "small paired pilot; does not identify the best possible nonsearch model",
                              "historical training data and labels may use simulation; restriction is decision-time"]}
    write(root / "config.json", config)
    mlflow.set_tracking_uri(manifest["tracking_uri"])
    client = MlflowClient()
    name = "CatanZero_NoLookahead_Comparison"
    experiment = client.get_experiment_by_name(name)
    eid = experiment.experiment_id if experiment else client.create_experiment(
        name, artifact_location=(REPO / "runs/mlflow/artifacts").as_uri())
    mlflow.set_experiment(experiment_id=eid)
    started, results = perf_counter(), []
    with mlflow.start_run(run_name="frozen_A_B_vs_F_static_paired") as active:
        run_id = active.info.run_id
        mlflow.log_params({"boards": args.boards, "scheduled_games": args.boards * 2, "base_seed": args.base_seed,
                           "workers": args.workers, "lineup": manifest["lineup"], "models_retrained": False,
                           "forward_simulation_allowed": False, "baseline_version": config["static_heuristic"]["version"],
                           "A_run_id": manifest["A_mixed"]["run_id"], "B_run_id": manifest["B_terminal"]["run_id"]})
        mlflow.log_artifact(str(root / "config.json"), "protocol_before_gameplay")
        jobs = []
        for board in range(args.boards):
            for leg in (0, 1):
                directory = root / "matches" / f"board_{board:04d}_leg_{leg}"
                directory.mkdir()
                jobs.append((root / "manifest.json", directory, board, leg, args.base_seed + board, args.max_seconds))
        with ThreadPoolExecutor(max_workers=args.workers) as pool, (root / "results.jsonl").open("x", encoding="utf-8") as stream:
            for future in as_completed([pool.submit(job, *j) for j in jobs]):
                result = future.result()
                results.append(result)
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                print(f"[{len(results)}/{len(jobs)}] board={result['board_index']} leg={result['leg']} {result['termination']} winner={result['winner_policy']} actions={result['actions']}", flush=True)
        complete = [r for r in results if r["termination"] == "victory"]
        wins = Counter(r["winner_policy"] for r in complete)
        unknown = len(results) - len(complete)
        groups = {}
        for p, seats in (("A_mixed", 1), ("B_terminal", 1), ("F_static", 2)):
            timings = [r["decision_seconds"][p] for r in results]
            n = sum(t["count"] for t in timings)
            groups[p] = {"wins": wins[p], "seats_per_game": seats,
                         "win_share_completed": wins[p] / len(complete) if complete else None,
                         "all_scheduled_win_bounds": [wins[p]/len(results), (wins[p]+unknown)/len(results)],
                         "mean_decision_seconds": sum(t["sum"] for t in timings)/n if n else None}
        summary = {"games": len(results), "completed_games": len(complete), "unfinished_games": unknown,
                   "termination_counts": dict(Counter(r["termination"] for r in results)), "groups": groups,
                   "actions": sum(r["actions"] for r in results), "forbidden_engine_calls": sum(len(r["forbidden_engine_calls"]) for r in results),
                   "elapsed_seconds": perf_counter()-started, "evaluation_run_id": run_id, "experiment_id": eid,
                   "tracking_uri": manifest["tracking_uri"], "lineup": manifest["lineup"],
                   "note": "F_static wins pooled over two seats; all policies act without engine transitions; unfinished games unknown."}
        write(root / "summary.json", summary)
        audited = subprocess.run([str(ENGINE), "-u", str(REPO / "tools/audit_observation_games.py"), str(root)],
                                 cwd=REPO, capture_output=True, text=True, check=True)
        print(audited.stdout, flush=True)
        assert all(sha(Path(path)) == expected for path, expected in original_hashes.items())
        mlflow.log_metrics({"completed_games": len(complete), "unfinished_games": unknown,
                            "forbidden_engine_calls": summary["forbidden_engine_calls"], "audit_passed": 1,
                            **{f"{p}_wins": g["wins"] for p, g in groups.items()}})
        mlflow.log_artifacts(str(root), "no_lookahead")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
