"""Paired bounded depth comparison on fresh boards; weights remain frozen."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import mlflow
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def run_job(config, root, depth, index, seconds):
    output = root / f"depth_{depth}" / f"game_{index:04d}"
    command = [str(REPO / ".venv/Scripts/python.exe"), "-u", "-m", "catanzero.regression_search",
               "--config", str(config), "--output", str(output), "--depth", str(depth), "--index", str(index)]
    with (root / f"worker_depth_{depth}_game_{index:04d}.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, timeout=seconds+30,
                       check=True, env={**os.environ, "PYTHONHASHSEED": "0"})
    return json.loads((output / "result.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.games < 4 or args.games % 4 or args.workers < 1:
        parser.error("balanced games divisible by four and positive workers required")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = REPO / "runs/online-sarsa-low-exploration-500-v3"
    manifest = json.loads((REPO / "runs/mlflow-label-study-v2/manifest.json").read_text())
    excluded = list(manifest["source_board_hashes"])
    excluded += json.loads((REPO / "runs/fqi-pilot-500-v1/experience_audit.json").read_text())["collection"]["board_hashes"]
    ancestor = source
    while ancestor:
        for phase in ("initial_evaluation", "live_training", "final_evaluation"):
            excluded.extend(json.loads(s)["initial_map_hash"] for s in (ancestor / phase / "results.jsonl").read_text().splitlines())
        parent = json.loads((ancestor / "config.json").read_text()).get("resume_from")
        ancestor = Path(parent) if parent else None
    for name in ("linear-monte-carlo-refit-v2", "gbrl-mixed-fast-200-v1"):
        for phase in ("initial_evaluation", "live_training", "final_evaluation"):
            path = REPO / "runs" / name / phase / "results.jsonl"
            if path.exists():
                excluded.extend(json.loads(s)["initial_map_hash"] for s in path.read_text().splitlines())
    shutil.copyfile(source / "linear_model.json", root / "linear_model.json")
    shutil.copyfile(source / "normalizer.json", root / "normalizer.json")
    config = {"algorithm": "public-belief beam search with frozen linear action values",
              "source": str(source), "model": str(root / "linear_model.json"), "normalizer": str(root / "normalizer.json"),
              "model_sha256": hashlib.sha256((root / "linear_model.json").read_bytes()).hexdigest(),
              "depth_definition": "depth 1: direct Q action ranking; depth 2: root action plus next own decision Q backup",
              "depth_2_continuation": "F_static opponents until next own decision; shaped reward + discounted max Q",
              "depth_2_approximation": "top 3 root moves, 2 sampled public-count beliefs, at most 24 real simulated actions per branch",
              "beam": 3, "beliefs": 2, "max_branch_actions": 24, "base_seed": 16000000,
              "max_actions": 2000, "max_seconds": 90, "excluded_board_hashes": sorted(set(excluded)),
              "gamma": .999, "vp_potential_weight": .05, "games_per_depth": args.games, "workers": args.workers,
              "weights_updated": False, "opponents": "three frozen F_static, no opponent forward simulation",
              "future_randomness": "independent synthetic belief/branch streams; actual future stream inaccessible to scoring",
              "selection": "fixed beam/belief budgets; no tuning or automatic model promotion based on test"}
    write(root / "config.json", config)
    for depth in (1, 2):
        write(root / f"depth_{depth}_policy.json", {"depth": depth, "model": "linear_model.json",
              "normalizer": "normalizer.json", "weights_updated": False,
              "beam": config["beam"], "beliefs": config["beliefs"],
              "max_branch_actions": config["max_branch_actions"],
              "definition": config["depth_definition"], "model_sha256": config["model_sha256"]})
    mlflow.set_tracking_uri(manifest["tracking_uri"])
    client = mlflow.MlflowClient()
    name = "CatanZero_Regression_Search_Depth"
    experiment = client.get_experiment_by_name(name)
    eid = experiment.experiment_id if experiment else client.create_experiment(name, artifact_location=(REPO / "runs/mlflow/artifacts").as_uri())
    mlflow.set_experiment(experiment_id=eid)
    started = perf_counter()
    rows = []
    with mlflow.start_run(run_name="frozen_linear_depth_1_vs_2_fast_pilot") as run:
        write(root / "tracking.json", {"run_id": run.info.run_id, "tracking_uri": manifest["tracking_uri"]})
        mlflow.log_params({"depths": "1,2", "beam": 3, "beliefs": 2, "games_per_depth": args.games,
                           "weights_updated": False, "source_model": str(source), "workers": args.workers})
        mlflow.log_artifact(str(root / "config.json"), "protocol_before_test")
        with ThreadPoolExecutor(max_workers=args.workers) as pool, (root / "results.jsonl").open("w", encoding="utf-8") as stream:
            jobs = [pool.submit(run_job, root / "config.json", root, depth, i, config["max_seconds"])
                    for i in range(args.games) for depth in (1, 2)]
            for future in as_completed(jobs):
                row = future.result()
                rows.append(row)
                stream.write(json.dumps(row)+"\n")
                stream.flush()
                if len(rows) % 4 == 0:
                    print(f"PROGRESS: {len(rows)}/{2*args.games} games, elapsed={perf_counter()-started:.1f}s", flush=True)
        summaries = {}
        for depth in (1, 2):
            group = [r for r in rows if r["depth"] == depth]
            complete = [r for r in group if r["termination"] == "victory"]
            wins = sum(r["candidate_won"] is True for r in complete)
            decisions = sum(r["decision_count"] for r in group)
            summaries[str(depth)] = {"scheduled": args.games, "completed": len(complete), "unfinished": args.games-len(complete),
                                      "wins": wins, "scheduled_win_bounds": [wins/args.games, (wins+args.games-len(complete))/args.games],
                                      "mean_choice_seconds": sum(r["decision_seconds_sum"] for r in group)/decisions if decisions else None,
                                      "max_choice_seconds": max(r["decision_seconds_max"] for r in group),
                                      "simulated_actions": sum(r["search_stats"]["simulated_actions"] for r in group),
                                      "searched_decisions": sum(r["search_stats"]["searched_decisions"] for r in group),
                                      "changed_choices": sum(r["search_stats"]["changed_choices"] for r in group),
                                      "budget_fallbacks": sum(r["search_stats"]["budget_fallbacks"] for r in group),
                                      "leaf_cache_hits": sum(r["search_stats"]["leaf_cache_hits"] for r in group)}
        by_pair = {(r["depth"], r["game_index"]): r for r in rows}
        positive = negative = paired_complete = 0
        for index in range(args.games):
            a, b = by_pair[1, index], by_pair[2, index]
            for key in ("seed", "candidate_seat", "candidate_color", "initial_map_hash", "initial_stream_hash"):
                assert a[key] == b[key]
            if a["termination"] == b["termination"] == "victory":
                paired_complete += 1
                positive += b["candidate_won"] and not a["candidate_won"]
                negative += a["candidate_won"] and not b["candidate_won"]
        assert config["model_sha256"] == hashlib.sha256((root / "linear_model.json").read_bytes()).hexdigest()
        report = {"depths": summaries, "paired_complete_games": paired_complete,
                  "depth_2_only_wins": int(positive), "depth_1_only_wins": int(negative),
                  "paired_two_sided_p": float(binomtest(positive, positive+negative, .5).pvalue) if positive+negative else 1.,
                  "elapsed_seconds": perf_counter()-started, "weights_unchanged": True,
                  "limitations": ["small pilot, no automatic promotion", "depth counts own decisions, not all table actions or complete turns",
                                  "only top 3 root candidates searched; may omit good moves", "2 public-count beliefs, not exact chance expectations",
                                  "terminal and chance outcomes simulated, true hidden cards and future RNG not used",
                                  "opponents continue under F_static, not adversarial minimax", "branch cutoff falls back to original Q for entire decision",
                                  "no independent full engine replay audit of these new games"]}
        write(root / "report.json", report)
        (root / "README.md").write_text(
            "# Frozen regression: depth 1 versus bounded depth 2\n\n"
            "Same SARSA linear regression weights and normalization in both variants. No new model training. "
            "Depth 1 chooses max Q directly. Depth 2 compares the top three Q-ranked moves by two sampled public-information branches, "
            "continuing frozen F_static opponents until the next own decision. It backs up the original shaped reward plus discounted max Q. "
            "This is a bounded sampled search, not full alpha-beta or four-turn planning.\n\n"
            f"Depth 1: {summaries['1']['wins']}/{args.games} wins, {summaries['1']['unfinished']} unfinished. "
            f"Depth 2: {summaries['2']['wins']}/{args.games} wins, {summaries['2']['unfinished']} unfinished. "
            f"Paired p={report['paired_two_sided_p']:.4g}.\n\n"
            "Candidate ordering is canonical, model weights frozen, all four seats balanced, boards fresh and paired, actual game RNG unchanged during decisions. "
            "Search samples synthetic hidden cards and independent future randomness. Small pilot; no automatic promotion.\n",
            encoding="utf-8")
        for file in ("src/catanzero/regression_search.py", "tools/compare_regression_search.py", "src/catanzero/online_linear.py"):
            dest = root / "implementation_snapshot" / file
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / file, dest)
        mlflow.log_metrics({"depth_1_wins": summaries['1']['wins'], "depth_2_wins": summaries['2']['wins'],
                            "depth_1_mean_choice_seconds": summaries['1']['mean_choice_seconds'],
                            "depth_2_mean_choice_seconds": summaries['2']['mean_choice_seconds'], "paired_p": report['paired_two_sided_p']})
        for path in root.iterdir():
            if path.is_file():
                mlflow.log_artifact(str(path), "depth_comparison")
        print("RESULT: "+json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
