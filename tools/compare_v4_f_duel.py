"""Paired two-player factorial evaluation, frozen v4 and native F search."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import mlflow
from scipy.stats import binomtest
from train_three_game_sarsa import exclusions
from train_online_sarsa import REPO, write
from catanzero.duel_protocol import CONDITIONS, wilson_interval


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_job(root, condition, index, max_seconds):
    output = root / condition / f"game_{index:04d}"
    if (output / "result.json").exists():
        return json.loads((output / "result.json").read_text())
    if output.exists():
        previous = output.with_name(output.name + "_failed_attempt")
        assert output.resolve().is_relative_to(root.resolve())
        assert previous.resolve().is_relative_to(root.resolve())
        output.rename(previous)
    command = [str(REPO / ".venv/Scripts/python.exe"), "-u", "-m", "catanzero.duel_search",
               "--config", str(root / "config.json"), "--output", str(output),
               "--condition", condition, "--index", str(index)]
    with (root / f"worker_{condition}_{index:04d}.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                       timeout=max_seconds + 60, check=True,
                       env={**os.environ, "PYTHONHASHSEED": "0"})
    return json.loads((output / "result.json").read_text())


def summarize(rows, games):
    groups = {}
    for condition in CONDITIONS:
        group = [r for r in rows if r["condition"] == condition]
        complete = [r for r in group if r["termination"] == "victory"]
        wins = sum(r["candidate_won"] is True for r in complete)
        unfinished = len(group) - len(complete)
        timing = {}
        for who in ("v4", "F"):
            count = sum(r["timing"][who]["count"] for r in group)
            timing[who] = {"decisions": count,
                           "mean_seconds": sum(r["timing"][who]["seconds"] for r in group) / count if count else None}
        groups[condition] = {"scheduled": games, "finished": len(group), "completed": len(complete),
                             "unfinished": unfinished, "v4_wins": wins,
                             "F_wins": len(complete) - wins,
                             "v4_win_rate": wins / games if len(group) == games and not unfinished else None,
                             "scheduled_win_bounds": [wins/games, (wins+games-len(complete))/games],
                             "wilson_95": wilson_interval(wins, games) if len(group) == games and not unfinished else None,
                             "timing": timing,
                             "search_stats": {key: sum(r["search_stats"][key] for r in group)
                                              for key in (group[0]["search_stats"] if group else [])}}
    pairs = {}
    lookup = {(r["condition"], r["game_index"]): r for r in rows}
    for first, second in (("neither", "f_only"), ("f_only", "both"),
                           ("neither", "v4_only"), ("v4_only", "both"), ("neither", "both")):
        plus = minus = complete = 0
        for index in range(games):
            a, b = lookup.get((first, index)), lookup.get((second, index))
            if a is None or b is None:
                continue
            for key in ("seed", "candidate_seat", "candidate_color", "initial_map_hash", "initial_stream_hash"):
                assert a[key] == b[key], (key, index)
            if a["termination"] == b["termination"] == "victory":
                complete += 1
                plus += b["candidate_won"] and not a["candidate_won"]
                minus += a["candidate_won"] and not b["candidate_won"]
        pairs[f"{first}_to_{second}"] = {"paired_complete": complete,
                                        "v4_gained_wins": int(plus), "v4_lost_wins": int(minus),
                                        "paired_p_unadjusted": float(binomtest(plus, plus+minus, .5).pvalue) if plus+minus else 1.}
    return {"conditions": groups, "paired": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--recover", action="store_true", help="reuse completed games; retry only missing results")
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or args.workers < 1:
        parser.error("positive workers and even games required")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.recover)
    source = REPO / "runs/online-sarsa-features-100-v4"
    excluded, manifest = exclusions(source)
    previous_config = json.loads((root / "config.json").read_text()) if args.recover else None
    if previous_config:
        assert previous_config["games_per_condition"] == args.games
        excluded = previous_config["excluded_board_hashes"]
    hashes = {name: digest(source / name) for name in ("linear_model.json", "normalizer.json", "model.joblib")}
    for name in ("linear_model.json", "normalizer.json"):
        shutil.copyfile(source / name, root / name)
    config = {"algorithm": "frozen v4 versus native one-action F search, factorial two-player evaluation",
              "source": str(source), "source_hashes": hashes,
              "model": str(root / "linear_model.json"), "normalizer": str(root / "normalizer.json"),
              "players": 2, "target_vp": 10, "games_per_condition": args.games, "workers": args.workers,
              "conditions": {k: {"v4_depth": d, "F_policy": f} for k, (d, f) in CONDITIONS.items()},
              "v4_depth_definition": "1=direct Q, no simulation; 2=root action to next own decision plus max Q backup",
              "F_depth_definition": "F_static=no simulation; F=native ValueFunctionPlayer, one simulated action per legal choice",
              "search_continuation": "matches real opponent (F_static or native F), no recursive opponent search",
              "beam": 3, "beliefs": 2, "max_branch_actions": 24, "gamma": .999, "vp_potential_weight": .05,
              "base_seed": 42000000, "max_actions": 4000, "max_seconds": 180,
              "excluded_board_hashes": excluded, "weights_updated": False,
              "transfer": "same 264 inputs/weights/normalization, absent opponent slots zero in raw features",
              "rules": "standard Catanatron base engine with two players; no special official two-player variant rules",
              "hidden_information": "native F and v4 search use synthetic public-count beliefs and detached future RNG",
              "selection": "fixed protocol; no tuning, training, automatic model promotion or registry alias changes"}
    if args.recover:
        config["technical_recovery"] = "399 completed games retained; one failed belief-sampling game retried with identical seed, 4096 instead of 128 rejection attempts; successful first-128 samples unchanged"
    write(root / "config.json", config)
    for file in ("src/catanzero/features.py", "src/catanzero/static_policies.py", "src/catanzero/regression_search.py",
                 "src/catanzero/duel_search.py", "src/catanzero/duel_protocol.py", "src/catanzero/search_labels.py", "tools/compare_v4_f_duel.py"):
        dest = root / "implementation_snapshot" / file
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / file, dest)
    mlflow.set_tracking_uri(manifest["tracking_uri"])
    client = mlflow.MlflowClient()
    name = "CatanZero_V4_F_Duel_Search"
    experiment = client.get_experiment_by_name(name)
    eid = experiment.experiment_id if experiment else client.create_experiment(
        name, artifact_location=(REPO / "runs/mlflow/artifacts").as_uri())
    mlflow.set_experiment(experiment_id=eid)
    started = perf_counter()
    rows = []
    with mlflow.start_run(run_name="v4_vs_one_F_search_factorial_100") as run:
        write(root / "tracking.json", {"run_id": run.info.run_id, "tracking_uri": manifest["tracking_uri"]})
        mlflow.log_params({"source": str(source), "players": 2, "games_per_condition": args.games,
                           "workers": args.workers, "v4_beam": 3, "v4_beliefs": 2, "weights_updated": False})
        mlflow.log_artifact(str(root / "config.json"), "protocol_before_test")
        print(f"START: {args.games*4} games / {args.workers} workers", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool, (root / "results.jsonl").open("w", encoding="utf-8") as stream:
            jobs = [pool.submit(run_job, root, condition, i, config["max_seconds"])
                    for i in range(args.games) for condition in CONDITIONS]
            for future in as_completed(jobs):
                row = future.result()
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if len(rows) % 20 == 0:
                    partial = summarize(rows, args.games)
                    write(root / "progress.json", partial)
                    print(f"PROGRESS: {len(rows)}/{4*args.games}, elapsed={perf_counter()-started:.1f}s; "
                          + "; ".join(f"{k}: {v['v4_wins']} wins/{v['finished']} finished ({v['unfinished']} capped)"
                                     for k,v in partial["conditions"].items()), flush=True)
        assert hashes == {name: digest(source / name) for name in hashes}
        assert all(digest(root / name) == hashes[name] for name in ("linear_model.json", "normalizer.json"))
        report = {**summarize(rows, args.games), "elapsed_seconds": perf_counter()-started,
                  "weights_unchanged": True, "test_boards": len({r['initial_map_hash'] for r in rows}),
                  "limitations": ["v4 trained with four players; two-player zero-slot transfer is out of training distribution",
                                  "bounded sampled beam search, not full minimax; v4 depth and F depth use different definitions",
                                  "F_static and native F are different heuristics as well as different search mechanisms",
                                  "actual future draws may differ after decisions diverge",
                                  "paired p-values exploratory, unadjusted for five comparisons",
                                  "branch action statistics omit internal F candidate simulations"]}
        assert report["test_boards"] == args.games
        write(root / "report.json", report)
        for condition, group in report["conditions"].items():
            mlflow.log_metrics({f"{condition}_v4_wins": group["v4_wins"],
                                f"{condition}_unfinished": group["unfinished"]})
        mlflow.log_metric("elapsed_seconds", report["elapsed_seconds"])
        for name in ("config.json", "report.json", "results.jsonl", "linear_model.json", "normalizer.json"):
            mlflow.log_artifact(str(root / name), "evaluation")
        mlflow.log_artifacts(str(root / "implementation_snapshot"), "implementation_snapshot")
        print("RESULT: " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
