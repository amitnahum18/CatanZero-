"""Fresh paired gameplay comparison of two frozen linear models against F_static."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
import joblib
import mlflow
from scipy.stats import binomtest
from train_online_sarsa import play_phase, write, REPO
from refit_linear_returns import FrozenLinear
from catanzero.board_exclusions import normalization_source_board_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.games < 4 or args.games % 4 or args.workers < 1:
        parser.error("game count divisible by four and positive workers required")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    sources = {"baseline": REPO / "runs/online-sarsa-low-exploration-500-v3",
               "expanded": REPO / "runs/online-sarsa-features-100-v4"}
    manifest = json.loads((REPO / "runs/mlflow-label-study-v2/manifest.json").read_text())
    excluded = list(manifest["source_board_hashes"])
    expansion_config = json.loads((sources["expanded"] / "config.json").read_text())
    excluded += normalization_source_board_hashes(Path(expansion_config["new_normalization_source"]))
    excluded += json.loads((REPO / "runs/fqi-pilot-500-v1/experience_audit.json").read_text())["collection"]["board_hashes"]
    ancestor = sources["expanded"]
    seen = set()
    while ancestor:
        assert str(ancestor) not in seen
        seen.add(str(ancestor))
        for phase in ("initial_evaluation", "live_training", "final_evaluation"):
            excluded.extend(json.loads(s)["initial_map_hash"] for s in (ancestor / phase / "results.jsonl").read_text().splitlines())
        parent = json.loads((ancestor / "config.json").read_text()).get("resume_from")
        ancestor = Path(parent) if parent else None
    for name in ("linear-monte-carlo-refit-v2", "gbrl-mixed-fast-200-v1"):
        for phase in ("initial_evaluation", "live_training", "final_evaluation"):
            path = REPO / "runs" / name / phase / "results.jsonl"
            if path.exists():
                excluded.extend(json.loads(s)["initial_map_hash"] for s in path.read_text().splitlines())
    path = REPO / "runs/regression-search-depth-1-vs-2-v1/results.jsonl"
    excluded.extend(json.loads(s)["initial_map_hash"] for s in path.read_text().splitlines())
    models, hashes = {}, {}
    for label, source in sources.items():
        destination = root / label
        destination.mkdir()
        for name in ("model.joblib", "linear_model.json", "normalizer.json", "registration.json"):
            shutil.copyfile(source / name, destination / name)
        hashes[label] = hashlib.sha256((source / "model.joblib").read_bytes()).hexdigest()
        norm = json.loads((destination / "normalizer.json").read_text())
        models[label] = FrozenLinear(joblib.load(destination / "model.joblib"), norm, 0)
    config = {"comparison": "frozen original 164-input SARSA v3 vs expanded 264-input SARSA v4",
              "sources": {k: str(p) for k, p in sources.items()}, "source_model_sha256": hashes,
              "games_per_model": args.games, "workers": args.workers, "base_seed": 20000000,
              "opponent": "three frozen F_static", "weights_updated": False, "forward_simulation": False,
              "paired": "same fresh initial maps, seats and chance-stream seeds; future draws can diverge by action",
              "excluded_board_hashes": sorted(set(excluded)),
              "no_tuning": "no training or model selection on this evaluation; original registry aliases unchanged"}
    write(root / "config.json", config)
    mlflow.set_tracking_uri(manifest["tracking_uri"])
    client = mlflow.MlflowClient()
    name = "CatanZero_Frozen_Linear_Comparison"
    experiment = client.get_experiment_by_name(name)
    eid = experiment.experiment_id if experiment else client.create_experiment(name, artifact_location=(REPO / "runs/mlflow/artifacts").as_uri())
    mlflow.set_experiment(experiment_id=eid)
    opponents = {"opponent_policy": "F_static", "max_actions": 2000, "max_seconds": 90, "trace_compresslevel": 1}
    rows, summaries = {}, {}
    started = perf_counter()
    with mlflow.start_run(run_name=f"baseline_v3_vs_feature_v4_{args.games}_paired_games") as run:
        write(root / "tracking.json", {"run_id": run.info.run_id, "tracking_uri": manifest["tracking_uri"]})
        mlflow.log_params({"games_per_model": args.games, "baseline_inputs": 164, "expanded_inputs": 264,
                           "weights_updated": False, "forward_simulation": False, "opponent": "F_static", "workers": args.workers})
        mlflow.log_artifact(str(root / "config.json"), "protocol_before_play")
        for label in ("baseline", "expanded"):
            print(f"STAGE: {label}, {args.games} frozen games vs three F_static", flush=True)
            rows[label], summaries[label] = play_phase(root, f"{label}_evaluation", models[label], args.games, args.workers,
                    training=False, base_seed=config["base_seed"], excluded=config["excluded_board_hashes"],
                    opponent_config=opponents, learner_policy=f"Linear_SARSA_{label}")
        plus = minus = pairs = 0
        for a, b in zip(rows['baseline'], rows['expanded']):
            assert (a['seed'], a['initial_map_hash'], a['learner_seat']) == (b['seed'], b['initial_map_hash'], b['learner_seat'])
            if a['termination'] == b['termination'] == 'victory':
                pairs += 1
                plus += b['learner_won'] and not a['learner_won']
                minus += a['learner_won'] and not b['learner_won']
        for label, source in sources.items():
            assert hashlib.sha256((source / "model.joblib").read_bytes()).hexdigest() == hashes[label]
            assert hashlib.sha256((root / label / "model.joblib").read_bytes()).hexdigest() == hashes[label]
        before, after = summaries['baseline'], summaries['expanded']
        difference = (after['learner_wins']-before['learner_wins'])/args.games
        report = {"baseline": before, "expanded": after, "complete_pairs": pairs,
                  "expanded_only_wins": int(plus), "baseline_only_wins": int(minus),
                  "paired_two_sided_p": float(binomtest(plus, plus+minus, .5).pvalue) if plus+minus else 1.,
                  "win_rate_difference_percentage_points": 100*difference,
                  "relative_win_count_change": after['learner_wins']/before['learner_wins']-1 if before['learner_wins'] else None,
                  "source_models_unchanged": True, "total_seconds": perf_counter()-started,
                  "limitations": ["model versus three F_static; not direct head-to-head between learned models",
                                  "prior feature additions and 100 extra training games differ together",
                                  "capped games remain unknown", "no calibrated human-level rating",
                                  "no independent full engine replay audit of this evaluation"]}
        write(root / "report.json", report)
        (root / "README.md").write_text(
            "# Frozen linear comparison against three F_static\n\n"
            f"200 planned fresh paired test boards ({args.games} per model), balanced seats, unchanged weights, no forward simulation. "
            "Baseline is the previously selected 164-feature SARSA v3, not the earliest zero-start model. Expanded is 264-feature SARSA v4.\n\n"
            f"Baseline wins: {before['learner_wins']}/{args.games}; expanded wins: {after['learner_wins']}/{args.games}. "
            f"Unfinished: {before['unfinished_games']} / {after['unfinished_games']}. "
            f"Expanded-only wins: {plus}; baseline-only wins: {minus}; paired p={report['paired_two_sided_p']:.6g}.\n\n"
            "This is gameplay evaluation only, without training or automatic promotion. Feature changes and prior additional training are not isolated. "
            "No independent full engine replay audit of these games has been performed.\n", encoding="utf-8")
        for path in root.iterdir():
            if path.is_file():
                mlflow.log_artifact(str(path), "frozen_comparison")
        for label in sources:
            mlflow.log_artifacts(str(root / label), f"frozen_models/{label}")
        mlflow.log_metrics({"baseline_wins": before['learner_wins'], "expanded_wins": after['learner_wins'],
                            "win_difference_pp": 100*difference, "paired_p": report['paired_two_sided_p']})
        print("RESULT: "+json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
