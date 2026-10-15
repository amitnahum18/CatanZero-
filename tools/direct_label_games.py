"""Exploratory direct 2A-vs-2B extension after the F-anchor pilot."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
from mlflow.tracking import MlflowClient

from label_target_experiment import EXPERIMENT_NAME, play_job, summarize, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    manifest = json.loads((study / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["models_saved_before_gameplay"]:
        raise ValueError("models not yet saved")
    output = study / "direct"
    output.mkdir(exist_ok=False)
    (output / "matches").mkdir()
    manifest["lineup"] = "AABB"
    write_json(output / "manifest.json", manifest)
    write_json(output / "config.json", {
        "lineup": ["A_mixed", "A_mixed", "B_terminal", "B_terminal"],
        "boards": 4, "legs_per_board": 2, "base_seed": 130000, "workers": 4,
        "max_seconds": 60, "max_actions": 3000, "max_turns": 1000,
        "phase": "exploratory secondary evaluation after F-anchor results",
        "models_retrained": False, "seat_swaps": True,
    })
    jobs = []
    for board in range(4):
        for leg in (0, 1):
            directory = output / "matches" / f"board_{board:04d}_leg_{leg}"
            directory.mkdir()
            jobs.append((output / "manifest.json", directory, board, leg, 130000 + board, 60))
    mlflow.set_tracking_uri(manifest["tracking_uri"])
    mlflow.set_experiment(experiment_id=manifest["experiment_id"])
    client = MlflowClient()
    results = []
    with mlflow.start_run(run_name="direct_2A_vs_2B_exploratory") as active:
        mlflow.log_params({"lineup": "A,A,B,B", "boards": 4, "scheduled_games": 8,
                           "A_run_id": manifest["A_mixed"]["run_id"], "B_run_id": manifest["B_terminal"]["run_id"],
                           "models_retrained": False, "seat_swaps": True, "base_seed": 130000})
        with ThreadPoolExecutor(max_workers=4) as pool, (output / "results.jsonl").open("x", encoding="utf-8") as stream:
            for future in as_completed([pool.submit(play_job, job) for job in jobs]):
                result = future.result()
                results.append(result)
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                print(f"[{len(results)}/8] board={result['board_index']} leg={result['leg']} {result['termination']} winner={result.get('winner_policy')} actions={result['actions']}", flush=True)
        summary = summarize(results, direct=True)
        summary.update(evaluation_run_id=active.info.run_id, phase="exploratory direct match", lineup="A,A,B,B")
        write_json(output / "summary.json", summary)
        mlflow.log_metrics({"completed_games": summary["completed_games"], "unfinished_games": summary["unfinished_games"],
                            **{f"{p}_wins": g["wins"] for p, g in summary["groups"].items()}})
        mlflow.log_artifacts(str(output), "direct")
        for p in ("A_mixed", "B_terminal"):
            client.set_tag(manifest[p]["run_id"], "direct_evaluation_run_id", active.info.run_id)
            client.log_metric(manifest[p]["run_id"], "direct_game_wins", summary["groups"][p]["wins"])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
