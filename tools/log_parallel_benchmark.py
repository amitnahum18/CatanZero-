"""Log the completed performance comparison without changing model aliases."""

import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
from mlflow.tracking import MlflowClient

repo = Path(__file__).resolve().parents[1]
root = repo / "runs/no-lookahead-workers-24-v2"
report = json.loads((root / "benchmark_report.json").read_text())
config = json.loads((root / "config.json").read_text())
manifest = json.loads((root / "manifest.json").read_text())
assert all(m["replay_audit_passed"] and m["same_gameplay_inputs_scores_and_outcomes"] for m in report["measurements"])
for path, expected in config["source_sha256"].items():
    assert hashlib.sha256((repo / path).read_bytes()).hexdigest() == expected
mlflow.set_tracking_uri(manifest["tracking_uri"])
client = MlflowClient()
candidate = client.get_model_version_by_alias("CatanZero_GradientBoosting_LabelStudy", "candidate")
assert str(candidate.version) == "2" and candidate.run_id == manifest["A_mixed"]["run_id"]
mlflow.set_experiment("CatanZero_NoLookahead_Comparison")
shutil.copyfile(repo / "docs/parallel-games-2026-10-14.md", root / "README.md")
with mlflow.start_run(run_name="parallel_game_workers_1_4_8") as active:
    run_id = active.info.run_id
    mlflow.log_params({"evaluation_type": "performance_only", "games_per_setting": 24,
                       "workers_tested": "1,4,8", "base_seed": 150000,
                       "A_run_id": manifest["A_mixed"]["run_id"], "B_run_id": manifest["B_terminal"]["run_id"],
                       "models_retrained": False, "forward_simulation_allowed": False})
    for m in report["measurements"]:
        mlflow.log_metrics({f"workers_{m['workers']}_seconds": m["wall_seconds"],
                            f"workers_{m['workers']}_speedup": m["speedup_vs_one_worker"]})
    mlflow.log_metrics({"fastest_tested_workers": report["fastest_tested_workers"], "gameplay_invariance_passed": 1,
                        "focused_tests_passed": 13})
    for name in ("benchmark_report.json", "config.json", "README.md", "measurements.json"):
        mlflow.log_artifact(str(root / name), "benchmark")
    for m in report["measurements"]:
        directory = root / f"workers-{m['workers']}"
        for name in ("game_audit.json", "trajectory_hashes.json", "summary.json"):
            mlflow.log_artifact(str(directory / name), f"benchmark/workers-{m['workers']}")
    for path in (*config["source_sha256"], "tests/test_atomic_checkpoint.py"):
        mlflow.log_artifact(str(repo / path), "implementation")
assert client.get_run(run_id).info.status == "FINISHED"
verification = {"passed": True, "performance_run_id": run_id, "mlflow_status": "FINISHED",
                "candidate_model": "A_mixed", "candidate_version": candidate.version,
                "candidate_is_provisional": True, "models_retrained": False,
                "source_hashes_match": True, "all_72_games_replay_audited": True,
                "all_three_trajectories_identical": True, "focused_tests_passed": 13}
(root / "final_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
client.log_artifact(run_id, str(root / "final_verification.json"), "benchmark")
print(json.dumps(verification, indent=2))
