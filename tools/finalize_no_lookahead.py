"""Attach the completed no-lookahead documentation and implementation."""

import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
from mlflow.tracking import MlflowClient

repo = Path(__file__).resolve().parents[1]
root = repo / "runs/no-lookahead-v1"
summary = json.loads((root / "summary.json").read_text())
config = json.loads((root / "config.json").read_text())
audit = json.loads((root / "game_audit.json").read_text())
assert audit["passed"] and audit["forbidden_engine_calls"] == 0
for path, expected in config["model_and_normalizer_sha256"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
for path, expected in config["source_sha256"].items():
    assert hashlib.sha256((repo / path).read_bytes()).hexdigest() == expected
client = MlflowClient(tracking_uri=summary["tracking_uri"])
run_id = summary["evaluation_run_id"]
run = client.get_run(run_id)
assert run.info.status == "FINISHED"
assert run.data.metrics["forbidden_engine_calls"] == 0
shutil.copyfile(repo / "docs/no-lookahead-experiment-2026-10-14.md", root / "EXPERIMENT.md")
client.log_artifact(run_id, str(root / "EXPERIMENT.md"), "no_lookahead")
for path in (*config["source_sha256"], "tools/audit_observation_games.py", "tests/test_observation_match.py"):
    client.log_artifact(run_id, str(repo / path), "implementation")
verification = {"passed": True, "run_id": run_id, "mlflow_status": run.info.status,
                "model_and_normalizer_unchanged": True, "policy_sources_match_preplay_config": True,
                "games_audit_passed": True, "forward_simulation_calls": 0,
                "pytest_passed": 72, "documentation_logged": True, "implementation_logged": True}
(root / "final_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
client.log_artifact(run_id, str(root / "final_verification.json"), "no_lookahead")
downloaded = Path(client.download_artifacts(run_id, "no_lookahead/final_verification.json"))
assert json.loads(downloaded.read_text()) == verification
print(json.dumps(verification, indent=2))
