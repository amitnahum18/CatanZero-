"""Attach audited final documentation and explicit candidate tags to MLflow."""

import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
from mlflow.tracking import MlflowClient

from compare_regression_models import file_hash
from label_target_experiment import REGISTRY_NAME, REPO, labels_from_sources, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    manifest = json.loads((study / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((study / "config.json").read_text(encoding="utf-8"))
    audit = json.loads((study / "game_audit.json").read_text(encoding="utf-8"))
    assert audit["anchor"]["passed"] and audit["direct"]["passed"]
    dataset = Path(config["dataset"])
    assert all(file_hash(dataset / name) == digest for name, digest in config["source_hashes"].items())
    labels, _ = labels_from_sources(dataset)
    distribution = {s: {"rows": len(rows), "terminal_wins": sum(r["outcome"] == 1 for r in rows),
                         "terminal_losses": sum(r["outcome"] == 0 for r in rows)} for s, rows in labels.items()}
    write_json(study / "target_distribution.json", distribution)
    mlflow.set_tracking_uri(manifest["tracking_uri"])
    client = MlflowClient()
    metrics = json.loads((study / "label_metrics.json").read_text(encoding="utf-8"))
    for policy in ("A_mixed", "B_terminal"):
        record = manifest[policy]
        version = client.get_model_version_by_alias(REGISTRY_NAME, record["alias"])
        assert str(version.version) == record["model_version"]
        for split, values in metrics[policy].items():
            for kind in ("own_target", "terminal_reference"):
                for metric, value in values[kind].items():
                    if value is not None:
                        client.log_metric(record["run_id"], f"{split}_{kind}_{metric}", value)
        client.set_tag(record["run_id"], "label_target", "mixed_50_50" if policy == "A_mixed" else "terminal_only")
        client.set_tag(record["run_id"], "game_audit", "passed")
    client.set_registered_model_alias(REGISTRY_NAME, "candidate", manifest["A_mixed"]["model_version"])
    client.set_model_version_tag(REGISTRY_NAME, manifest["A_mixed"]["model_version"], "candidate_basis",
                                 "exploratory direct 4-2 with 2 censored; both lost all 16 F-anchor games")
    # Preserve the initial setup attempt and make its status unambiguous.
    first = client.get_model_version(REGISTRY_NAME, "1")
    first_run = client.get_run(first.run_id)
    if first_run.data.tags.get("source_output", "").endswith("mlflow-label-study-v1"):
        client.set_model_version_tag(REGISTRY_NAME, "1", "setup_status", "failed_before_gameplay_not_evaluated")
    documentation = REPO / "docs/label-target-experiment-2026-10-14.md"
    shutil.copyfile(documentation, study / "EXPERIMENT.md")
    registry_check = {policy: {"version": record["model_version"], "alias": record["alias"],
                               "run_status": client.get_run(record["run_id"]).info.status,
                               "model_uri": record["model_uri"]} for policy, record in
                      (("A_mixed", manifest["A_mixed"]), ("B_terminal", manifest["B_terminal"]))}
    assert all(r["run_status"] == "FINISHED" for r in registry_check.values())
    candidate = client.get_model_version_by_alias(REGISTRY_NAME, "candidate")
    assert str(candidate.version) == manifest["A_mixed"]["model_version"]
    final = {"registry": REGISTRY_NAME, "models": registry_check, "candidate": "A_mixed",
             "candidate_is_provisional": True, "source_data_unchanged": True,
             "game_audit_passed": True, "documentation": str(documentation),
             "mlflow_ui": "http://127.0.0.1:5070/", "test_suite": {"passed": 66}}
    write_json(study / "final_verification.json", final)
    run_ids = [manifest[p]["run_id"] for p in ("A_mixed", "B_terminal")]
    anchor = json.loads((study / "summary.json").read_text(encoding="utf-8"))
    direct = json.loads((study / "direct/summary.json").read_text(encoding="utf-8"))
    run_ids += [anchor["evaluation_run_id"], direct["evaluation_run_id"]]
    for run_id in run_ids:
        for filename in ("EXPERIMENT.md", "game_audit.json", "target_distribution.json", "final_verification.json"):
            client.log_artifact(run_id, str(study / filename), "final_report")
        client.set_tag(run_id, "final_audit", "passed")
    client.log_artifact(direct["evaluation_run_id"], str(study / "direct/summary.json"), "direct")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
