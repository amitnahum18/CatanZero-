"""Verify standalone loading; repair serving packaging without changing trees."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
from gbrl.models import GBTModel

REPO = Path(__file__).resolve().parents[1]


class StandaloneTreeActionValues(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from gbrl.models import GBTModel
        self.model = GBTModel.load_learner(context.artifacts["trees"], device="cpu")
        self.normalizer = json.loads(Path(context.artifacts["normalizer"]).read_text())

    def predict(self, context, model_input, params=None):
        x = np.asarray(model_input, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 164 or not np.isfinite(x).all():
            raise ValueError("Expected 164 finite raw features per candidate action")
        x = np.clip((x-self.normalizer["offset"])/self.normalizer["scale"], -4, 4)
        return self.model(np.ascontiguousarray(x, dtype=np.float32), requires_grad=False, tensor=False).reshape(-1)


def check_child(root, uri, tracking_uri, raw, expected):
    sample = root / "standalone_reload_sample.npz"
    np.savez_compressed(sample, raw=raw, expected=expected)
    source = "import mlflow,numpy as np; mlflow.set_tracking_uri("+repr(tracking_uri)+"); sample=np.load("+repr(str(sample))+"); model=mlflow.pyfunc.load_model("+repr(uri)+"); actual=model.predict(sample['raw']); assert np.array_equal(actual,sample['expected']); print('STANDALONE_RELOAD_EXACT')"
    return subprocess.run([str(REPO / ".ml-venv/Scripts/python.exe"), "-c", source], cwd=REPO,
                          capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONPATH": ""})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    root = args.study.resolve()
    read = lambda name: json.loads((root / name).read_text())
    registration, tracking, normalizer = read("registration.json"), read("tracking.json"), read("normalizer.json")
    original_hash = hashlib.sha256((root / "model.gbrl_model").read_bytes()).hexdigest()
    trees = GBTModel.load_learner(str(root / "model"), device="cpu")
    raw = np.asarray(normalizer["offset"])+np.random.default_rng(123).normal(size=(16, 164))*normalizer["scale"]
    x = np.ascontiguousarray(np.clip((raw-normalizer["offset"])/normalizer["scale"], -4, 4), dtype=np.float32)
    expected = trees(x, requires_grad=False, tensor=False).reshape(-1)
    result = check_child(root, registration["model_uri"], tracking["tracking_uri"], raw, expected)
    repaired = result.returncode != 0
    if repaired:
        # Only fix missing local training modules, not arbitrary serving errors.
        if "ModuleNotFoundError" not in result.stderr or "train_gbrl_sarsa" not in result.stderr:
            raise RuntimeError(result.stderr[-4000:])
        mlflow.set_tracking_uri(tracking["tracking_uri"])
        client = MlflowClient()
        requirements = [s for s in (REPO / "requirements-mlflow.txt").read_text().splitlines() if s and not s.startswith("#")]
        with mlflow.start_run(run_id=tracking["run_id"]):
            info = mlflow.pyfunc.log_model(name="portable_model", python_model=StandaloneTreeActionValues(),
                                          artifacts={"trees": str(root / "model.gbrl_model"), "normalizer": str(root / "normalizer.json")},
                                          pip_requirements=requirements+["gbrl==1.1.9", "torch==2.13.0"],
                                          signature=infer_signature(raw, expected), input_example=raw[:2])
            version = mlflow.register_model(info.model_uri, registration["registered_model"])
            client.set_registered_model_alias(registration["registered_model"], registration["alias"], str(version.version))
            registration["prior_serving_model_uri"] = registration["model_uri"]
            registration.update(version=str(version.version), model_uri=info.model_uri)
            (root / "registration.json").write_text(json.dumps(registration, indent=2))
            mlflow.set_tag("serving_packaging_repaired_without_retraining", True)
        result = check_child(root, registration["model_uri"], tracking["tracking_uri"], raw, expected)
    assert result.returncode == 0 and "STANDALONE_RELOAD_EXACT" in result.stdout
    assert original_hash == hashlib.sha256((root / "model.gbrl_model").read_bytes()).hexdigest()
    verification = {"passed": True, "standalone_reload_exact": True, "training_scripts_not_required_for_loading": True,
                    "packaging_repaired": repaired, "saved_tree_sha256_unchanged": original_hash,
                    "registered_model_version": registration["version"]}
    (root / "standalone_reload.json").write_text(json.dumps(verification, indent=2))
    client = MlflowClient(tracking_uri=tracking["tracking_uri"])
    client.log_artifact(tracking["run_id"], str(root / "standalone_reload.json"), "verification")
    client.log_artifact(tracking["run_id"], str(root / "registration.json"), "verification")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
