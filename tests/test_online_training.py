import json
from pathlib import Path
import sys

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("mlflow")
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from train_online_sarsa import CentralSARSA, play_phase, exploration_rate


def scaler():
    return {"feature_names": [f"f{i}" for i in range(164)], "fit_split": "train",
            "offset": [0.0]*164, "scale": [1.0]*164}


def test_lower_exploration_schedule_and_frozen_evaluation():
    assert exploration_rate(0, 500, True, .1, .05) == .1
    assert exploration_rate(499, 500, True, .1, .05) == .05
    assert exploration_rate(250, 500, False, .1, .05) == 0
    rates = [exploration_rate(i, 500, True, .1, .05) for i in range(500)]
    assert all(.05 <= value <= .1 for value in rates)
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    with pytest.raises(ValueError):
        exploration_rate(0, 500, True, .05, .1)


def test_central_updates_preserve_existing_weights_and_wait_for_batch(tmp_path):
    model = CentralSARSA(tmp_path, scaler(), batch_size=4)
    assert np.count_nonzero(model.model.coef_) == 0 and model.version == 0
    x = np.zeros(164)
    x[0] = 1
    for i in range(3):
        model.add(x, np.zeros(164), 1., 0., {"sample": i})
    model.flush()
    assert model.version == 0
    model.add(x, np.zeros(164), 1., 0., {"sample": 3})
    model.flush()
    first = model.model.coef_.copy()
    assert first[0] > 0 and model.version == 1 and model.transitions == 4
    model.add(x, np.zeros(164), 1., 0., {"sample": 4})
    model.flush(force=True)
    assert model.model.coef_[0] > first[0] and model.version == 2
    model.trace.close()


def test_actual_parallel_games_drive_central_learning_and_frozen_eval(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    norm = json.loads((repo / "runs/fqi-pilot-500-v1/normalizer.json").read_text())
    model = CentralSARSA(tmp_path, norm)
    results, summary = play_phase(tmp_path, "train", model, 4, 2, training=True,
                                  base_seed=1500000, excluded=[])
    assert len(results) == summary["completed_games"] == 4
    assert model.version > 0 and model.transitions > 0 and np.linalg.norm(model.model.coef_) > 0
    weights, version = model.model.coef_.copy(), model.version
    results, summary = play_phase(tmp_path, "eval", model, 4, 2, training=False,
                                  base_seed=1600000, excluded=[r["initial_map_hash"] for r in results])
    assert summary["completed_games"] == 4 and not summary["weights_updated"]
    assert np.array_equal(model.model.coef_, weights) and model.version == version
    model.trace.close()


def test_continuation_preserves_weights_optimizer_and_next_update(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "continued"
    source.mkdir()
    destination.mkdir()
    norm = scaler()
    original = CentralSARSA(source, norm)
    x = np.ones(164)
    original.add(x, np.zeros(164), 1., 0., {})
    original.flush(force=True)
    (source / "normalizer.json").write_text(json.dumps(norm))
    (source / "linear_model.json").write_text(json.dumps(original.document()))
    joblib.dump(original.model, source / "model.joblib")
    continued = CentralSARSA(destination, norm, resume_from=source)
    assert continued.version == 0 and continued.source_version == original.version
    assert np.array_equal(continued.model.coef_, original.model.coef_)
    assert continued.model.t_ == original.model.t_
    for model in (original, continued):
        model.add(x, x, .1, .9, {})
        model.flush(force=True)
        model.trace.close()
    assert np.array_equal(continued.model.coef_, original.model.coef_)
    assert np.array_equal(continued.model.intercept_, original.model.intercept_)
    with pytest.raises(ValueError, match="identical frozen normalization"):
        CentralSARSA(tmp_path / "invalid", {**norm, "scale": [2.] * 164}, resume_from=source)
