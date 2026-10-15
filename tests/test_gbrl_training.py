import json
from pathlib import Path
import random
import sys

import pytest
pytest.importorskip("gbrl")
import numpy as np
from gbrl.models import GBTModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from train_gbrl_sarsa import GBRLSARSA, tree_predict
from train_online_sarsa import play_phase


def normalizer():
    return {"feature_names": [f"f{i}" for i in range(164)], "fit_split": "train",
            "offset": [0.] * 164, "scale": [1.] * 164}


def test_tree_updates_use_actual_next_action_and_preserve_prior_trees(tmp_path):
    model = GBRLSARSA(tmp_path, normalizer(), batch_size=32)
    inputs = np.random.default_rng(42).normal(size=(32, 164)).astype(np.float32)
    for i, x in enumerate(inputs):
        model.add(x, x, 1., .9, {"id": i})
    model.flush()
    assert model.version == 1 and model.transitions == 32
    x = inputs[0]
    assert model.predict([x])[0] > 0
    previous = model.predict([x])[0]
    for i, sample in enumerate(inputs):
        model.add(sample, sample, 1., .9, {"id": i+32})
    model.flush(force=True)
    assert model.predict([x])[0] > previous and model.version == 2
    restored = GBTModel.load_learner(str(tmp_path / "snapshots/version_000002"), device="cpu")
    assert np.array_equal(tree_predict(restored, [x]), model.predict([x]))
    model.trace.close()
    import gzip
    rows = [json.loads(line) for line in gzip.open(tmp_path / "updates.jsonl.gz", "rt")]
    assert abs(rows[1]["samples"][0]["target"]-(1.+.9*previous)) < 1e-7


def test_real_games_use_tree_label_updates_and_frozen_evaluation(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    norm = json.loads((repo / "runs/fqi-pilot-500-v1/normalizer.json").read_text())
    model = GBRLSARSA(tmp_path, norm)
    results, train = play_phase(tmp_path, "train", model, 4, 2, training=True,
                               base_seed=4600000, excluded=[], exploration_start=.1, exploration_end=.05,
                               learner_policy="GBRL_SARSA")
    assert train["completed_games"] == 4 and model.version > 0
    assert all(r["learner_policy"] == "GBRL_SARSA" for r in results)
    version = model.version
    _, frozen = play_phase(tmp_path, "eval", model, 4, 2, training=False,
                           base_seed=4700000, excluded=[r["initial_map_hash"] for r in results],
                           learner_policy="GBRL_SARSA")
    assert frozen["completed_games"] == 4 and model.version == version
    assert not frozen["weights_updated"] and frozen["forbidden_engine_calls"] == 0
    model.trace.close()


def test_resume_preserves_trees_and_next_sarsa_update(tmp_path):
    source, continuation = tmp_path / "source", tmp_path / "continuation"
    source.mkdir()
    continuation.mkdir()
    norm = normalizer()
    original = GBRLSARSA(source, norm, batch_size=32)
    x = np.random.default_rng(43).normal(size=(32, 164)).astype(np.float32)
    for row in x:
        original.add(row, row, 1., .9, {})
    original.flush()
    original.model.save_learner(str(source / "model"))
    (source / "normalizer.json").write_text(json.dumps(norm))
    resumed = GBRLSARSA(continuation, norm, batch_size=32, resume_from=source)
    assert resumed.version == 0 and resumed.initial_trees == 1
    assert np.array_equal(resumed.predict(x), original.predict(x))
    for model in (original, resumed):
        for row in x:
            model.add(row, row, .5, .9, {})
        model.flush()
        model.trace.close()
    assert np.array_equal(resumed.predict(x), original.predict(x))
