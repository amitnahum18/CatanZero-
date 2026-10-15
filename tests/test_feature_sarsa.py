import json
from pathlib import Path
import sys
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from train_feature_sarsa import expanded_normalizer
from train_online_sarsa import CentralSARSA, play_phase


def test_feature_expansion_preserves_old_values_and_learns_new_weights(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    source = repo / "runs/online-sarsa-low-exploration-500-v3"
    norm = expanded_normalizer(source, repo / "runs/regression-build-capacity-v3")
    old = joblib.load(source / "model.joblib")
    model = CentralSARSA(tmp_path, norm, expand_from=source)
    assert model.input_columns == 264
    assert np.array_equal(model.model.coef_[:164], old.coef_)
    assert not model.model.coef_[164:].any() and model.model.t_ == old.t_
    raw = np.asarray(norm['offset'])+np.random.default_rng(1).normal(size=(8,264))*norm['scale']
    x = model.normalize(raw)
    np.testing.assert_allclose(model.model.predict(x), old.predict(x[:,:164]), atol=1e-12, rtol=0)
    terminal_x = np.zeros(264)
    terminal_x[-1] = 1
    model.add(terminal_x, np.zeros(264), 1., 0., {})
    model.flush(force=True)
    assert model.model.coef_[-1] > 0
    model.trace.close()


def test_real_264_column_environments_deliver_guarded_sarsa_updates(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    source = repo / "runs/online-sarsa-low-exploration-500-v3"
    norm = expanded_normalizer(source, repo / "runs/regression-build-capacity-v3")
    model = CentralSARSA(tmp_path, norm, expand_from=source)
    rows, summary = play_phase(tmp_path, "smoke", model, 4, 2, training=True, base_seed=19000000,
                               excluded=[], opponent_config={"max_actions":200,"trace_compresslevel":1})
    assert len(rows)==4 and model.version>0 and model.transitions>0
    assert summary['forbidden_engine_calls']==0
    assert len(model.model.coef_)==264
    model.trace.close()
