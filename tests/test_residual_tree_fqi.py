import json
from pathlib import Path
import sys
import joblib
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"tools"))
from train_residual_tree_fqi import ResidualTrees,fit_iteration,play_phase


def scorer():
    source=REPO/"runs/online-sarsa-features-100-v4"
    return ResidualTrees(json.loads((source/"linear_model.json").read_text()),
                         json.loads((source/"normalizer.json").read_text()))


def test_initial_transformed_scores_keep_v4_action_order_and_fits_preserve_anchor(tmp_path):
    model=scorer()
    original=joblib.load(REPO/"runs/online-sarsa-features-100-v4/model.joblib")
    raw=np.tile(np.asarray(model.normalizer['offset']),(64,1))
    raw[:,model.vp_index]=4;raw[:,model.enemy_index]=5
    action=model.names.index("action_BUY_DEVELOPMENT_CARD")
    raw[32:,action]=1
    source_q=original.predict(model.normalize(raw))
    assert np.argmax(model.predict(raw))==np.argmax(source_q)
    np.testing.assert_allclose(model.predict(raw)-2*source_q,model.predict(raw)[0]-2*source_q[0],atol=1e-12)
    before_base=model.weights.copy();old=model.predict(raw)
    data={"x":raw,"next_x":np.empty((0,264)),"ranges":np.zeros((64,2),dtype=np.int64),
          "reward":np.r_[np.repeat(-1.0,32),np.repeat(1.0,32)],"discount":np.zeros(64),
          "phi":np.zeros(64),"next_phi":np.zeros(64)}
    info=fit_iteration(model,data,tmp_path/"fit")
    assert info["actual_depth_max"]<=10 and model.version==1
    assert not np.array_equal(model.predict(raw),old)
    np.testing.assert_array_equal(model.weights,before_base)
    assert model.predict(raw)[-1] > model.predict(raw)[0]
    restored=ResidualTrees(model.base,model.normalizer,joblib.load(tmp_path/"fit/forest.joblib"))
    np.testing.assert_array_equal(restored.predict(raw),model.predict(raw))


def test_real_two_player_F_collection_and_depth10_fit_without_learner_simulation(tmp_path):
    model=scorer()
    rows,summary,data=play_phase(tmp_path,"smoke",model,4,2,training=True,
                                 base_seed=48000000,excluded=[],epsilon=.1)
    assert summary["completed"]==4 and all(r["players"]==2 and r["opponent_policy"]=="F" for r in rows)
    assert all(r["native_f_decisions"]>0 and r["forbidden_engine_calls"]==0 for r in rows)
    assert len(data["x"])>100 and data["x"].shape[1]==264
    for r,(a,b),d,p in zip(data["reward"],data["ranges"],data["discount"],data["phi"]):
        assert (-1-p-1e-12 <= r <= 1.7) and ((b>a)==bool(d))
    info=fit_iteration(model,data,tmp_path/"fit")
    assert info["actual_depth_max"]<=10 and info["mean_prediction_change"]>0
