import json
from pathlib import Path
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'tools'))
from optimize_tree_policy_vs_f import TreePolicy,fit_ranker,monte_carlo_data


def test_ranker_trains_depth10_and_policy_scores_are_stable(tmp_path):
    raw=dict(np.load(REPO/'runs/tree-policy-native-f-max-v1/teacher_data.npz'))
    ranges=np.r_[0,np.cumsum(raw['groups'])]
    train=np.flatnonzero(~raw['validation_groups'])[:20]
    val=np.flatnonzero(raw['validation_groups'])[:10]
    groups=np.r_[train,val]
    data={'x':np.concatenate([raw['x'][ranges[i]:ranges[i+1]] for i in groups]),
          'y':np.concatenate([raw['y'][ranges[i]:ranges[i+1]] for i in groups]),
          'groups':raw['groups'][groups],'validation_groups':np.r_[np.zeros(len(train),dtype=bool),np.ones(len(val),dtype=bool)]}
    booster,scale,info=fit_ranker(data,{'num_leaves':7,'min_child_samples':2,'reg_lambda':1.})
    assert info['best_iterations']>0 and scale>0
    norm=json.loads((REPO/'runs/online-sarsa-features-100-v4/normalizer.json').read_text())
    model=TreePolicy(booster,norm,scale)
    model.save(tmp_path)
    import lightgbm as lgb
    other=TreePolicy(lgb.Booster(model_file=str(tmp_path/'ranker.txt')),norm,scale)
    np.testing.assert_array_equal(model.predict(data['x']),other.predict(data['x']))


def test_monte_carlo_return_labels_come_from_real_own_terminal_results():
    root=REPO/'runs/residual-extratrees-depth10-f-duel-v1'
    x,y,w,ids=monte_carlo_data(root,['training_round_1'])
    assert x.shape[1]==264 and len(x)==len(y)==len(ids)
    for target,item in zip(y,ids):
        assert target==(1. if item['won'] else -1.)-item['phi']
        assert item['phase']=='training_round_1'
    assert w.min()>0 and np.isfinite(x).all()
