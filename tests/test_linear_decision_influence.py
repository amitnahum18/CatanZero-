import gzip
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from sklearn.linear_model import SGDRegressor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_linear_decisions import analyze


def test_large_shared_state_weight_cancels_and_action_removal_detects_rank_loss(tmp_path):
    source=tmp_path/'source';source.mkdir()
    norm={'feature_names':['shared_state','action_gain','action_cost'],'offset':[0,0,0],'scale':[1,1,1]}
    for path in (tmp_path,source):
        (path/'normalizer.json').write_text(json.dumps(norm))
    (tmp_path/'config.json').write_text(json.dumps({'resume_from':str(source)}))
    (source/'linear_model.json').write_text(json.dumps({'weights':[0,-1,-.2]}))
    (tmp_path/'linear_model.json').write_text(json.dumps({'weights':[100,1,-.2]}))
    schema={'groups':{'state':{'names':['shared_state']},'actions':{'names':['action_gain','action_cost']}}}
    (tmp_path/'feature_schema.json').write_text(json.dumps(schema))
    estimator=SGDRegressor();estimator.partial_fit(np.zeros((1,3)),[0])
    estimator.coef_=np.asarray([100.,1.,-.2]);estimator.intercept_=np.zeros(1)
    joblib.dump(estimator,tmp_path/'model.joblib')
    with gzip.open(tmp_path/'decision_cases.jsonl.gz','wt') as stream:
        stream.write(json.dumps({'raw_features':[[3,0,0],[3,1,1]],'initial_phase':False,
            'candidate_action_kinds':['END_TURN','BUILD_SETTLEMENT'],'game_index':0,'action_index':5})+'\n')
    report=analyze(tmp_path)
    assert report['ranking'][0]['feature']=='action_gain'
    assert report['ranking'][0]['fixed_model_feature_removal_switch_fraction']==1
    assert report['ranking'][-1]['feature']=='shared_state'
    assert report['ranking'][-1]['mean_absolute_candidate_difference_contribution']==0
    assert report['state_only_in_sample']==['shared_state']
    assert report['new_vs_old_greedy_decision_change_fraction']==1
