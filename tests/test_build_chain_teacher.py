import gzip
import pickle
import json
from pathlib import Path
from catanzero.label_f_policy import label_game

REPO=Path(__file__).resolve().parents[1]


def test_road_to_settlement_annotation_improves_known_build_chain_without_changing_inputs(tmp_path):
    source=REPO/'runs/tree-policy-native-f-max-v1/RL_round_1/game_0000'
    info=label_game(source,tmp_path/'enhanced',trade_followup=True,build_followup=True)
    original=REPO/'runs/tree-policy-trade-aware-v3/labels/round_4_game_0000/labels.pkl.gz'
    with gzip.open(original,'rb') as stream:old=pickle.load(stream)
    with gzip.open(tmp_path/'enhanced/labels.pkl.gz','rb') as stream:new=pickle.load(stream)
    assert len(old)==len(new) and info['road_build_followups']>0
    for a,b in zip(old,new):
        assert a['features']==b['features']
        assert b['action_index']==a['action_index']
    a=next(r for r in old if r['action_index']==204);b=next(r for r in new if r['action_index']==204)
    names=json.loads((REPO/'runs/tree-policy-trade-aware-v3/normalizer.json').read_text())['feature_names']
    roads=[i for i,x in enumerate(a['features']) if x[names.index('action_BUILD_ROAD')]]
    assert max(b['score'][i]-a['score'][i] for i in roads)>1e14
