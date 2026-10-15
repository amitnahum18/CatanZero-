from pathlib import Path
import json
from catanzero.label_f_policy import label_game

REPO=Path(__file__).resolve().parents[1]


def test_trade_followup_changes_known_missed_city_without_changing_features(tmp_path):
    source=REPO/'runs/residual-extratrees-depth10-f-duel-v1/training_round_1/game_0039'
    label_game(source,tmp_path/'old',False)
    label_game(source,tmp_path/'new',True)
    import gzip,pickle
    with gzip.open(tmp_path/'old/labels.pkl.gz','rb') as stream:old=pickle.load(stream)
    with gzip.open(tmp_path/'new/labels.pkl.gz','rb') as stream:new=pickle.load(stream)
    a=next(r for r in old if r['action_index']==247)
    b=next(r for r in new if r['action_index']==247)
    assert a['features']==b['features']
    names=json.loads((REPO/'runs/tree-policy-dagger-max-v2/normalizer.json').read_text())['feature_names']
    assert a['features'][a['teacher_index']][names.index('action_END_TURN')]==1
    assert b['features'][b['teacher_index']][names.index('action_MARITIME_TRADE')]==1
    assert b['score'][b['teacher_index']]>a['score'][a['teacher_index']]+1e14
