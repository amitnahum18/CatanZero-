import gzip
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from train_short_relative_sarsa import CheckpointSARSA
from train_online_sarsa import play_phase


def test_live_parallel_transition_rewards_use_only_observed_public_leader(tmp_path):
    repo=Path(__file__).resolve().parents[1];source=repo/'runs/online-sarsa-features-100-v4'
    norm=json.loads((source/'normalizer.json').read_text())
    model=CheckpointSARSA(tmp_path,norm,source,.005)
    results,summary=play_phase(tmp_path,'live_training',model,4,2,training=True,base_seed=36000000,
        excluded=[],gap_weight=.02,exploration_start=.1,exploration_end=.05,
        opponent_config={'max_actions':120,'trace_compresslevel':1})
    model.trace.close()
    assert model.finished_games==4 and model.version>0 and model.model.eta0==.005
    assert summary['forbidden_engine_calls']==0 and np.isfinite(model.model.coef_).all()
    with gzip.open(tmp_path/'updates.jsonl.gz','rt',encoding='utf-8') as stream:
        sample=json.loads(next(stream))['samples'][0]
    game=sample['game_index']
    with gzip.open(tmp_path/'live_training'/f'game_{game:04d}'/'decisions.jsonl.gz','rt',encoding='utf-8') as stream:
        rows={r['action_index']:r for r in map(json.loads,stream)}
    def phi(index):
        obs=rows[index]['inputs']['observation'];actor=obs['actor']
        own=next(p for p in obs['players'] if p['color']==actor)
        leader=max(p['VICTORY_POINTS'] for p in obs['players'] if p['color']!=actor)
        return .05*own['actual_vp']+.02*(own['actual_vp']-leader)
    assert not sample['terminal']
    expected=sample['discount']*phi(sample['next_action_index'])-phi(sample['action_index'])
    assert abs(sample['reward']-expected)<1e-12
