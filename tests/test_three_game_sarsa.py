import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from train_three_game_sarsa import ThreeGameSARSA
from train_online_sarsa import play_phase


def test_updates_only_after_three_distinct_finished_games_and_checkpoint(tmp_path):
    norm={'feature_names':['a','b'],'fit_split':'train','offset':[0.,0.],'scale':[1.,1.]}
    model=ThreeGameSARSA(tmp_path,norm)
    for index in range(6):
        model.add(np.ones(2),np.zeros(2),1.,0.,{'game_index':index})
        model.flush()
        assert model.version==index//3
        model.finish_game(index,censored=index==1)
        assert model.version==(index+1)//3
    assert model.version==2 and model.transitions==6 and not model.queue
    assert [b['completed_games'] for b in model.batches]==[3,6]
    assert (tmp_path/'checkpoints/version_000002.joblib').exists()
    with pytest.raises(ValueError,match='twice'):
        model.finish_game(5)
    model.trace.close()


def test_parallel_real_games_create_exact_episode_cadence_without_extra_tail_update(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    source=repo/'runs/online-sarsa-features-100-v4'
    norm=json.loads((source/'normalizer.json').read_text())
    model=ThreeGameSARSA(tmp_path,norm,resume_from=source)
    results,summary=play_phase(tmp_path,'live_training',model,6,3,training=True,base_seed=28000000,
        excluded=[],exploration_start=.1,exploration_end=.05,opponent_config={'max_actions':150,'trace_compresslevel':1})
    assert len(results)==6 and model.version==2 and model.completed_games==6
    assert model.transitions>0 and not model.queue and summary['forbidden_engine_calls']==0
    assert len({g['game_index'] for b in model.batches for g in b['newly_finished_games']})==6
    model.trace.close()
