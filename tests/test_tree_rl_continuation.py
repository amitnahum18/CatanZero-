"""Reject invalid paired claims; preserve the reward's victory objective."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from continue_champion_tree_rl import paired_stats


def result(index,won):
    return {'game_index':index,'seed':100+index,'learner_seat':index%2,
            'initial_map_hash':str(index),'initial_stream_hash':str(index),
            'termination':'victory','learner_won':won}


def test_paired_test_counts_only_actual_same_initial_game_results():
    baseline=[result(0,False),result(1,True),result(2,True),result(3,False)]
    continued=[result(0,True),result(1,False),result(2,True),result(3,False)]
    stats=paired_stats(baseline,continued)
    assert stats['candidate_only_wins']==stats['baseline_only_wins']==1
    assert stats['two_sided_p']==1
    changed=[{**r} for r in continued];changed[0]['initial_stream_hash']='other'
    with pytest.raises(AssertionError):paired_stats(baseline,changed)
    changed=[{**r} for r in continued];changed[0]['termination']='time_limit'
    changed[0]['learner_won']=None
    with pytest.raises(ValueError,match='unknown tails'):paired_stats(baseline,changed)
