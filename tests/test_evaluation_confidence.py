import pytest
from evaluation_confidence import win_rate_acceptance


def test_acceptance_requires_lower_bound_above_target_not_observed_rate():
    assert not win_rate_acceptance(539,1000)['passed']
    assert not win_rate_acceptance(570,1000)['passed']
    assert not win_rate_acceptance(581,1000)['passed']
    assert win_rate_acceptance(582,1000)['passed']


def test_small_sample_and_boundaries():
    assert not win_rate_acceptance(124,200)['passed']
    assert win_rate_acceptance(125,200)['passed']
    assert win_rate_acceptance(0,10)['interval'][0]==0
    assert win_rate_acceptance(10,10)['interval'][1]==1
    with pytest.raises(ValueError):win_rate_acceptance(0,0)
