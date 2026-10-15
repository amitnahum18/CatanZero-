import pytest
from catanzero.online_linear import shaped_transition


def state(index,vp,leader):
    return {'action_index':index,'vp':vp,'leader_public_vp':leader,'learner_color':'RED'}


def test_leader_gain_reduces_reward_and_own_gain_improves_it():
    original=state(0,4,5)
    ahead=state(1,5,5)
    behind=state(1,4,6)
    a,_=shaped_transition(original,ahead,terminal=False,won=None,gap_weight=.02,gamma=1)
    b,_=shaped_transition(original,behind,terminal=False,won=None,gap_weight=.02,gamma=1)
    assert a==pytest.approx(.07)
    assert b==pytest.approx(-.02)
    same,_=shaped_transition(original,state(1,4,5),terminal=False,won=None,gap_weight=.02,gamma=1)
    assert same==pytest.approx(0)


@pytest.mark.parametrize('won',[True,False])
def test_relative_potential_telescopes_and_terminal_potential_is_zero(won):
    gamma=.9
    states=[state(0,2,3),state(2,4,5),state(5,6,5),state(8,10,8)]
    returns=0.;discount=1.
    for index in range(3):
        r,d=shaped_transition(states[index],states[index+1],terminal=index==2,won=won if index==2 else None,
            gamma=gamma,gap_weight=.02)
        returns+=discount*r
        discount*=d
    initial_phi=.05*2+.02*(2-3)
    assert returns==pytest.approx(gamma**7*int(won)-initial_phi)


def test_gap_reward_requires_observed_public_points_and_default_stays_compatible():
    a=state(0,2,3);b=state(2,3,4)
    a.pop('leader_public_vp')
    with pytest.raises(ValueError,match='public opponent'):
        shaped_transition(a,b,terminal=False,won=None,gap_weight=.02)
    r,d=shaped_transition(a,b,terminal=False,won=None)
    assert d==pytest.approx(.999**2) and r==pytest.approx(.999**2*.05*3-.05*2)
