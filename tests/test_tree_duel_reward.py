import pytest
from catanzero.tree_duel_reward import transition_reward,potential


def state(index,vp,enemy):
    return {"action_index":index,"learner_color":"RED","vp":vp,"leader_public_vp":enemy}


@pytest.mark.parametrize("won",[True,False])
def test_total_shaped_return_matches_symmetric_outcome_regardless_of_game_length(won):
    states=[state(0,2,2),state(3,3,2),state(50,4,6),state(110,7,9),state(1000,10,9)]
    value=0.
    for index,(a,b) in enumerate(zip(states,states[1:])):
        r,d=transition_reward(a,b,terminal=index==3,won=won if index==3 else None)
        assert d == (0. if index==3 else 1.)
        value+=r
    assert value == pytest.approx((1. if won else -1.)-potential(2,2))


def test_own_points_help_and_enemy_points_hurt_without_repeated_full_gap_penalties():
    a=state(0,4,5)
    assert transition_reward(a,state(1,5,5),terminal=False)[0] == pytest.approx(.07)
    assert transition_reward(a,state(1,4,6),terminal=False)[0] == pytest.approx(-.02)
    assert transition_reward(a,state(1,4,5),terminal=False)[0] == pytest.approx(0.)


def test_censored_game_and_cross_actor_are_rejected():
    with pytest.raises(ValueError,match="censored"):
        transition_reward(state(0,2,2),state(1,2,2),terminal=True)
    b={**state(1,2,2),"learner_color":"BLUE"}
    with pytest.raises(ValueError,match="actors"):
        transition_reward(state(0,2,2),b,terminal=False)
