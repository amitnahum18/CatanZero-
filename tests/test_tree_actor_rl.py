"""Numerical checks for the actual clipped actor objective."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from optimize_tree_actor_rl import grouped_softmax,ClippedActorObjective,SamplingPolicy


def fixture():
    starts=np.asarray([0,3]);lengths=np.asarray([3,3]);pred=np.asarray([-.2,.1,.4,0.,.2,.1])
    data={'starts':starts,'lengths':lengths,'selected_rows':np.asarray([0,4]),
          'old_probabilities':grouped_softmax(pred,starts,lengths),'fixed_critic':np.zeros(6),
          'advantage':np.asarray([1.,-.6]),'weight':np.asarray([1.,1.])}
    return pred,data


def test_policy_gradient_matches_finite_difference_for_positive_and_negative_advantages():
    pred,data=fixture();objective=ClippedActorObjective(data,1.,1.)
    grad,hess=objective(pred,None);numerical=np.empty(len(pred));eps=1e-6
    for i in range(len(pred)):
        plus=pred.copy();minus=pred.copy();plus[i]+=eps;minus[i]-=eps
        numerical[i]=(objective.loss(plus)-objective.loss(minus))/(2*eps)
    np.testing.assert_allclose(grad,numerical,rtol=0,atol=1e-8)
    assert grad[0]<0 and grad[4]>0 and (hess>0).all()
    np.testing.assert_allclose(np.add.reduceat(grad,data['starts']),0,atol=1e-12)


def test_clipping_stops_outward_policy_updates_and_keeps_KL_restore_force():
    pred,data=fixture();objective=ClippedActorObjective(data,1.,1.)
    pred[0]+=5;pred[4]-=5
    grad,_=objective(pred,None);p=objective.probabilities(pred)
    np.testing.assert_allclose(grad,.1*(p-data['old_probabilities']),atol=1e-12)


def test_sampling_records_the_actual_probability_without_future_game_inputs():
    import random
    class Reference:
        teacher=kind=normalizer=mc=None
        scale=1.;version=700;mix=.2
    policy=SamplingPolicy(Reference(),.25)
    policy.predict=lambda raw:np.asarray([0.,.2,.4])
    choice=policy.choose({'features':np.zeros((3,264))},random.Random(3),0)
    expected=np.exp(np.asarray([0.,.2,.4])/.25);expected/=expected.sum()
    assert choice['probability']==expected[choice['index']]
