import random
import numpy as np
from catanatron.serialization import state_to_json
from catanatron.state_functions import player_key
from catanatron.models.enums import RESOURCES
from catanzero.counterfactual_402 import evaluate_pair, pair_statistics, terminal_rollout
from catanzero.dataset import observation, ordered_actions
from catanzero.features import BUILD_VERSION
from catanzero.simulation import create_game, SimulationConfig


class FakePolicy:
    normalizer={'feature_version':BUILD_VERSION}
    def predict(self,raw):
        return np.zeros(len(raw))


def test_paired_rollouts_preserve_source_and_ignore_actual_future_rng():
    game=create_game(SimulationConfig(players=2),42)
    actions=ordered_actions(game)[:2]
    before=state_to_json(game)
    a=evaluate_pair(game,actions,[10,11],FakePolicy(),max_actions=4)
    assert state_to_json(game)==before
    game.random=game.state.random=random.Random(777)
    b=evaluate_pair(game,actions,[10,11],FakePolicy(),max_actions=4)
    assert a==b
    assert a['censored_pairs']==2 and not a['training_eligible']
    assert a['learned_decisions']>0 and a['F_decisions']>0


def test_terminal_winner_is_detected_before_action_cap():
    game=create_game(SimulationConfig(players=2),42)
    action=ordered_actions(game)[0]
    key=player_key(game.state,action.color)
    game.state.player_state[f'{key}_ACTUAL_VICTORY_POINTS']=9
    game.state.player_state[f'{key}_VICTORY_POINTS']=9
    result=terminal_rollout(game,action,20,FakePolicy(),max_actions=1)
    assert result['outcome']==1 and result['steps']==1
    assert game.winning_color() is None


def test_hidden_allocation_and_deck_order_do_not_change_rollout_policy_inputs():
    game=create_game(SimulationConfig(),81)
    actor=game.state.current_color()
    enemies=[c for c in game.state.colors if c!=actor]
    first,second=[player_key(game.state,c) for c in enemies[:2]]
    game.state.player_state[f'{first}_WOOD_IN_HAND']=2
    game.state.player_state[f'{second}_ORE_IN_HAND']=2
    game.state.resource_freqdeck[RESOURCES.index('WOOD')]-=2
    game.state.resource_freqdeck[RESOURCES.index('ORE')]-=2
    changed=game.copy()
    changed.state.player_state[f'{first}_WOOD_IN_HAND']=0
    changed.state.player_state[f'{first}_ORE_IN_HAND']=2
    changed.state.player_state[f'{second}_ORE_IN_HAND']=0
    changed.state.player_state[f'{second}_WOOD_IN_HAND']=2
    changed.state.development_listdeck.reverse()
    changed.random=changed.state.random=random.Random(999)
    assert observation(game,actor)==observation(changed,actor)

    class RecordingPolicy(FakePolicy):
        def __init__(self):self.calls=[]
        def predict(self,raw):
            self.calls.append(raw.copy())
            return super().predict(raw)
    a,b=RecordingPolicy(),RecordingPolicy()
    action=ordered_actions(game)[0]
    left=terminal_rollout(game,action,123,a,max_actions=4)
    right=terminal_rollout(changed,action,123,b,max_actions=4)
    assert left==right and len(a.calls)==len(b.calls)>0
    for x,y in zip(a.calls,b.calls):np.testing.assert_array_equal(x,y)


def test_preference_uses_paired_terminal_rewards_and_rejects_censoring():
    stats=pair_statistics([[0]*8,[1]*8])
    assert stats['reward_advantage_alternative']==2
    assert stats['statistically_resolved_05'] and stats['training_eligible']
    stats=pair_statistics([[0,None],[1,0]])
    assert stats['reward_advantage_alternative'] is None and not stats['training_eligible']
