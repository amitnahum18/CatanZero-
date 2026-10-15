from copy import deepcopy
import pytest
from catanatron.serialization import map_to_json, action_to_json
from catanzero.dataset import observation, ordered_actions
from catanzero.decision_interactions import CandidateDecisionFeatures, interactions
from catanzero.simulation import create_game, SimulationConfig
from catanzero.static_policies import forbid_forward_simulation


def base():
    return {'action_BUY_DEVELOPMENT_CARD':0.,'action_BUILD_ROAD':0.,'own_points_to_win':8.,
        'own_public_vp_gap':-2.,'after_action_settlement_piece_limited_capacity':1.,'own_city_can_build_now':1.,
        'own_city_resource_deficit':0.,'own_army_award_gap':3.,'action_road_length_gain':1.,
        'own_road_award_gap':4.,'action_new_sites_at_network':2.,'action_best_new_site_yield':.4,
        'action_estimated_vp_gain':1.,'production_delta_ORE':.2,'own_resource_ORE':0.,
        'production_delta_WHEAT':.1,'own_resource_WHEAT':2.,'action_END_TURN':0.,
        'after_action_expected_discard_next_roll':.5,'action_MARITIME_TRADE':0.,'after_action_city_resource_deficit':1.,
        'action_PLAY_KNIGHT_CARD':0.,'own_playable_KNIGHT':1.,'own_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN':0.,
        'own_development_count':4.,'robber_enemy_production_blocked':.5,'victim_public_vp':9.}


def test_contextual_products_activate_only_for_relevant_action_and_budget():
    values=base();values['action_BUY_DEVELOPMENT_CARD']=1.
    out=interactions(values)
    assert out['interaction_buy_dev_city_ready']==1
    assert out['interaction_buy_dev_held_card_count']==4
    assert out['interaction_road_funded_new_sites']==0
    assert out['interaction_production_delta_wheat_city_scarcity']==0
    assert out['interaction_vp_gain_race_urgency']==pytest.approx(1/3)
    values['action_BUILD_ROAD']=1.
    assert interactions(values)['interaction_road_funded_new_sites']==2
    values['after_action_settlement_piece_limited_capacity']=0.
    assert interactions(values)['interaction_road_funded_new_sites']==0


def test_real_candidate_features_ignore_future_and_private_opponent_information():
    game=create_game(SimulationConfig(),84)
    inputs={'observation':observation(game,game.state.current_color()),
        'legal_actions':[action_to_json(a) for a in ordered_actions(game)],'action':action_to_json(ordered_actions(game)[0])}
    prototype=CandidateDecisionFeatures(map_to_json(game.state.board.map))
    changed=deepcopy(inputs)
    changed.update(target=1.,seed=999,winner='future')
    changed['observation'].update(winner='future',random_state='future',development_listdeck=['VICTORY_POINT']*100)
    for player in changed['observation']['players']:
        if player['color']!=changed['observation']['actor']:
            player.update(actual_vp=999,resources={'ORE':999},development_cards={'KNIGHT':999})
    before=game.state.random.getstate()
    with forbid_forward_simulation():
        a=prototype.features(inputs)
        assert a==prototype.features(changed)
    assert len(a)==24 and before==game.state.random.getstate()
    wrong=deepcopy(inputs);wrong['action'][0]='INVALID'
    with pytest.raises(ValueError):
        prototype.features(wrong)
