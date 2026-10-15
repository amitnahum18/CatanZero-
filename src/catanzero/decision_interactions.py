"""Undeployed candidate interactions motivated by the every-three-game analysis.

Consumes only the existing public/predecision feature encoder. No normalization
is fitted here, no learned model is changed, and no future gameplay is sampled.
"""
from math import isfinite
from catanzero.features import FeatureEncoder, feature_schema, BUILD_VERSION
from catanzero.production_statistics import ProductionStatistics


def interactions(base):
    buy=base['action_BUY_DEVELOPMENT_CARD']
    road=base['action_BUILD_ROAD']
    urgency=1/(1+base['own_points_to_win'])
    behind=max(-base['own_public_vp_gap'],0.)
    settle_budget=min(base['after_action_settlement_piece_limited_capacity'],1.)
    return {
        'interaction_buy_dev_city_ready':buy*base['own_city_can_build_now'],
        'interaction_buy_dev_city_gap':buy*base['own_city_resource_deficit']/5,
        'interaction_buy_dev_army_proximity':buy/(1+base['own_army_award_gap']),
        'interaction_buy_dev_victory_proximity':buy*urgency,
        'interaction_road_award_proximity':road*base['action_road_length_gain']/(1+base['own_road_award_gap']),
        'interaction_road_funded_new_sites':road*base['action_new_sites_at_network']*settle_budget,
        'interaction_road_funded_new_site_yield':road*base['action_best_new_site_yield']*settle_budget,
        'interaction_vp_gain_race_urgency':base['action_estimated_vp_gain']*(1+behind)*urgency,
        'interaction_production_delta_ore_city_scarcity':base['production_delta_ORE']*max(3-base['own_resource_ORE'],0)/3,
        'interaction_production_delta_wheat_city_scarcity':base['production_delta_WHEAT']*max(2-base['own_resource_WHEAT'],0)/2,
        'interaction_end_turn_discard_risk':base['action_END_TURN']*base['after_action_expected_discard_next_roll'],
        'interaction_trade_city_progress_urgency':base['action_MARITIME_TRADE']*(base['own_city_resource_deficit']-base['after_action_city_resource_deficit'])*urgency,
        'interaction_play_knight_army_proximity':base['action_PLAY_KNIGHT_CARD']/(1+base['own_army_award_gap']),
        'interaction_skip_playable_knight_army_proximity':base['action_END_TURN']*base['own_playable_KNIGHT']*(1-base['own_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN'])/(1+base['own_army_award_gap']),
        'interaction_buy_dev_held_card_count':buy*base['own_development_count'],
        'interaction_robber_block_leader':base['robber_enemy_production_blocked']*base['victim_public_vp'],
    }


class CandidateDecisionFeatures:
    def __init__(self,map_document):
        self.encoder=FeatureEncoder(map_document,version=BUILD_VERSION)
        self.names=feature_schema(BUILD_VERSION)['feature_names']
        self.production=ProductionStatistics(map_document)

    def features(self,inputs):
        # The established encoder validates actor, legal intent and absence of
        # sampled outcomes before experimental features consume the input.
        base=dict(zip(self.names,self.encoder.encode(inputs),strict=True))
        result={**interactions(base),**self.production.features(inputs)}
        if len(result)!=24 or not all(isfinite(v) for v in result.values()):
            raise ValueError('Invalid candidate feature values')
        return result
