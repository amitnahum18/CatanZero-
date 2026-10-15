from copy import deepcopy
from catanzero.temporal_features import TemporalFeatures


def observation(actor='RED', vp=1, wood=1):
    return {'actor':actor,'players':[
        {'color':'RED','VICTORY_POINTS':vp,'actual_vp':vp,'LONGEST_ROAD_LENGTH':0,
         'resources':{'WOOD':wood,'BRICK':1,'WHEAT':1,'SHEEP':1,'ORE':0}},
        {'color':'BLUE','VICTORY_POINTS':2,'actual_vp':2,'LONGEST_ROAD_LENGTH':0,
         'resources':{'WOOD':2,'BRICK':1,'WHEAT':1,'SHEEP':1,'ORE':0}}]}


def test_lags_are_strictly_past_and_candidate_queries_do_not_commit():
    memory=TemporalFeatures()
    for i in range(5):
        obs=observation(vp=i,wood=i)
        features=memory.features(obs,'game-a')
        assert features==memory.features(obs,'game-a')
        assert features['history_5_available']==0
        memory.commit(obs,'game-a','BUILD_ROAD')
    features=memory.features(observation(vp=5,wood=5),'game-a')
    for lag in (1,3,5):
        assert features[f'history_{lag}_delta_own_vp']==lag
        assert features[f'history_{lag}_delta_resources']==lag
    assert features['history_5_share_BUILD_ROAD']==1
    assert len(features)==30


def test_history_resets_per_game_and_stays_per_actor():
    memory=TemporalFeatures()
    memory.commit(observation(),'game-a','END_TURN')
    assert memory.features(observation(),'game-a')['history_1_available']==1
    assert memory.features(observation(actor='BLUE'),'game-a')['history_1_available']==0
    assert memory.features(observation(),'game-b')['history_1_available']==0


def test_future_fields_and_private_opponent_values_never_enter_history():
    original=observation()
    changed=deepcopy(original)
    changed.update(winner='RED',seed=999,future_roll=[6,6])
    changed['players'][1].update(resources={'ORE':999},actual_vp=999,development_cards={'VICTORY_POINT':999})
    a,b=TemporalFeatures(),TemporalFeatures()
    a.commit(original,'g','END_TURN')
    b.commit(changed,'g','END_TURN')
    assert a.features(original,'g')==b.features(changed,'g')
