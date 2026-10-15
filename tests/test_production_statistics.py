import pytest
from catanatron.models.map import number_probability
from catanatron.serialization import map_to_json, action_to_json
from catanzero.simulation import create_game, SimulationConfig
from catanzero.dataset import observation, ordered_actions
from catanzero.static_policies import forbid_forward_simulation
from catanzero.production_statistics import income_moments, ProductionStatistics


def test_equal_mean_different_dice_concentration_has_different_variance_and_zero_risk():
    concentrated = {n:{} for n in range(2,13)}
    spread = {n:{} for n in range(2,13)}
    concentrated[6] = {'WOOD':2}
    spread[6] = spread[8] = {'WOOD':1}
    a,b = income_moments(concentrated),income_moments(spread)
    assert a['mean_income']==pytest.approx(b['mean_income'])
    assert a['income_variance']>b['income_variance']
    assert a['probability_no_income']>b['probability_no_income']


def test_resource_covariance_detects_same_number_and_opposite_number_production():
    same={n:{} for n in range(2,13)}
    separate={n:{} for n in range(2,13)}
    same[6]={'WOOD':1,'BRICK':1}
    separate[6]={'WOOD':1}
    separate[8]={'BRICK':1}
    p=number_probability(6)
    assert income_moments(same)['covariance_WOOD_BRICK']==pytest.approx(p-p*p)
    assert income_moments(separate)['covariance_WOOD_BRICK']==pytest.approx(-p*p)


def test_prototype_uses_only_current_public_geometry_without_engine_calls():
    game=create_game(SimulationConfig(),84)
    action=ordered_actions(game)[0]
    obs=observation(game,game.state.current_color())
    inputs={'observation':obs,'action':action_to_json(action)}
    prototype=ProductionStatistics(map_to_json(game.state.board.map))
    with forbid_forward_simulation():
        result=prototype.features(inputs)
    assert len(result)==8
    assert result['stat_current_probability_no_income']==pytest.approx(1.)
    assert result['stat_delta_probability_no_income']<0
