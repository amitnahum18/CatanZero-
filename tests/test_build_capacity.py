from copy import deepcopy
from catanatron.serialization import map_to_json, action_to_json
from catanzero.simulation import SimulationConfig, create_game
from catanzero.dataset import observation, ordered_actions
from catanzero.features import FeatureEncoder, feature_schema, BUILD_VERSION, STRATEGIC_VERSION
from catanzero.regression_data import privacy_invariance_check
from catanzero.static_policies import forbid_forward_simulation


def fixture():
    game = create_game(SimulationConfig(), 84)
    encoder = FeatureEncoder(map_to_json(game.state.board.map), version=BUILD_VERSION)
    actor = game.state.current_color().value
    path = [min(encoder.strategic.neighbors)]
    for _ in range(2):
        path.append(next(n for n in sorted(encoder.strategic.neighbors[path[-1]]) if n not in path))
    obs = observation(game, game.state.current_color())
    obs["initial_phase"] = False
    obs["buildings"] = [[path[0], actor, "SETTLEMENT"]]
    own = next(p for p in obs["players"] if p["color"] == actor)
    own["HAS_ROLLED"] = True
    own["resources"].update(WOOD=2, BRICK=2, SHEEP=1, WHEAT=1)
    action = [actor, "BUILD_ROAD", path[:2]]
    return encoder, {"observation": obs, "action": action, "legal_actions": [deepcopy(action)], "map_ref": 0}, own


def values(encoder, inputs):
    return dict(zip(feature_schema(BUILD_VERSION)["feature_names"], encoder.encode(inputs)))


def test_two_roads_budget_geometry_and_shared_settlement_budget():
    encoder, inputs, own = fixture()
    with forbid_forward_simulation():
        v = values(encoder, inputs)
        assert privacy_invariance_check(encoder, {"inputs": inputs})
    assert v["own_road_resource_capacity"] == 2
    assert v["own_can_fund_two_roads"] == 1 and v["own_two_road_sequence_feasible"] == 1
    assert v["own_can_fund_road_and_settlement"] == 1
    assert v["own_can_fund_two_roads_and_settlement"] == 0
    assert v["after_action_road_resource_capacity"] == 1
    own["ROADS_AVAILABLE"] = 1
    assert values(encoder, inputs)["own_two_road_sequence_feasible"] == 0
    own["ROADS_AVAILABLE"] = 0
    assert values(encoder, inputs)["own_road_piece_limited_capacity"] == 0


def test_affordability_is_separate_from_current_legality_and_city_piece_return():
    encoder, inputs, own = fixture()
    own["resources"].update(ORE=3, WHEAT=2)
    own["SETTLEMENTS_AVAILABLE"] = 0
    node = inputs["observation"]["buildings"][0][0]
    inputs["action"] = [inputs["observation"]["actor"], "BUILD_CITY", node]
    inputs["legal_actions"] = [inputs["action"]]
    v = values(encoder, inputs)
    assert v["own_road_resource_capacity"] == 2 and v["own_road_can_build_now"] == 0
    assert v["own_city_piece_limited_capacity"] == 1
    assert v["after_action_settlement_piece_limited_capacity"] == 0  # wheat spent
    own["resources"]["WHEAT"] = 3
    assert values(encoder, inputs)["after_action_settlement_piece_limited_capacity"] == 1


def test_free_road_credits_and_initial_phase_do_not_claim_two_consecutive_roads():
    encoder, inputs, own = fixture()
    own["resources"].update(WOOD=0, BRICK=0)
    own["HAS_ROLLED"] = False
    inputs["observation"]["free_roads_available"] = 2
    v = values(encoder, inputs)
    assert v["own_funded_road_capacity"] == 2 and v["own_two_road_sequence_feasible"] == 1
    assert v["after_action_free_road_capacity"] == 1
    inputs["observation"]["initial_phase"] = True
    assert values(encoder, inputs)["own_two_road_sequence_feasible"] == 0


def test_v3_schema_preserves_all_previous_columns():
    schema = feature_schema(BUILD_VERSION)
    assert schema["input_columns"] == 264
    assert schema["feature_names"][:228] == feature_schema(STRATEGIC_VERSION)["feature_names"]
    assert len(set(schema["feature_names"])) == 264
