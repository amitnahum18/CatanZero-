from copy import deepcopy
from math import isclose

import pytest
from catanatron.serialization import action_to_json, map_to_json

from catanzero.dataset import observation, ordered_actions
from catanzero.features import FeatureEncoder, feature_schema
from catanzero.regression_data import disjoint_feature_splits, fit_normalizer, normalize, privacy_invariance_check
from catanzero.simulation import SimulationConfig, create_game


def fixture_inputs():
    game = create_game(SimulationConfig(), 84)
    game.execute(ordered_actions(game)[0])
    actions = ordered_actions(game)
    return game, {"map_ref": 42, "observation": observation(game, game.state.current_color()),
                  "action": action_to_json(actions[0]),
                  "legal_actions": [action_to_json(a) for a in actions]}


def test_fixed_feature_schema_and_future_private_invariance():
    game, inputs = fixture_inputs()
    encoder = FeatureEncoder(map_to_json(game.state.board.map))
    schema = feature_schema()
    vector = encoder.encode(inputs)
    assert schema["input_columns"] == len(vector) == 164
    assert schema["output_columns"] == 1
    assert sum(g["count"] for g in schema["groups"].values()) == 164
    assert len(set(schema["feature_names"])) == 164
    assert privacy_invariance_check(encoder, {"inputs": inputs, "targets": {}, "heuristic": {}})
    assert not any(name in schema["feature_names"] for name in
                   ("weighted_score", "terminal_outcome", "heuristic_score", "actor_bot", "seed", "game_index"))


def test_undirected_roads_not_double_counted():
    game, inputs = fixture_inputs()
    encoder = FeatureEncoder(map_to_json(game.state.board.map))
    edge = inputs["action"][2]
    modified = deepcopy(inputs)
    color = inputs["observation"]["actor"]
    modified["observation"]["roads"] = [[edge, color]]
    duplicated = deepcopy(modified)
    duplicated["observation"]["roads"].append([list(reversed(edge)), color])
    assert encoder.encode(modified) == encoder.encode(duplicated)
    features = dict(zip(feature_schema()["feature_names"], encoder.encode(modified)))
    assert features["location_own_roads_touching"] == 1


def test_location_production_uses_real_map_neighbors():
    game = create_game(SimulationConfig(), 42)
    encoder = FeatureEncoder(map_to_json(game.state.board.map))
    actions = ordered_actions(game)
    for action in actions[:5]:
        inputs = {"map_ref": 0, "observation": observation(game, game.state.current_color()),
                  "action": action_to_json(action), "legal_actions": [action_to_json(a) for a in actions]}
        values = dict(zip(feature_schema()["feature_names"], encoder.encode(inputs)))
        expected = game.state.board.map.node_production[action.value]
        for resource in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"):
            assert isclose(values[f"location_potential_{resource}"], expected[resource])


def test_sampled_future_roll_or_card_rejected():
    game, inputs = fixture_inputs()
    encoder = FeatureEncoder(map_to_json(game.state.board.map))
    changed = deepcopy(inputs)
    changed["action"] = [inputs["observation"]["actor"], "ROLL", [6, 6]]
    changed["legal_actions"] = [changed["action"]]
    with pytest.raises(ValueError, match="sampled outcome"):
        encoder.encode(changed)
    changed["action"] = [inputs["observation"]["actor"], "BUY_DEVELOPMENT_CARD", "VICTORY_POINT"]
    with pytest.raises(ValueError, match="sampled outcome"):
        encoder.encode(changed)


def test_train_only_normalization_and_binary_columns():
    schema = {"input_columns": 3, "feature_names": ["resource", "flag", "constant"],
              "binary_features": ["flag"], "version": "test"}
    scaler = fit_normalizer([[2, 0, 7], [4, 1, 7]], schema)
    assert scaler["offset"] == [3, 0, 7]
    assert scaler["scale"] == [1, 1, 1]
    assert normalize([2, 0, 7], scaler) == [-1, 0, 0]
    before = deepcopy(scaler)
    assert normalize([1000, 1, 9], scaler) == [997, 1, 2]
    assert scaler == before  # Applying validation/test does not fit again.


def test_actor_color_names_are_not_features():
    game, inputs = fixture_inputs()
    encoder = FeatureEncoder(map_to_json(game.state.board.map))
    colors = [p["color"] for p in inputs["observation"]["players"]]
    mapping = dict(zip(colors, reversed(colors)))
    changed = deepcopy(inputs)
    obs = changed["observation"]
    for key in ("actor", "turn_owner"):
        obs[key] = mapping[obs[key]]
    for player in obs["players"]:
        player["color"] = mapping[player["color"]]
    for building in obs["buildings"]:
        building[1] = mapping[building[1]]
    for road in obs["roads"]:
        road[1] = mapping[road[1]]
    for action in [changed["action"], *changed["legal_actions"]]:
        # Deepcopy preserves references between the chosen action and legal
        # actions only if the fixture reuses lists; this fixture serializes separately.
        action[0] = mapping[action[0]]
    assert encoder.encode(changed) == encoder.encode(inputs)


def test_feature_collision_filter_uses_inputs_not_labels_and_keeps_training():
    rows = [{"split": split, "x": x, "y": y, "metadata": {"split": split, "index": i}}
            for i, (split, x, y) in enumerate([
                ("test", [1, 2], 0.9), ("train", [1, 2], 0.1),
                ("validation", [1, 2], 0.8), ("validation", [3, 4], 0.5),
                ("test", [3, 4], 0.4), ("test", [5, 6], 0.7)])]
    kept, excluded = disjoint_feature_splits(rows)
    assert len(kept) == len(excluded) == 3
    assert [r["metadata"]["index"] for r in kept] == [1, 3, 5]
    changed = deepcopy(rows)
    for row in changed:
        row["y"] = 1 - row["y"]
    assert [r["metadata"] for r in disjoint_feature_splits(changed)[0]] == [r["metadata"] for r in kept]
