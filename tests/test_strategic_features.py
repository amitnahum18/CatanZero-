from copy import deepcopy
import json

import pytest
from catanatron.serialization import map_to_json, action_to_json
from catanzero.dataset import observation, ordered_actions
from catanzero.features import FeatureEncoder, feature_schema, VERSION, STRATEGIC_VERSION
from catanzero.regression_data import privacy_invariance_check, fit_normalizer
from catanzero.simulation import SimulationConfig, create_game
from catanzero.static_policies import forbid_forward_simulation
from catanzero.strategic_features import award_holder, longest_road


def fixture():
    game = create_game(SimulationConfig(), 84)
    actions = ordered_actions(game)
    inputs = {"observation": observation(game, game.state.current_color()), "map_ref": 0,
              "legal_actions": [action_to_json(a) for a in actions], "action": action_to_json(actions[0])}
    return FeatureEncoder(map_to_json(game.state.board.map), version=STRATEGIC_VERSION), inputs


def values(encoder, inputs):
    return dict(zip(feature_schema(STRATEGIC_VERSION)["feature_names"], encoder.encode(inputs)))


def test_new_schema_keeps_original_prefix_and_has_no_forward_or_private_dependency():
    game = create_game(SimulationConfig(), 84)
    encoder, inputs = fixture()
    old = FeatureEncoder(map_to_json(game.state.board.map))
    legacy = old.encode(inputs)
    with forbid_forward_simulation():
        expanded = encoder.encode(inputs)
        assert expanded[:164] == legacy
        assert privacy_invariance_check(encoder, {"inputs": inputs})
    schema = feature_schema(STRATEGIC_VERSION)
    assert schema["input_columns"] == len(expanded) == 228
    assert schema["feature_names"][:164] == feature_schema()["feature_names"]
    assert sum(g["count"] for g in schema["groups"].values()) == 228
    assert len(set(schema["feature_names"])) == 228
    assert "action_estimated_vp_gain" not in schema["binary_features"]
    assert "after_action_city_can_afford" in schema["binary_features"]
    with pytest.raises(ValueError, match="unsupported"):
        feature_schema("typo")


def test_resource_discard_threshold_ignores_development_cards_and_handles_spending():
    encoder, inputs = fixture()
    actor = inputs["observation"]["actor"]
    own = next(p for p in inputs["observation"]["players"] if p["color"] == actor)
    own["resources"].update({r: 0 for r in own["resources"]})
    own["resources"]["WOOD"] = 8
    own["resource_count"] = 8
    own["development_cards"]["KNIGHT"] = 3
    own["development_count"] = 3
    inputs["action"] = [actor, "DISCARD_RESOURCE", "WOOD"]
    inputs["legal_actions"] = [inputs["action"]]
    v = values(encoder, inputs)
    assert v["own_discard_if_seven"] == 4
    assert v["own_expected_discard_next_roll"] == pytest.approx(4/6)
    assert v["after_action_discard_if_seven"] == 0
    assert v["after_action_above_discard_limit"] == 0
    inputs["observation"]["discard_limit"] = 9
    assert values(encoder, inputs)["own_discard_if_seven"] == 0


def test_exact_longest_trails_branches_cycles_and_enemy_cut():
    chain = tuple((i, i+1) for i in range(5))
    assert longest_road(chain, ()) == 5
    assert longest_road(chain, (2,)) == 3
    assert longest_road(((0, 1), (1, 2), (1, 3)), ()) == 2
    assert longest_road(((0, 1), (0, 2), (1, 2), (2, 3)), ()) == 4
    assert award_holder({"a": 5, "b": 5}, "a", 5) == "a"
    assert award_holder({"a": 5, "b": 5}, None, 5) is None
    assert award_holder({"a": 4, "b": 5}, "a", 5) == "b"
    assert award_holder({"a": 4, "b": 4}, "a", 5) is None


def path_on_map(encoder, length):
    def extend(path):
        if len(path) == length+1:
            return path
        for other in sorted(encoder.strategic.neighbors[path[-1]]):
            if other not in path:
                result = extend(path+[other])
                if result:
                    return result
    return extend([min(encoder.strategic.neighbors)])


def test_fifth_road_award_and_known_winning_points_and_tie_retention():
    encoder, inputs = fixture()
    obs, actor = inputs["observation"], inputs["observation"]["actor"]
    own = next(p for p in obs["players"] if p["color"] == actor)
    path = path_on_map(encoder, 5)
    obs["roads"] = [[[a, b], actor] for a, b in zip(path[:4], path[1:5])]
    own["LONGEST_ROAD_LENGTH"] = 4
    own["actual_vp"] = 8
    own["resources"]["WOOD"] = own["resources"]["BRICK"] = 1
    obs["initial_phase"] = False
    inputs["action"] = [actor, "BUILD_ROAD", path[4:6]]
    inputs["legal_actions"] = [inputs["action"]]
    v = values(encoder, inputs)
    assert v["action_road_length_gain"] == 1
    assert v["action_road_award_gain"] == 1
    assert v["action_bonus_vp_gain"] == v["action_estimated_vp_gain"] == 2
    assert v["action_estimated_win"] == 1
    enemy = next(p for p in obs["players"] if p["color"] != actor)
    enemy["LONGEST_ROAD_LENGTH"], enemy["HAS_ROAD"] = 5, True
    assert values(encoder, inputs)["action_road_award_gain"] == 0


def test_city_points_and_resource_affordability():
    encoder, inputs = fixture()
    obs, actor = inputs["observation"], inputs["observation"]["actor"]
    own = next(p for p in obs["players"] if p["color"] == actor)
    node = inputs["action"][2]
    obs["buildings"] = [[node, actor, "SETTLEMENT"]]
    own["resources"].update({r: 0 for r in own["resources"]})
    own["resources"].update(ORE=3, WHEAT=2)
    inputs["action"] = [actor, "BUILD_CITY", node]
    inputs["legal_actions"] = [inputs["action"]]
    v = values(encoder, inputs)
    assert v["own_building_vp"] == 1
    assert v["action_estimated_vp_gain"] == 1
    assert v["own_city_can_afford"] == 1 and v["own_city_resource_deficit"] == 0
    assert v["after_action_city_can_afford"] == 0 and v["after_action_city_resource_deficit"] == 5


def test_road_opens_new_settlement_site_and_enemy_buildings_block_paths():
    encoder, inputs = fixture()
    actor = inputs["observation"]["actor"]
    path = path_on_map(encoder, 3)
    graph = encoder.strategic
    buildings = {path[0]: (actor, "SETTLEMENT")}
    before = graph.reachable_sites(actor, {tuple(sorted(path[:2])): actor}, buildings)
    after = graph.reachable_sites(actor, {tuple(sorted(path[:2])): actor, tuple(sorted(path[1:3])): actor}, buildings)
    assert before[path[2]] == 1 and after[path[2]] == 0
    blocked = {**buildings, path[1]: ("ENEMY", "SETTLEMENT")}
    assert path[2] not in graph.reachable_sites(actor, {tuple(sorted(path[:2])): actor}, blocked)


def test_color_renaming_and_duplicate_undirected_roads_do_not_change_v2():
    encoder, inputs = fixture()
    actor = inputs["observation"]["actor"]
    path = path_on_map(encoder, 1)
    inputs["observation"]["roads"] = [[path, actor]]
    duplicated = deepcopy(inputs)
    duplicated["observation"]["roads"].append([path[::-1], actor])
    assert encoder.encode(inputs) == encoder.encode(duplicated)
    colors = [p["color"] for p in inputs["observation"]["players"]]
    mapping = dict(zip(colors, reversed(colors)))
    changed = deepcopy(inputs)
    for key in ("actor", "turn_owner"):
        changed["observation"][key] = mapping[changed["observation"][key]]
    for p in changed["observation"]["players"]:
        p["color"] = mapping[p["color"]]
    for b in changed["observation"]["buildings"]:
        b[1] = mapping[b[1]]
    for r in changed["observation"]["roads"]:
        r[1] = mapping[r[1]]
    changed["action"][0] = mapping[changed["action"][0]]
    for a in changed["legal_actions"]:
        a[0] = mapping[a[0]]
    assert encoder.encode(changed) == encoder.encode(inputs)


def test_v2_train_only_normalizer_preserves_boolean_flags():
    encoder, inputs = fixture()
    schema = feature_schema(STRATEGIC_VERSION)
    x = encoder.encode(inputs)
    scaler = fit_normalizer([x, x], schema)
    assert scaler["feature_version"] == STRATEGIC_VERSION
    assert len(scaler["offset"]) == 228
    for name in schema["binary_features"]:
        i = schema["feature_names"].index(name)
        assert scaler["offset"][i] == 0 and scaler["scale"][i] == 1


def test_third_knight_awards_two_points_but_does_not_take_tied_army():
    encoder, inputs = fixture()
    obs, actor = inputs["observation"], inputs["observation"]["actor"]
    own = next(p for p in obs["players"] if p["color"] == actor)
    own["PLAYED_KNIGHT"] = 2
    inputs["action"] = [actor, "PLAY_KNIGHT_CARD", None]
    inputs["legal_actions"] = [inputs["action"]]
    assert values(encoder, inputs)["action_estimated_vp_gain"] == 2
    enemy = next(p for p in obs["players"] if p["color"] != actor)
    enemy["PLAYED_KNIGHT"], enemy["HAS_ARMY"] = 3, True
    assert values(encoder, inputs)["action_army_award_gain"] == 0


def test_second_initial_settlement_includes_known_starting_resource_grant():
    encoder, inputs = fixture()
    actor = inputs["observation"]["actor"]
    first = inputs["action"][2]
    node = max(encoder.strategic.neighbors)
    inputs["observation"]["buildings"] = [[first, actor, "SETTLEMENT"]]
    own = next(p for p in inputs["observation"]["players"] if p["color"] == actor)
    own["resources"].update({r: 0 for r in own["resources"]})
    inputs["action"] = [actor, "BUILD_SETTLEMENT", node]
    inputs["legal_actions"] = [inputs["action"]]
    grant = {r: sum(t.resource == r for t in encoder.map.adjacent_tiles[node]) for r in own["resources"]}
    expected = sum(max(need-grant[r], 0) for r, need in {"WOOD": 1, "BRICK": 1}.items())
    assert values(encoder, inputs)["after_action_road_resource_deficit"] == expected


def test_versioned_linear_and_tree_consumers_accept_new_schema(tmp_path):
    from catanzero.linear_model import LinearMoveScorer
    from catanzero.static_policies import ObservationPolicy
    from catanzero.learned_match import LearnedPlayer
    from catanzero.regression_search import RegressionSearch
    encoder, inputs = fixture()
    game = create_game(SimulationConfig(), 84)
    map_document = map_to_json(game.state.board.map)
    schema = feature_schema(STRATEGIC_VERSION)
    vectors = [encoder.encode({**inputs, "action": a}) for a in inputs["legal_actions"]]
    normalizer = fit_normalizer(vectors, schema)
    norm_path = tmp_path / "normalizer.json"
    norm_path.write_text(json.dumps(normalizer))
    weights = [0.]*schema["input_columns"]
    weights[schema["feature_names"].index("location_yield_rank")] = 1.
    path = tmp_path / "linear_model.json"
    path.write_text(json.dumps({"feature_version": STRATEGIC_VERSION, "feature_names": schema["feature_names"],
                                "coefficients": weights, "intercept": 0.}))
    scorer = LinearMoveScorer(path)
    scored = scorer.rank(map_document, inputs)
    assert scored and scorer.feature_version == STRATEGIC_VERSION
    with pytest.raises(ValueError, match="version mismatch"):
        scorer.score(FeatureEncoder(map_document), inputs)
    tree_path = tmp_path / "trees.json"
    tree_path.write_text(json.dumps({"feature_version": STRATEGIC_VERSION, "feature_names": schema["feature_names"],
                                    "input_columns": 228, "baseline": .5, "trees": []}))
    manifest = {"normalizer_path": str(norm_path), "test": {"portable_path": str(tree_path)}}
    tree_policy = ObservationPolicy("test", map_document, manifest)
    assert tree_policy.encoder.version == STRATEGIC_VERSION
    assert tree_policy.choose(inputs)[0] == 0
    learned = LearnedPlayer(game.state.current_color(), "test", tree_path, norm_path)
    assert learned.decide(game, ordered_actions(game)) == ordered_actions(game)[0]
    search = RegressionSearch({"weights": weights, "intercept": 0., "feature_clip": 4}, normalizer, map_document)
    assert search.encoder.version == STRATEGIC_VERSION
    assert search.choose(game, ordered_actions(game), 100)[0] in range(len(ordered_actions(game)))
