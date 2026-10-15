"""Fixed, actor-relative predecision features for a small linear regressor.

Only a public board map and the inputs whitelist are read. No simulator step,
hidden-hand sample, target, heuristic value, seed or trajectory is consulted.
The spatial encoding summarizes locations; it is deliberately not lossless.
"""

from collections import Counter
from functools import lru_cache
from math import isfinite

from catanatron.models.enums import RESOURCES, DEVELOPMENT_CARDS, ActionPrompt
from catanatron.models.map import LandTile, number_probability
from catanatron.serialization import map_from_json

VERSION = "predecision_relative_spatial_linear_v1"
STRATEGIC_VERSION = "predecision_relative_strategic_v2"
BUILD_VERSION = "predecision_relative_build_capacity_v3"
ACTION_TYPES = ("ROLL", "MOVE_ROBBER", "DISCARD_RESOURCE", "BUILD_ROAD",
                "BUILD_SETTLEMENT", "BUILD_CITY", "BUY_DEVELOPMENT_CARD",
                "PLAY_KNIGHT_CARD", "PLAY_YEAR_OF_PLENTY", "PLAY_MONOPOLY",
                "PLAY_ROAD_BUILDING", "MARITIME_TRADE", "END_TURN")
PLAYER_FIELDS = ("VICTORY_POINTS", "ROADS_AVAILABLE", "SETTLEMENTS_AVAILABLE",
                 "CITIES_AVAILABLE", "HAS_ROAD", "HAS_ARMY", "HAS_ROLLED",
                 "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN", "LONGEST_ROAD_LENGTH",
                 *[f"PLAYED_{card}" for card in DEVELOPMENT_CARDS],
                 "resource_count", "development_count", "actual_vp")
OPPONENT_FIELDS = ("VICTORY_POINTS", "resource_count", "development_count",
                   "LONGEST_ROAD_LENGTH", "PLAYED_KNIGHT", "HAS_ARMY", "HAS_ROAD")


@lru_cache(maxsize=4)
def feature_schema(version=VERSION):
    if version not in (VERSION, STRATEGIC_VERSION, BUILD_VERSION):
        raise ValueError(f"unsupported feature version: {version}")
    groups = {}
    groups["context"] = ["turn", "initial_phase", "discarding", "moving_robber",
                         "free_roads_available", "discard_limit", "development_deck_size",
                         "target_vp", "legal_action_count"]
    groups["context"] += [f"bank_{r}" for r in RESOURCES]
    groups["context"] += [f"prompt_{p.value}" for p in ActionPrompt]
    groups["context"] += [f"turn_owner_relative_{i}" for i in range(4)]
    groups["own_player"] = [f"own_{field}" for field in PLAYER_FIELDS]
    groups["own_player"] += [f"own_resource_{r}" for r in RESOURCES]
    groups["own_player"] += [f"own_dev_{c}" for c in DEVELOPMENT_CARDS]
    groups["own_player"] += [f"own_playable_{c}" for c in DEVELOPMENT_CARDS if c != "VICTORY_POINT"]
    groups["own_player"] += [f"own_production_{r}" for r in RESOURCES]
    groups["own_player"] += ["own_port_generic"] + [f"own_port_{r}" for r in RESOURCES]
    groups["opponents"] = [name for i in range(1, 4)
                           for name in [*[f"opponent_{i}_{f}" for f in OPPONENT_FIELDS],
                                        f"opponent_{i}_production_total"]]
    groups["action_type"] = [f"action_{a}" for a in ACTION_TYPES]
    groups["action_resources"] = [f"action_{kind}_{r}" for kind in ("cost", "gain", "requested") for r in RESOURCES]
    groups["action_location"] = [f"location_{kind}_{r}" for kind in ("potential", "active") for r in RESOURCES]
    groups["action_location"] += ["location_port_generic"] + [f"location_port_{r}" for r in RESOURCES]
    groups["action_location"] += ["location_own_settlements", "location_own_cities",
                                  "location_enemy_buildings", "location_own_roads_touching",
                                  "location_enemy_roads_touching"]
    groups["deterministic_effect"] = [f"production_delta_{r}" for r in RESOURCES]
    groups["deterministic_effect"] += ["vp_gain", "robber_enemy_production_blocked",
                                        "victim_public_vp", "victim_resource_count"]
    groups["action_context"] = [f"{kind}_{r}" for kind in
                               ("cost_hand_fraction", "gain_build_deficit", "location_production_balance")
                               for r in RESOURCES]
    names = [name for group in groups.values() for name in group]
    binary = {"initial_phase", "discarding", "moving_robber"}
    binary.update(name for name in names if name.startswith(("prompt_", "turn_owner_relative_", "action_"))
                  and not name.startswith(("action_cost_", "action_gain_", "action_requested_")))
    binary.update(name for name in names if "_port_" in name or name.startswith("own_playable_"))
    binary.update(name for name in names if any(name.endswith(f"_{f}") for f in
                  ("HAS_ROAD", "HAS_ARMY", "HAS_ROLLED", "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN")))
    binary.update(name for name in names if name.startswith("action_requested_"))
    if version in (STRATEGIC_VERSION, BUILD_VERSION):
        from catanzero.strategic_features import EXTRA_GROUPS, EXTRA_BINARY
        groups.update(EXTRA_GROUPS)
        names = [name for group in groups.values() for name in group]
        binary.update(EXTRA_BINARY)
    if version == BUILD_VERSION:
        from catanzero.build_capacity import BUILD_GROUPS, BUILD_BINARY
        groups.update(BUILD_GROUPS)
        names = [name for group in groups.values() for name in group]
        binary.update(BUILD_BINARY)
    assert len(names) == len(set(names))
    return {"version": version, "input_columns": len(names), "output_columns": 1,
            "target_name": "weighted_score", "feature_names": names,
            "groups": {k: {"count": len(v), "names": v} for k, v in groups.items()},
            "binary_features": sorted(binary),
            "feature_information": "predecision own/private plus opponents/public; deterministic spatial summaries",
            "excluded": ["game_index", "action_index", "actor_bot", "split", "map_ref as a value",
                         "heuristic", "targets", "winner", "seed", "future outcomes",
                         "opponent exact resources/development cards/actual VP", "future RNG/deck order"]}


class FeatureEncoder:
    def __init__(self, map_document, version=VERSION, *, allow_fewer_players=False):
        feature_schema(version)
        self.version = version
        self.allow_fewer_players = allow_fewer_players
        self.map = map_from_json(map_document)
        self.node_tiles = self.map.adjacent_tiles
        self.node_ports = {node: {resource for resource, nodes in self.map.port_nodes.items() if node in nodes}
                           for node in self.map.land_nodes}
        self.tiles = self.map.tiles
        self.strategic = None
        if version in (STRATEGIC_VERSION, BUILD_VERSION):
            from catanzero.strategic_features import StrategicFeatures
            self.strategic = StrategicFeatures(self.map)

    def encode(self, inputs):
        obs, action = inputs["observation"], inputs["action"]
        actor, kind, value = action
        if actor != obs["actor"] or kind not in ACTION_TYPES:
            raise ValueError("unsupported action or actor mismatch")
        if kind in ("ROLL", "BUY_DEVELOPMENT_CARD", "PLAY_KNIGHT_CARD", "PLAY_ROAD_BUILDING", "END_TURN") and value is not None:
            raise ValueError("action includes a sampled outcome instead of a predecision intent")
        if action not in inputs["legal_actions"]:
            raise ValueError("chosen action is not in the legal-action list")
        players = obs["players"]
        if len(players) != 4 and not (self.allow_fewer_players and len(players) in (2, 3)):
            raise ValueError("this feature version supports four-player base games")
        seat = next(i for i, p in enumerate(players) if p["color"] == actor)
        ordered = players[seat:] + players[:seat]
        own = ordered[0]
        public_players = {p["color"]: p for p in players}
        buildings = {node: (color, bt) for node, color, bt in obs["buildings"]}
        roads = {}
        for edge, color in obs["roads"]:
            edge = tuple(sorted(edge))
            if edge in roads and roads[edge] != color:
                raise ValueError("conflicting owners for the same undirected road")
            roads[edge] = color
        robber = tuple(obs["robber"])
        units = {color: Counter() for color in public_players}
        for node, (color, bt) in buildings.items():
            units[color][node] = 2 if bt == "CITY" else 1

        def node_yield(node, active):
            result = Counter()
            for tile in self.node_tiles[node]:
                if tile.resource and (not active or self.tiles[robber] is not tile):
                    result[tile.resource] += number_probability(tile.number)
            return result

        production = {color: Counter() for color in public_players}
        for color, nodes in units.items():
            for node, amount in nodes.items():
                for resource, chance in node_yield(node, True).items():
                    production[color][resource] += amount * chance

        features = {key: float(obs[key]) for key in
                    ("turn", "initial_phase", "discarding", "moving_robber", "free_roads_available",
                     "discard_limit", "development_deck_size", "target_vp")}
        features["legal_action_count"] = len(inputs["legal_actions"])
        # Explicit transfer mode: absent relative seats have zero raw features.
        # Do not invent players or change the trained feature order/normalizer.
        features.update({f"turn_owner_relative_{i}": 0.0 for i in range(4)})
        for i in range(1, 4):
            features.update({f"opponent_{i}_{f}": 0.0 for f in OPPONENT_FIELDS})
            features[f"opponent_{i}_production_total"] = 0.0
        features.update({f"bank_{r}": obs["resource_bank"][r] for r in RESOURCES})
        features.update({f"prompt_{p.value}": float(obs["prompt"] == p.value) for p in ActionPrompt})
        features.update({f"turn_owner_relative_{i}": float(obs["turn_owner"] == p["color"])
                         for i, p in enumerate(ordered)})
        features.update({f"own_{field}": own[field] for field in PLAYER_FIELDS})
        features.update({f"own_resource_{r}": own["resources"][r] for r in RESOURCES})
        features.update({f"own_dev_{c}": own["development_cards"][c] for c in DEVELOPMENT_CARDS})
        features.update({f"own_playable_{c}": float(own["playable_at_turn_start"][c])
                         for c in DEVELOPMENT_CARDS if c != "VICTORY_POINT"})
        features.update({f"own_production_{r}": production[actor][r] for r in RESOURCES})
        owned_ports = set().union(*(self.node_ports[node] for node in units[actor])) if units[actor] else set()
        for resource in [None, *RESOURCES]:
            features[f"own_port_{resource or 'generic'}"] = float(resource in owned_ports)
        for i, player in enumerate(ordered[1:], 1):
            features.update({f"opponent_{i}_{f}": player[f] for f in OPPONENT_FIELDS})
            features[f"opponent_{i}_production_total"] = sum(production[player["color"]].values())
        features.update({f"action_{a}": float(kind == a) for a in ACTION_TYPES})
        cost, gain, requested = Counter(), Counter(), Counter()
        if kind == "BUILD_ROAD" and not obs["initial_phase"] and not obs["free_roads_available"]:
            cost.update(("WOOD", "BRICK"))
        elif kind == "BUILD_SETTLEMENT" and not obs["initial_phase"]:
            cost.update(("WOOD", "BRICK", "SHEEP", "WHEAT"))
        elif kind == "BUILD_CITY":
            cost.update({"ORE": 3, "WHEAT": 2})
        elif kind == "BUY_DEVELOPMENT_CARD":
            cost.update(("ORE", "SHEEP", "WHEAT"))
        elif kind == "MARITIME_TRADE":
            cost.update(r for r in value[:-1] if r is not None)
            gain[value[-1]] += 1
        elif kind == "DISCARD_RESOURCE":
            cost[value] += 1
        elif kind == "PLAY_YEAR_OF_PLENTY":
            gain.update(value)
        elif kind == "PLAY_MONOPOLY":
            requested[value] = 1
        for prefix, amounts in (("cost", cost), ("gain", gain), ("requested", requested)):
            features.update({f"action_{prefix}_{r}": amounts[r] for r in RESOURCES})

        location = [value] if kind in ("BUILD_SETTLEMENT", "BUILD_CITY") else list(value) if kind == "BUILD_ROAD" else []
        for active, prefix in ((False, "potential"), (True, "active")):
            loc_yield = Counter()
            for node in location:
                loc_yield.update(node_yield(node, active))
            features.update({f"location_{prefix}_{r}": loc_yield[r] / max(len(location), 1) for r in RESOURCES})
        location_ports = set().union(*(self.node_ports[n] for n in location)) if location else set()
        for resource in [None, *RESOURCES]:
            features[f"location_port_{resource or 'generic'}"] = float(resource in location_ports)
        features["location_own_settlements"] = sum(buildings.get(n) == (actor, "SETTLEMENT") for n in location)
        features["location_own_cities"] = sum(buildings.get(n) == (actor, "CITY") for n in location)
        features["location_enemy_buildings"] = sum(n in buildings and buildings[n][0] != actor for n in location)
        touching = [(edge, color) for edge, color in roads.items() if set(edge).intersection(location)]
        features["location_own_roads_touching"] = sum(color == actor for _, color in touching)
        features["location_enemy_roads_touching"] = sum(color != actor for _, color in touching)
        delta = Counter()
        if kind in ("BUILD_SETTLEMENT", "BUILD_CITY"):
            delta.update(node_yield(value, True))
        features["vp_gain"] = float(kind in ("BUILD_SETTLEMENT", "BUILD_CITY"))
        features["robber_enemy_production_blocked"] = 0.0
        features["victim_public_vp"] = features["victim_resource_count"] = 0.0
        if kind == "MOVE_ROBBER":
            coordinate, victim = value
            tile = self.tiles[tuple(coordinate)]
            old_tile = self.tiles[robber]
            if not isinstance(tile, LandTile):
                raise ValueError("robber target must be a land tile")
            def tile_yield(color, t):
                return sum(units[color][node] for node in t.nodes.values()) * number_probability(t.number) if t.resource else 0.0
            if old_tile.resource:
                delta[old_tile.resource] += tile_yield(actor, old_tile)
            if tile.resource:
                delta[tile.resource] -= tile_yield(actor, tile)
                features["robber_enemy_production_blocked"] = sum(tile_yield(p["color"], tile) for p in ordered[1:])
            if victim is not None:
                features["victim_public_vp"] = public_players[victim]["VICTORY_POINTS"]
                features["victim_resource_count"] = public_players[victim]["resource_count"]
        features.update({f"production_delta_{r}": delta[r] for r in RESOURCES})
        # Explicit interactions keep a linear scorer sensitive to why this
        # resource/action is useful in the current state, not just action type.
        requirements = {"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 2, "ORE": 3}
        for r in RESOURCES:
            hand = own["resources"][r]
            features[f"cost_hand_fraction_{r}"] = cost[r] / max(hand, 1)
            features[f"gain_build_deficit_{r}"] = gain[r] * max(requirements[r] - hand, 0)
            features[f"location_production_balance_{r}"] = features[f"location_active_{r}"] / (1 + production[actor][r])
        if self.strategic is not None:
            features.update(self.strategic.encode(inputs, features))
        if self.version == BUILD_VERSION:
            from catanzero.build_capacity import build_features
            features.update(build_features(inputs, features, self.strategic))
        schema = feature_schema(self.version)
        if set(features) != set(schema["feature_names"]) or any(not isfinite(float(v)) for v in features.values()):
            raise ValueError("invalid feature vector")
        return [float(features[name]) for name in schema["feature_names"]]
