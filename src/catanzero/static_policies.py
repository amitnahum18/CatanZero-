"""Direct observation-to-action policies, with no game transition or rollout."""

from contextlib import contextmanager
from math import isfinite
from unittest.mock import patch

from catanatron import Game
from catanatron.models.enums import RESOURCES

from catanzero.features import FeatureEncoder, feature_schema, VERSION
from catanzero.portable_trees import PortableTrees


STATIC_WEIGHTS = {
    "vp": 1000.0, "production": 100.0, "production_balance": 20.0,
    "new_resource": 12.0, "enemy_blocked": 60.0, "hand_readiness": 4.0,
    "hand_count": 0.1, "excess_over_seven": -0.2,
    "port": 1.0, "road_target_yield": 12.0, "road_new_endpoint": 1.0,
    "development_purchase": 2.0, "knight": 0.5, "free_roads_card": 1.0,
    "monopoly_public_estimate": 0.3, "robber_victim_vp": 0.05,
}
STATIC_VERSION = "observation_heuristic_v1"


@contextmanager
def forbid_forward_simulation():
    """Fail even when a policy catches a forbidden engine call internally.

    The real game's one selected transition happens outside this scope.
    This is an assertion for our trusted policies, not an untrusted-code sandbox.
    """
    attempts = []

    def blocked(method):
        def reject(*args, **kwargs):
            attempts.append(method)
            raise RuntimeError(f"forward simulation forbidden: Game.{method}")
        return reject

    with patch.object(Game, "copy", blocked("copy")), \
            patch.object(Game, "execute", blocked("execute")), \
            patch.object(Game, "__init__", blocked("__init__")):
        yield attempts
        if attempts:
            raise RuntimeError(f"policy attempted forward simulation: {attempts}")


class ObservationPolicy:
    """Receives a static map and whitelisted observation, never a Game."""

    def __init__(self, policy, map_document, manifest, *, allow_fewer_players=False):
        self.policy = policy
        self.scorer = None if policy == "F_static" else PortableTrees.load(
            manifest[policy]["portable_path"], manifest["normalizer_path"])
        version = VERSION if self.scorer is None else self.scorer.model.get("feature_version", VERSION)
        self.encoder = FeatureEncoder(map_document, version=version, allow_fewer_players=allow_fewer_players)
        self.names = feature_schema(version)["feature_names"]
        if self.scorer is not None and self.scorer.model["feature_names"] != self.names:
            raise ValueError("model feature schema mismatch")
        # Current board topology only. No game states or future legal moves.
        self.neighbors = {n: set() for n in self.encoder.map.land_nodes}
        for tile in self.encoder.map.tiles.values():
            for edge in tile.edges.values():
                a, b = edge
                if a in self.neighbors and b in self.neighbors:
                    self.neighbors[a].add(b)
                    self.neighbors[b].add(a)

    @staticmethod
    def hand_utility(hand):
        recipes = ({"ORE": 3, "WHEAT": 2},
                   {"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 1},
                   {"ORE": 1, "SHEEP": 1, "WHEAT": 1})
        readiness = sum(sum(min(hand[r], need) for r, need in recipe.items()) / sum(recipe.values())
                        for recipe in recipes)
        count = sum(hand.values())
        return (STATIC_WEIGHTS["hand_readiness"] * readiness + STATIC_WEIGHTS["hand_count"] * count
                + STATIC_WEIGHTS["excess_over_seven"] * max(count - 7, 0))

    def heuristic(self, inputs, vector):
        f = dict(zip(self.names, vector))
        obs = inputs["observation"]
        _, kind, value = inputs["action"]
        w = STATIC_WEIGHTS
        hand = {r: f[f"own_resource_{r}"] for r in RESOURCES}
        # Arithmetic of known costs/gains; no engine step or hypothetical board.
        adjusted_hand = {r: hand[r] - f[f"action_cost_{r}"] + f[f"action_gain_{r}"] for r in RESOURCES}
        score = self.hand_utility(adjusted_hand) - self.hand_utility(hand)
        score += w["vp"] * f["vp_gain"]
        score += w["production"] * sum(f[f"production_delta_{r}"] for r in RESOURCES)
        score += w["enemy_blocked"] * f["robber_enemy_production_blocked"]
        if kind in ("BUILD_SETTLEMENT", "BUILD_CITY"):
            score += w["production_balance"] * sum(f[f"location_production_balance_{r}"] for r in RESOURCES)
            score += w["new_resource"] * sum(f[f"production_delta_{r}"] > 0 and f[f"own_production_{r}"] == 0
                                           for r in RESOURCES)
            if kind == "BUILD_SETTLEMENT":
                score += w["port"] * sum(f[f"location_port_{r}"] and not f[f"own_port_{r}"]
                                        for r in ("generic", *RESOURCES))
        elif kind == "BUILD_ROAD":
            occupied = {n for n, _, _ in obs["buildings"]}
            actor = obs["actor"]
            own_endpoints = {n for edge, color in obs["roads"] if color == actor for n in edge}
            for node in value:
                if node in occupied or node in own_endpoints or self.neighbors[node].intersection(occupied):
                    continue
                from catanatron.models.map import number_probability
                yield_total = sum(number_probability(t.number) for t in self.encoder.node_tiles[node] if t.resource)
                score += w["road_new_endpoint"] + w["road_target_yield"] * yield_total
        elif kind == "BUY_DEVELOPMENT_CARD":
            score += w["development_purchase"]
        elif kind == "PLAY_KNIGHT_CARD":
            score += w["knight"]
        elif kind == "PLAY_ROAD_BUILDING":
            score += w["free_roads_card"]
        elif kind == "PLAY_MONOPOLY":
            # Only public total hand sizes; never guesses an exact hidden hand.
            score += w["monopoly_public_estimate"] * sum(
                p["resource_count"] for p in obs["players"] if p["color"] != obs["actor"]
            ) / len(RESOURCES) / (1 + hand[value])
        elif kind == "MOVE_ROBBER" and f["victim_resource_count"]:
            score += w["robber_victim_vp"] * f["victim_public_vp"]
        return score

    def choose(self, inputs):
        actions = inputs["legal_actions"]
        if not actions:
            raise ValueError("no legal actions")
        if len(actions) == 1:
            return 0, None
        scores = []
        for action in actions:
            candidate = {**inputs, "action": action}
            vector = self.encoder.encode(candidate)
            scores.append(self.heuristic(candidate, vector) if self.scorer is None
                          else self.scorer.predict_raw_features(vector))
        if not all(isfinite(s) for s in scores):
            raise ValueError("nonfinite policy score")
        return max(range(len(scores)), key=scores.__getitem__), scores
