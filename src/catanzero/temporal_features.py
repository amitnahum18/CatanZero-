"""Experimental causal history summaries, separate from deployed feature schemas.

Call features() before selecting the current action; commit() once after the
real action has been selected/executed. Only prior own-decision observations
are retained. No model, target, reward, seed, or simulator access.
"""
from collections import deque
from math import sqrt

LAGS = (1, 3, 5)
METRICS = ("own_vp", "resources", "city_deficit", "settlement_deficit", "road_length", "leader_public_vp")
PAST_ACTIONS = ("END_TURN", "BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY", "MARITIME_TRADE")


def snapshot(obs):
    own = next(p for p in obs["players"] if p["color"] == obs["actor"])
    hand = own["resources"]
    return {"own_vp": float(own["actual_vp"]), "resources": float(sum(hand.values())),
            "city_deficit": float(max(3-hand['ORE'], 0)+max(2-hand['WHEAT'], 0)),
            "settlement_deficit": float(sum(max(1-hand[r], 0) for r in ('WOOD','BRICK','SHEEP','WHEAT'))),
            "road_length": float(own["LONGEST_ROAD_LENGTH"]),
            "leader_public_vp": float(max(p["VICTORY_POINTS"] for p in obs["players"] if p["color"] != obs["actor"]))}


class TemporalFeatures:
    def __init__(self):
        self.game_key = None
        self.histories = {}

    def history(self, obs, game_key):
        if game_key != self.game_key:
            self.game_key = game_key
            self.histories = {}
        return self.histories.setdefault(obs["actor"], deque(maxlen=5))

    def features(self, obs, game_key):
        history = self.history(obs, game_key)
        current = snapshot(obs)
        out = {}
        for lag in LAGS:
            exists = len(history) >= lag
            out[f"history_{lag}_available"] = float(exists)
            for metric in METRICS:
                out[f"history_{lag}_delta_{metric}"] = current[metric]-history[-lag][0][metric] if exists else 0.
        for metric in ("resources", "city_deficit"):
            values = [item[0][metric] for item in history]
            mean = sum(values)/len(values) if values else 0.
            std = sqrt(sum((v-mean)**2 for v in values)/len(values)) if values else 0.
            out[f"history_5_mean_{metric}"] = mean
            out[f"history_5_std_{metric}"] = std
        for kind in PAST_ACTIONS:
            out[f"history_5_share_{kind}"] = sum(item[1] == kind for item in history)/len(history) if history else 0.
        return out

    def commit(self, obs, game_key, action_kind):
        self.history(obs, game_key).append((snapshot(obs), action_kind))
