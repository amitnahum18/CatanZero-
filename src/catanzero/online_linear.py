"""Portable linear action values for online SARSA (no engine dependency)."""

import json
from math import isfinite
from pathlib import Path


class LinearActionValues:
    def __init__(self, model, normalizer):
        self.model, self.normalizer = model, normalizer
        self.weights = model["weights"]
        self.intercept = model["intercept"]
        self.size = len(self.weights)
        if self.size != len(normalizer["feature_names"]) or normalizer["fit_split"] != "train":
            raise ValueError("feature contract or normalization split mismatch")
        if len(normalizer["offset"]) != self.size or len(normalizer["scale"]) != self.size:
            raise ValueError("invalid normalization size")
        if any(s <= 0 or not isfinite(s) for s in normalizer["scale"]):
            raise ValueError("invalid scale")
        if not all(isfinite(v) for v in (*self.weights, self.intercept, *normalizer["offset"])):
            raise ValueError("nonfinite linear model")
        self.clip = model.get("feature_clip", 4.0)
        if self.clip <= 0 or not isfinite(self.clip):
            raise ValueError("invalid feature clip")

    @classmethod
    def load(cls, model, normalizer):
        return cls(json.loads(Path(model).read_text()), json.loads(Path(normalizer).read_text()))

    def normalize(self, x):
        if len(x) != self.size or not all(isfinite(v) for v in x):
            raise ValueError("invalid feature vector")
        return [max(-self.clip, min(self.clip, (v-o)/s))
                for v, o, s in zip(x, self.normalizer["offset"], self.normalizer["scale"])]

    def predict(self, x):
        return self.intercept + sum(w*v for w, v in zip(self.weights, self.normalize(x)))


def shaped_transition(previous, following, *, terminal, won, gamma=.999, vp_weight=.05, gap_weight=0.0):
    """Same-player semi-MDP reward; discounts all actual intervening actions."""
    gap = following["action_index"] - previous["action_index"]
    if gap < 1:
        raise ValueError("transition must advance the real trajectory")
    if previous["learner_color"] != following["learner_color"]:
        raise ValueError("transition crossed players")
    if terminal and won not in (True, False):
        raise ValueError("terminal outcome must be known; truncation is not a loss")
    continuation = 0.0 if terminal else gamma ** gap
    terminal_reward = gamma ** (gap-1) * float(won) if terminal else 0.0
    if not isfinite(gap_weight) or gap_weight < 0:
        raise ValueError('gap_weight must be finite and nonnegative')
    phi_previous = vp_weight * previous['vp']
    phi_next = 0.0 if terminal else vp_weight * following["vp"]
    if gap_weight:
        if 'leader_public_vp' not in previous or (not terminal and 'leader_public_vp' not in following):
            raise ValueError('Relative shaping requires observed public opponent leader points')
        phi_previous += gap_weight * (previous['vp']-previous['leader_public_vp'])
        if not terminal:
            phi_next += gap_weight * (following['vp']-following['leader_public_vp'])
    reward = terminal_reward + continuation * phi_next - phi_previous
    return reward, continuation
