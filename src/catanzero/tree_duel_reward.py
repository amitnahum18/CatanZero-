"""Symmetric outcome reward and undiscounted potential shaping for duels."""
from math import isfinite


def potential(vp, opponent_public_vp):
    return .05*vp + .02*(vp-opponent_public_vp)


def transition_reward(previous, following, *, terminal, won=None):
    if following["action_index"] <= previous["action_index"]:
        raise ValueError("transition must advance real game actions")
    if previous["learner_color"] != following["learner_color"]:
        raise ValueError("transition cannot cross actors")
    if terminal and won not in (True, False):
        raise ValueError("a censored game is not a terminal loss")
    before = potential(previous["vp"], previous["leader_public_vp"])
    after = 0. if terminal else potential(following["vp"], following["leader_public_vp"])
    outcome = (1. if won else -1.) if terminal else 0.
    reward, discount = outcome+after-before, 0. if terminal else 1.
    if not isfinite(reward):
        raise ValueError("nonfinite reward")
    return reward, discount
