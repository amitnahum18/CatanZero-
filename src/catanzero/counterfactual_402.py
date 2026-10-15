"""Paired terminal returns: a frozen learned policy versus public-information F."""
from math import comb
import random
from time import perf_counter

import numpy as np
from catanatron.serialization import action_to_json, map_to_json
from catanzero.dataset import observation, ordered_actions
from catanzero.features import FeatureEncoder
from catanzero.regression_search import public_f_choice
from catanzero.search_labels import live_belief
from catanzero.static_policies import forbid_forward_simulation


def learned_choice(game, actions, policy, encoder):
    actor = game.state.current_color()
    inputs = {'observation': observation(game, actor),
              'legal_actions': [action_to_json(a) for a in actions], 'map_ref': 0}
    with forbid_forward_simulation():
        raw = np.asarray([encoder.encode({**inputs, 'action': a}) for a in inputs['legal_actions']])
        scores = policy.predict(raw)
    return actions[int(np.argmax(scores))]


def terminal_rollout(game, action, seed, policy, *, max_actions=1200, deadline=None):
    """Never reuse the authoritative hidden hand or future environment stream."""
    actor = action.color
    child = live_belief(game, actor, seed)
    child.random = child.state.random = random.Random(seed ^ 0xD1CE)
    child.state.action_records = []
    if action not in ordered_actions(child):
        raise ValueError('forced action is illegal in public belief')
    child.execute(action)
    encoder = FeatureEncoder(map_to_json(child.state.board.map),
                             version=policy.normalizer['feature_version'], allow_fewer_players=True)
    policy_rng = random.Random(seed ^ 0xB311EF)
    steps, learned_decisions, opponent_decisions = 1, 0, 0
    while child.winning_color() is None and steps < max_actions:
        if deadline is not None and perf_counter() >= deadline:
            break
        actions = ordered_actions(child)
        before = child.state.random.getstate()
        if len(actions) == 1:
            selected = actions[0]
        elif child.state.current_color() == actor:
            selected = learned_choice(child, actions, policy, encoder)
            learned_decisions += 1
        else:
            selected = public_f_choice(child, actions, policy_rng.getrandbits(128))
            opponent_decisions += 1
        if child.state.random.getstate() != before:
            raise AssertionError('policy deliberation advanced the environment RNG')
        child.execute(selected)
        steps += 1
    winner = child.winning_color()
    return {'outcome': None if winner is None else int(winner == actor), 'steps': steps,
            'learned_decisions': learned_decisions, 'F_decisions': opponent_decisions}


def pair_statistics(outcomes):
    n = len(outcomes[0])
    if n == 0 or len(outcomes[1]) != n:
        raise ValueError('nonempty paired trials required')
    censored = sum(a is None or b is None for a, b in zip(*outcomes))
    plus = sum(a == 0 and b == 1 for a, b in zip(*outcomes))
    minus = sum(a == 1 and b == 0 for a, b in zip(*outcomes))
    discordant = plus + minus
    p = min(1., 2*sum(comb(discordant, k) for k in range(min(plus, minus)+1))/2**discordant)
    return {'allocated_pairs': n, 'censored_pairs': censored,
            'alternative_only_wins': plus, 'baseline_only_wins': minus,
            'paired_p_two_sided': p,
            'reward_advantage_alternative': None if censored else 2*(plus-minus)/n,
            'training_eligible': censored == 0 and plus != minus,
            'statistically_resolved_05': censored == 0 and p < .05}


def evaluate_pair(game, actions, seeds, policy, *, max_seconds=120., max_actions=1200):
    if len(actions) != 2 or actions[0] == actions[1]:
        raise ValueError('two distinct legal actions required')
    actor = actions[0].color
    before = (game.state.random.getstate(), observation(game, actor), len(game.state.action_records))
    deadline = perf_counter()+max_seconds
    outcomes = [[], []]
    counts = {'steps': 0, 'learned_decisions': 0, 'F_decisions': 0}
    for trial, seed in enumerate(seeds):
        pair = [None, None]
        for index in ((0, 1) if trial % 2 == 0 else (1, 0)):
            if perf_counter() >= deadline:
                continue
            result = terminal_rollout(game, actions[index], seed, policy,
                                      max_actions=max_actions, deadline=deadline)
            pair[index] = result['outcome']
            for key in counts:
                counts[key] += result[key]
        for index in range(2):
            outcomes[index].append(pair[index])
    if before != (game.state.random.getstate(), observation(game, actor), len(game.state.action_records)):
        raise AssertionError('counterfactual evaluation modified its source game')
    return {'trial_outcomes': outcomes, **pair_statistics(outcomes), **counts}
