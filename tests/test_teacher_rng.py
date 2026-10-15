"""Stochastic offline labels must not depend on candidate enumeration order."""
import gzip
import json
from pathlib import Path

from catanatron.players.value import get_value_fn
from catanatron.serialization import action_to_json
from catanzero.dataset import observation, ordered_actions
from catanzero.label_f_policy import detached_label_copy
from catanzero.search_labels import live_belief
from catanzero.simulation import SimulationConfig, create_game


def test_robber_label_scores_are_invariant_to_candidate_order():
    repo = Path(__file__).resolve().parents[1]
    seed, action_index = 46_000_000, 39
    game = create_game(SimulationConfig(players=2), seed)
    trace = repo / 'runs/residual-extratrees-depth10-f-duel-v1/training_round_1/game_0000/decisions.jsonl.gz'
    with gzip.open(trace, 'rt', encoding='utf-8') as stream:
        for row in map(json.loads, stream):
            if row['action_index'] == action_index:
                break
            selected = next(action for action in ordered_actions(game)
                            if action_to_json(action) == row['inputs']['action'])
            game.execute(selected)
    assert len(game.state.action_records) == action_index
    actor = game.state.current_color()
    actions = ordered_actions(game)
    view = live_belief(game, actor, seed*1_000_003+action_index)
    original_rng = view.state.random.getstate()
    original_observation = observation(view, actor)
    value = get_value_fn('base_fn', None)

    def score(order):
        values = {}
        for index in order:
            child = detached_label_copy(view)
            child.execute(actions[index])
            values[index] = value(child, actor)
        return values

    forward = score(range(len(actions)))
    reverse = score(reversed(range(len(actions))))
    assert forward == reverse
    # These two robber destinations have the same encoded effects and victim.
    # Previously their scores differed solely through different random steals.
    assert forward[8] == forward[11]
    assert view.state.random.getstate() == original_rng
    assert observation(view, actor) == original_observation
