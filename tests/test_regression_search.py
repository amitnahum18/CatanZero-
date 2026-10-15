import json
from pathlib import Path
import random

import pytest
from catanatron.models.enums import RESOURCES
from catanatron.serialization import map_to_json, state_to_json
from catanatron.state_functions import player_key
from catanzero.dataset import observation, ordered_actions
from catanzero.regression_search import RegressionSearch, backed_up_value
from catanzero.simulation import SimulationConfig, create_game
from catanzero.static_policies import forbid_forward_simulation


def policy(game, depth, **kwargs):
    source = Path(__file__).resolve().parents[1] / "runs/online-sarsa-low-exploration-500-v3"
    return RegressionSearch(json.loads((source / "linear_model.json").read_text()),
                            json.loads((source / "normalizer.json").read_text()), map_to_json(game.state.board.map), depth=depth, **kwargs)


def test_depth_one_is_direct_regression_without_any_simulation():
    game = create_game(SimulationConfig(), 81)
    p = policy(game, 1)
    with forbid_forward_simulation():
        selected, detail = p.choose(game, ordered_actions(game), 123)
    assert selected == max(range(len(detail['base_scores'])), key=detail['base_scores'].__getitem__)
    assert p.stats['simulated_actions'] == 0


def test_search_preserves_real_game_and_is_invariant_to_hidden_hands_and_future_rng():
    game = create_game(SimulationConfig(), 81)
    actor = game.state.current_color()
    enemies = [c for c in game.state.colors if c != actor]
    first, second = [player_key(game.state, c) for c in enemies[:2]]
    game.state.player_state[f'{first}_WOOD_IN_HAND'] = 2
    game.state.player_state[f'{second}_ORE_IN_HAND'] = 2
    game.state.resource_freqdeck[RESOURCES.index('WOOD')] -= 2
    game.state.resource_freqdeck[RESOURCES.index('ORE')] -= 2
    changed = game.copy()
    changed.state.player_state[f'{first}_WOOD_IN_HAND'] = 0
    changed.state.player_state[f'{first}_ORE_IN_HAND'] = 2
    changed.state.player_state[f'{second}_ORE_IN_HAND'] = 0
    changed.state.player_state[f'{second}_WOOD_IN_HAND'] = 2
    changed.state.development_listdeck.reverse()
    changed.random = changed.state.random = random.Random(918)
    assert observation(game, actor) == observation(changed, actor)
    before, stream = state_to_json(game), game.state.random.getstate()
    p, q = policy(game, 2), policy(changed, 2)
    result = p.choose(game, ordered_actions(game), 123)
    assert result == q.choose(changed, ordered_actions(changed), 123)
    assert before == state_to_json(game) and stream == game.state.random.getstate()
    assert p.stats['simulated_actions'] > 0 and not result[1]['fallback']
    assert len(result[1]['expanded_indices']) == 3


def test_backup_matches_sarsa_reward_and_terminal_win():
    assert backed_up_value(2, 3, 4, .25, gamma=.9) == pytest.approx(.9**4*(.15+.25)-.1)
    assert backed_up_value(2, 0, 4, 999, won=True, gamma=.9) == pytest.approx(.9**3-.1)
    assert backed_up_value(2, 0, 4, 999, won=False) == -.1
    with pytest.raises(ValueError):
        backed_up_value(0, 0, 0, 0)


def test_cutoff_falls_back_instead_of_ranking_incomplete_branches():
    game = create_game(SimulationConfig(), 81)
    # After settlement, initial road is forced and leads to another player;
    # a one-action branch budget cannot reach the next own decision.
    game.execute(ordered_actions(game)[0])
    p = policy(game, 2, max_branch_actions=1)
    selected, detail = p.choose(game, ordered_actions(game), 123)
    assert detail['fallback']
    assert selected == max(range(len(detail['base_scores'])), key=detail['base_scores'].__getitem__)
    assert all(value is None for value in detail['search_scores'])
