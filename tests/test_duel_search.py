import json
import gzip
from pathlib import Path
import random

import pytest
from catanatron.serialization import action_to_json, map_to_json, state_to_json
from catanzero.dataset import observation, ordered_actions
from catanzero.duel_search import CONDITIONS, run_match
from catanzero.features import FeatureEncoder, BUILD_VERSION, VERSION, feature_schema
from catanzero.regression_search import RegressionSearch, public_f_choice
from catanzero.search_labels import live_belief
from catanzero.simulation import create_game, SimulationConfig
from catanzero.static_policies import forbid_forward_simulation

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "runs/online-sarsa-features-100-v4"


def inputs(game):
    actions = [action_to_json(a) for a in ordered_actions(game)]
    return {"observation": observation(game, game.state.current_color()),
            "legal_actions": actions, "action": actions[0], "map_ref": 0}


@pytest.mark.parametrize("players", [2, 3, 4])
@pytest.mark.parametrize("version", [VERSION, BUILD_VERSION])
def test_explicit_transfer_zero_pads_absent_slots_and_preserves_four_players(players, version):
    game = create_game(SimulationConfig(players=players), 43_000_081)
    document = map_to_json(game.state.board.map)
    encoder = FeatureEncoder(document, version, allow_fewer_players=True)
    with forbid_forward_simulation():
        vector = encoder.encode(inputs(game))
    features = dict(zip(feature_schema(version)["feature_names"], vector))
    for seat in range(players, 4):
        assert features[f"turn_owner_relative_{seat}"] == 0
        assert all(value == 0 for name, value in features.items() if name.startswith(f"opponent_{seat}_"))
    if players == 4:
        assert vector == FeatureEncoder(document, version).encode(inputs(game))
    else:
        with pytest.raises(ValueError, match="four-player"):
            FeatureEncoder(document, version).encode(inputs(game))


@pytest.mark.parametrize("opponent", ["F_static", "F"])
def test_two_player_search_and_native_f_preserve_real_game_and_future_rng(opponent):
    game = create_game(SimulationConfig(players=2), 43_000_082)
    changed = game.copy()
    changed.random = changed.state.random = random.Random(999)
    changed.state.development_listdeck.reverse()
    document = map_to_json(game.state.board.map)
    model = json.loads((SOURCE / "linear_model.json").read_text())
    normalizer = json.loads((SOURCE / "normalizer.json").read_text())
    before, stream = state_to_json(game), game.state.random.getstate()
    a, b = [RegressionSearch(model, normalizer, document, depth=2,
                             allow_fewer_players=True, opponent_policy=opponent) for _ in range(2)]
    first = a.choose(game, ordered_actions(game), 123)
    assert first == b.choose(changed, ordered_actions(changed), 123)
    assert a.stats["simulated_actions"] > 0
    assert public_f_choice(game, ordered_actions(game), 987) == public_f_choice(changed, ordered_actions(changed), 987)
    assert before == state_to_json(game) and stream == game.state.random.getstate()


def test_all_four_real_duel_conditions_can_run_and_censor_explicitly(tmp_path):
    config = {"base_seed": 43_000_090, "model": str(SOURCE / "linear_model.json"),
              "normalizer": str(SOURCE / "normalizer.json"), "excluded_board_hashes": [],
              "beam": 3, "beliefs": 2, "max_branch_actions": 24,
              "max_actions": 12, "max_seconds": 30}
    hashes = set()
    for condition in CONDITIONS:
        result = run_match(config, condition, 0, tmp_path / condition)
        assert result["players"] == 2 and result["actions"] == 12
        assert result["termination"] == "action_limit" and result["candidate_won"] is None
        hashes.add(result["initial_map_hash"])
        assert result["search_stats"]["simulated_actions"] > 0 if result["v4_search"] else result["search_stats"]["simulated_actions"] == 0
    assert len(hashes) == 1


def test_rare_live_belief_rejection_recovers_the_observed_failed_duel():
    prefix = REPO / "runs/v4-vs-f-duel-search-100-v1/both/game_0028_failed_attempt/decisions.jsonl.gz"
    if not prefix.exists():
        pytest.skip("saved regression fixture is available with this experiment")
    game = create_game(SimulationConfig(players=2), 42_000_028)
    with gzip.open(prefix, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            game.execute(ordered_actions(game)[row["selected_index"]])
    seed = 42_000_028 * 1000003 + len(game.state.action_records)
    actor = game.state.current_color()
    with pytest.raises(ValueError, match="could not sample"):
        live_belief(game, actor, seed, max_attempts=128)
    before, rng = state_to_json(game), game.state.random.getstate()
    proxy = live_belief(game, actor, seed)
    assert proxy.winning_color() is None
    assert observation(proxy, actor) == observation(game, actor)
    assert before == state_to_json(game) and rng == game.state.random.getstate()
