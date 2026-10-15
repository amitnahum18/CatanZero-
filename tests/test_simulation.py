import json
import os
import subprocess
import sys

import pytest
from catanatron import Color
from catanatron.serialization import state_from_json, state_to_json

from catanzero.cli import main
from catanzero.simulation import SimulationConfig, SimulationRandomPlayer, run_game, summarize


def test_full_game_has_real_winner():
    game, result = run_game(SimulationConfig(), seed=42)
    assert result.termination == "victory"
    assert result.winner == game.winning_color().value
    assert result.victory_points[result.winner] >= 10
    assert result.actions == len(game.state.action_records) > 16
    assert set(result.seating_order) == {"RED", "BLUE", "WHITE", "ORANGE"}


@pytest.mark.parametrize(
    "config, reason",
    [(SimulationConfig(max_turns=1), "turn_limit"), (SimulationConfig(max_actions=1), "action_limit")],
)
def test_limits_do_not_invent_winners(config, reason):
    _, result = run_game(config, seed=42)
    assert result.winner is None
    assert result.termination == reason
    assert result.actions <= config.max_actions
    assert result.turns <= config.max_turns
    assert summarize([result])["truncated_games"] == 1


def test_seed_is_reproducible_across_hash_seeds(tmp_path):
    documents = []
    for hash_seed in ("1", "9876"):
        output = tmp_path / hash_seed
        subprocess.run(
            [sys.executable, "-m", "catanzero", "--seed", "42", "--save-games", "--output", str(output)],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            check=True,
            capture_output=True,
            text=True,
        )
        doc = json.loads((output / "game_0000.json").read_text(encoding="utf-8"))
        doc["game"].pop("id")  # UUID identifies a run, not its gameplay.
        documents.append(doc)
    assert documents[0] == documents[1]


def test_saved_state_continues_identically():
    original, _ = run_game(SimulationConfig(max_actions=50), seed=12)
    document = json.loads(json.dumps(state_to_json(original)))
    restored = state_from_json(
        document, [SimulationRandomPlayer(Color[color]) for color in document["colors"]]
    )
    for _ in range(30):
        assert original.play_tick() == restored.play_tick()
    assert state_to_json(original) == state_to_json(restored)


def test_cli_writes_batch_and_refuses_overwrite(tmp_path):
    output = tmp_path / "batch"
    assert main(["--games", "3", "--seed", "5", "--max-actions", "1", "--output", str(output)]) == 0
    results = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert [row["seed"] for row in results] == [5, 6, 7]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["games"] == summary["truncated_games"] == 3
    assert sum(summary["wins"].values()) == 0
    before = (output / "summary.json").read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["--output", str(output)])
    assert error.value.code == 2
    assert (output / "summary.json").read_bytes() == before


@pytest.mark.parametrize("players", [2, 3])
def test_player_counts(players):
    _, result = run_game(SimulationConfig(players=players, max_actions=1), seed=1)
    assert len(result.seating_order) == players


@pytest.mark.parametrize("args", [["--games", "0"], ["--players", "5"], ["--max-turns", "-1"]])
def test_cli_rejects_invalid_settings(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
