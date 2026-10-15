import json
from math import hypot

import pytest
from catanatron.serialization import state_to_json

from catanzero.simulation import SimulationConfig, run_game
from catanzero.cli import main
from catanzero.viewer import export_replay, replay_data


def test_replay_reaches_saved_winner_and_board():
    game, result = run_game(SimulationConfig(), 42)
    document = json.loads(json.dumps(state_to_json(game)))
    data = replay_data(document)
    assert len(data["frames"]) == result.actions + 1
    assert data["frames"][0]["buildings"] == []
    final = data["frames"][-1]
    assert final["winner"] == result.winner
    assert {p["color"]: p["vp"] for p in final["players"]} == result.victory_points
    assert final["buildings"] == document["board"]["buildings"]
    assert len(data["nodes"]) == 54
    for edge, _ in final["roads"]:
        a, b = (data["nodes"][str(node)] for node in edge)
        assert hypot(a[0] - b[0], a[1] - b[1]) == pytest.approx(64, abs=.001)


def test_partial_replay_export_preserves_source(tmp_path):
    game, _ = run_game(SimulationConfig(players=2, max_actions=25), 3)
    source = tmp_path / "game.json"
    source.write_text(json.dumps(state_to_json(game)), encoding="utf-8")
    before = source.read_bytes()
    output = export_replay(source)
    html = output.read_text(encoding="utf-8")
    assert "__REPLAY_DATA__" not in html
    assert '<html lang="he" dir="rtl">' in html
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match="overwrite"):
        export_replay(source, source)


def test_cli_replay_saves_game_and_html(tmp_path):
    output = tmp_path / "replay-run"
    assert main(["--seed", "42", "--max-actions", "1", "--replay", "--output", str(output)]) == 0
    assert (output / "game_0000.json").exists()
    assert (output / "game_0000.html").exists()
