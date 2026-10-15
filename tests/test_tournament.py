import json
import subprocess

import pytest
from catanatron.serialization import map_to_json

from catanzero.tournament import isolated_match, main, make_match, run_match, summarize_matches
from catanzero.viewer import replay_data


def test_same_board_stream_and_balanced_candidate_seats_across_bots():
    for seat in range(4):
        reference, candidate = make_match("F", 90000 + seat, seat)
        for code in ("AB:2", "M:10"):
            game, other = make_match(code, 90000 + seat, seat)
            assert candidate == other == game.state.colors[seat]
            assert map_to_json(game.state.board.map) == map_to_json(reference.state.board.map)
            assert game.state.development_listdeck == reference.state.development_listdeck
            assert game.state.random.getstate() == reference.state.random.getstate()
            assert all(p.LABEL == "Value Function" for p in game.state.players if p.color != candidate)


def test_full_baseline_game_saves_authoritative_legal_replay(tmp_path):
    result = run_match("F", 90000, 0, 0, tmp_path, 60, 1000, 3000)
    assert result["termination"] == "victory"
    assert result["candidate_won"] == (result["winner"] == result["candidate_color"])
    document = json.loads((tmp_path / "game.json").read_text())
    frames = replay_data(document)["frames"]
    assert frames[-1]["winner"] == result["winner"]
    assert len(frames) - 1 == result["actions"]
    assert all(t["legal_actions"] > 1 for group in result["decision_times"].values() for t in group)


def test_soft_timeout_and_summary_do_not_invent_losses(tmp_path):
    result = run_match("F", 1, 0, 0, tmp_path, 1e-9, 1000, 3000)
    assert result["termination"] == "time_limit"
    assert result["winner"] is result["candidate_won"] is None
    result["process_wall_seconds"] = 0.1
    win = {**result, "termination": "victory", "candidate_won": True}
    loss = {**result, "termination": "victory", "candidate_won": False}
    group = summarize_matches([win, loss, result])["F"]
    assert group["wins"] == group["losses"] == group["unfinished_games"] == 1
    assert group["win_rate_completed"] == 0.5
    assert group["win_fraction_censoring_bounds"] == [1 / 3, 2 / 3]


def test_supervisor_hard_timeout_uses_censored_checkpoint(tmp_path, monkeypatch):
    checkpoint = {"bot": "M:10", "game_index": 0, "seed": 1,
                  "candidate_seat": 0, "actions": 25,
                  "termination": "running", "winner": None, "candidate_won": None,
                  "decision_times": {"candidate": [], "opponents": []}}
    (tmp_path / "checkpoint.json").write_text(json.dumps(checkpoint))
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(subprocess, "run", timeout)
    result = isolated_match(("M:10", 1, 0, 0, tmp_path, 0.1, 1000, 3000))
    assert result["termination"] == "hard_time_limit"
    assert result["actions"] == 25
    assert result["candidate_won"] is None
    assert result["worker_hard_timeout"]


def test_parallel_supervised_batch_and_validation(tmp_path):
    output = tmp_path / "batch"
    assert main(["--bots", "F", "--games", "4", "--max-actions", "2", "--workers", "2", "--output", str(output)]) == 0
    summary = json.loads((output / "summary.json").read_text())
    group = summary["groups"]["F"]
    assert group["games"] == group["unfinished_games"] == 4
    assert group["seat_counts"] == {str(s): 1 for s in range(4)}
    assert group["wins"] == group["losses"] == 0
    invalid = tmp_path / "invalid"
    with pytest.raises(SystemExit):
        main(["--games", "3", "--output", str(invalid)])
    assert not invalid.exists()
