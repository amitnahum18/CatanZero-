import json

import pytest

from catanzero.scoring import ScoreConfig, main, score_move


def test_actor_outcomes_and_normalized_weights():
    config = ScoreConfig(3, 7)
    winning = score_move(0.8, "BLUE", "BLUE", "victory", config)
    losing = score_move(0.8, "RED", "BLUE", "victory", config)
    assert winning["weighted_score"] == pytest.approx(0.94)
    assert losing["weighted_score"] == pytest.approx(0.24)
    assert winning["scoring"]["outcome_weight"] == pytest.approx(0.7)
    assert score_move(0.8, "BLUE", "BLUE", "victory")["weighted_score"] == pytest.approx(0.9)


@pytest.mark.parametrize("reason", ["turn_limit", "action_limit"])
def test_truncation_is_not_a_loss(reason):
    targets = score_move(0.8, "BLUE", None, reason)
    assert targets["terminal_outcome"] is None
    assert targets["weighted_score"] is None
    assert targets["heuristic_score"] == 0.8
    assert score_move(0.8, "BLUE", None, reason, ScoreConfig(1, 0))["weighted_score"] == 0.8


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_heuristic(value):
    with pytest.raises(ValueError):
        score_move(value, "BLUE", "BLUE", "victory")


@pytest.mark.parametrize("weights", [(0, 0), (-1, 2), (float("nan"), 1), (1, float("inf"))])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        ScoreConfig(*weights)


def test_inconsistent_outcomes_are_rejected():
    with pytest.raises(ValueError):
        score_move(0.8, "BLUE", None, "victory")
    with pytest.raises(ValueError):
        score_move(0.8, "BLUE", "BLUE", "turn_limit")


def test_cli_preserves_inputs_and_separates_targets(tmp_path):
    source = tmp_path / "moves.jsonl"
    destination = tmp_path / "scored.jsonl"
    row = {"actor": "BLUE", "winner": "BLUE", "termination": "victory", "heuristic_score": 0.8,
           "inputs": {"turn": 70, "action": "BUILD_CITY"}}
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert main([str(source), "--output", str(destination)]) == 0
    result = json.loads(destination.read_text(encoding="utf-8"))
    assert result["inputs"] == row["inputs"]
    assert result["targets"]["weighted_score"] == pytest.approx(0.9)
    before = destination.read_bytes()
    with pytest.raises(SystemExit):
        main([str(source), "--output", str(destination)])
    assert destination.read_bytes() == before


def test_invalid_input_leaves_no_output(tmp_path):
    source = tmp_path / "bad.jsonl"
    destination = tmp_path / "output.jsonl"
    source.write_text('{"heuristic_score": 2}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        main([str(source), "--output", str(destination)])
    assert not destination.exists()
