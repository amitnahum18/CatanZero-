from pathlib import Path
import sys
import pytest
pytest.importorskip("gbrl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from train_gbrl_curriculum import victory_gate


def test_switch_requires_advantage_and_complete_evaluation():
    assert victory_gate({"completed_games": 200, "learner_wins": 90, "unfinished_games": 0})["passed"]
    assert not victory_gate({"completed_games": 200, "learner_wins": 50, "unfinished_games": 0})["passed"]
    assert not victory_gate({"completed_games": 199, "learner_wins": 90, "unfinished_games": 1})["passed"]
    assert not victory_gate({"completed_games": 0, "learner_wins": 0, "unfinished_games": 200})["passed"]
