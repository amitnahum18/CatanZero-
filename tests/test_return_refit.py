from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from refit_linear_returns import complete_returns


def trajectory():
    return [{"game_index": 7, "actor": "RED", "action_index": i*4,
             "next_action_index": (i+1)*4, "terminal": i == 2,
             "discount": 0. if i == 2 else .5, "reward": r,
             "selected_index": 0, "next_selected_index": None if i == 2 else 0,
             "x": [float(i)], "next_x": [float(i+1)]}
            for i, r in enumerate((1., 2., 3.))]


def test_returns_use_entire_actual_trajectory_and_terminal_zero():
    rows, returns = complete_returns(list(reversed(trajectory())))
    np.testing.assert_array_equal(returns, [2.75, 3.5, 3.])
    assert [r["action_index"] for r in rows] == [0, 4, 8]


def test_incomplete_or_discontinuous_games_cannot_get_labels():
    with pytest.raises(ValueError, match="terminal outcome"):
        complete_returns(trajectory()[:-1])
    rows = trajectory()
    rows[0]["next_x"] = [99.]
    with pytest.raises(ValueError, match="discontinuity"):
        complete_returns(rows)
    rows = trajectory()
    rows[-1]["discount"] = .5
    with pytest.raises(ValueError, match="zero continuation"):
        complete_returns(rows)
