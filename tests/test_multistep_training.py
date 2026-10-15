from pathlib import Path
import sys
import json
import pytest
pytest.importorskip("gbrl")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from train_gbrl_mixed import MultiStepSARSA
from train_online_sarsa import play_phase


def metadata(game, index, terminal=False):
    return {"game_index": game, "actor": "RED", "phase": "live_training", "action_index": index*4,
            "next_action_index": (index+1)*4, "terminal": terminal, "selected_index": 0,
            "next_selected_index": None if terminal else 0, "action_model_version": 0,
            "next_action_model_version": None if terminal else 0}


def test_n_step_returns_terminal_tails_and_unknown_cutoffs(tmp_path):
    norm = {"offset": [0.]*164, "scale": [1.]*164}
    model = MultiStepSARSA(tmp_path, norm, n_steps=3)
    x = np.zeros(164, dtype=np.float32)
    for index, reward in enumerate((1., 2., 3.)):
        model.add(x, x, reward, .5, metadata(0, index))
    assert len(model.queue) == 1
    assert model.queue[0][2] == 2.75 and model.queue[0][3] == .125
    model.add(x, x, 4., 0., metadata(0, 3, terminal=True))
    model.finish_game(0, censored=False)
    assert [s[2] for s in model.queue] == [2.75, 4.5, 5., 4.]
    assert [s[3] for s in model.queue] == [.125, 0., 0., 0.]
    model.add(x, x, 1., .5, metadata(1, 0))
    model.add(x, x, 2., .5, metadata(1, 1))
    model.finish_game(1, censored=True)
    assert model.queue[-2][2:4] == (2., .25)
    assert model.queue[-1][2:4] == (2., .5)
    assert not model.queue[-1][4]["terminal"] and model.queue[-1][4]["censored_short_return"]
    assert model.raw_transitions == len(model.queue) == 6 and not model.buffers
    model.trace.close()


def test_mixed_real_games_preserve_seats_and_bootstrap_caps(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    reference = repo / "runs/online-sarsa-low-exploration-500-v3"
    norm = json.loads((reference / "normalizer.json").read_text())
    model = MultiStepSARSA(tmp_path, norm)
    opponent = {"opponent_policy": "Mixed_linear_F_static", "opponent_model": str(reference / "linear_model.json"),
                "opponent_normalizer": str(reference / "normalizer.json"), "max_actions": 300,
                "trace_compresslevel": 1}
    results, summary = play_phase(tmp_path, "train", model, 8, 4, training=True,
                                  base_seed=14000000, excluded=[], learner_policy="GBRL_SARSA",
                                  opponent_config=opponent, gamma=.9999,
                                  exploration_start=.1, exploration_end=.05)
    assert len(results) == 8 and model.version > 0
    assert model.transitions == model.raw_transitions and not model.buffers
    assert sum(r["opponent_policy"] == "F_static" for r in results) == 4
    assert sum(r["opponent_policy"] == "Linear_SARSA" for r in results) == 4
    assert summary["forbidden_engine_calls"] == 0
    assert all(r["learner_won"] is None for r in results if r["termination"] != "victory")
    model.trace.close()
