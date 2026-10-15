import gzip
import pickle

from catanzero.rl_episode import episode
from catanzero.viewer import replay_data
import json


def test_collector_preserves_actor_and_unknown_capped_outcome(tmp_path):
    config = {"base_seed": 800000, "collect": True, "epsilon": 1.0, "max_actions": 20}
    result = episode(config, 0, tmp_path)
    assert result["termination"] == "action_limit"
    assert result["learner_won"] is None
    assert result["actions"] == result["guarded_decisions"] == 20
    assert result["forbidden_engine_calls"] == 0
    with gzip.open(tmp_path / "experience.pkl.gz", "rb") as stream:
        data = pickle.load(stream)
    assert all(s["actor"] == result["learner_color"] for s in data["states"])
    assert all(len(s["features"]) == s["candidate_count"]*164 for s in data["states"])
    assert all(s["selected_index"] < s["candidate_count"] for s in data["states"])
    assert len(replay_data(json.loads((tmp_path / "game.json").read_text()))["frames"]) == 21


def test_teacher_collection_is_deterministic_across_output_directories(tmp_path):
    config = {"base_seed": 800000, "collect": True, "epsilon": .1, "max_actions": 30}
    first = episode(config, 3, tmp_path / "first")
    other = episode(config, 3, tmp_path / "other")
    for field in ("winner", "termination", "actions", "exploration_selections", "learner_states", "initial_map_hash"):
        assert first[field] == other[field]
    with gzip.open(tmp_path / "first/experience.pkl.gz", "rb") as stream:
        one = pickle.load(stream)
    with gzip.open(tmp_path / "other/experience.pkl.gz", "rb") as stream:
        two = pickle.load(stream)
    assert one["states"] == two["states"] and one["trace"] == two["trace"]
