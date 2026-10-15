import gzip
import json

from catanzero.online_env import OnlineEnvironment
from catanzero.viewer import replay_data
from pathlib import Path


def test_persistent_worker_returns_only_same_player_decisions_and_censors_caps(tmp_path):
    env = OnlineEnvironment({"base_seed": 1400000, "max_actions": 28})
    response = env.reset(0, tmp_path / "first")
    actor = response["learner_color"]
    actions = []
    while response["kind"] == "decision":
        assert response["learner_color"] == actor
        assert all(len(x) == 164 for x in response["features"])
        assert "game" not in response and "random_state" not in response
        actions.append(response["action_index"])
        response = env.step(0, 0, [0]*len(response["features"]), True)
    assert response["termination"] == "action_limit" and response["learner_won"] is None
    assert response["guarded_decisions"] == response["actions"] == 28
    assert response["learner_policy"] == "Linear_SARSA"
    assert actions == sorted(set(actions))
    saved = json.loads((tmp_path / "first/game.json").read_text())
    assert len(replay_data(saved)["frames"]) == 29
    with gzip.open(tmp_path / "first/decisions.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 28 and all(r["forward_simulation_guard"] for r in rows)
    assert sum(r["policy"] == "Linear_SARSA" for r in rows) == response["learner_decisions"]
    # Reuse the same worker, preserving no previous board or outcome.
    response = env.reset(1, tmp_path / "second")
    assert response["game_index"] == 1
    while response["kind"] == "decision":
        response = env.step(0, 1, [0]*len(response["features"]))
    assert env.stream is None


def test_three_frozen_linear_opponents_use_public_vectors_without_search(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    reference = repo / "runs/online-sarsa-low-exploration-500-v3"
    env = OnlineEnvironment({"base_seed": 4800000, "max_actions": 40, "learner_policy": "GBRL_SARSA",
                             "opponent_policy": "Linear_SARSA", "opponent_model": str(reference / "linear_model.json"),
                             "opponent_normalizer": str(reference / "normalizer.json")})
    response = env.reset(0, tmp_path / "game")
    while response["kind"] == "decision":
        response = env.step(0, 0, [0]*len(response["features"]), True)
    assert response["opponent_policy"] == "Linear_SARSA" and response["forbidden_engine_calls"] == 0
    rows = [json.loads(line) for line in gzip.open(tmp_path / "game/decisions.jsonl.gz", "rt")]
    opponents = [r for r in rows if r["policy"] == "Linear_SARSA"]
    assert opponents and all(not r["explored"] and r["model_version"] is None for r in opponents)
    assert all(r["policy"] in ("GBRL_SARSA", "Linear_SARSA") for r in rows)


def test_ab2_opponent_search_uses_detached_rng_and_replays_legal_actions(tmp_path):
    env = OnlineEnvironment({"base_seed": 76900000, "players": 2, "max_actions": 12,
                             "opponent_policy": "AB:2"})
    response = env.reset(1, tmp_path / "game")
    while response["kind"] == "decision":
        response = env.step(0, 402, [0] * len(response["features"]))
    assert response["termination"] == "action_limit"
    assert response["players"] == 2 and response["opponent_policy"] == "AB:2"
    assert response["native_search_decisions"] > 0
    rows = [json.loads(line) for line in gzip.open(tmp_path / "game/decisions.jsonl.gz", "rt")]
    assert len(rows) == 12
    assert sum(row["policy"] == "AB:2" for row in rows) == response["native_search_decisions"]
    assert all(row["inputs"]["action"] in row["inputs"]["legal_actions"] for row in rows)
