"""Recompute every online target and incremental update from the zero model."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import SGDRegressor


def audit(root):
    config = json.loads((root / "config.json").read_text())
    normalizer = json.loads((root / "normalizer.json").read_text())
    report = json.loads((root / "report.json").read_text())
    if config.get("resume_from"):
        source = Path(config["resume_from"])
        assert hashlib.sha256((source / "model.joblib").read_bytes()).hexdigest() == config["resume_estimator_sha256"]
        estimator = joblib.load(source / "model.joblib")
        assert json.loads((source / "normalizer.json").read_text()) == normalizer
        assert float(estimator.t_) == config["initial_optimizer_t"]
    else:
        estimator = SGDRegressor(**config["estimator_parameters"])
        estimator.partial_fit(np.zeros((1, 164)), np.zeros(1))
        assert np.count_nonzero(estimator.coef_) == 0 and estimator.intercept_[0] == 0
    initial = json.loads((root / "snapshots/version_000000.json").read_text())
    assert np.array_equal(estimator.coef_, initial["weights"])
    assert estimator.intercept_[0] == initial["intercept"]
    seen, transitions, nonzero = set(), 0, 0
    by_game, first_last = {}, {}
    with gzip.open(root / "updates.jsonl.gz", "rt", encoding="utf-8") as stream:
        for expected_version, line in enumerate(stream, 1):
            update = json.loads(line)
            assert update["version"] == expected_version
            x = np.asarray([s["x"] for s in update["samples"]])
            following = np.asarray([s["next_x"] for s in update["samples"]])
            rewards = np.asarray([s["reward"] for s in update["samples"]])
            discounts = np.asarray([s["discount"] for s in update["samples"]])
            assert x.shape[1] == 164 and np.max(np.abs(x)) <= 4 and np.max(np.abs(following)) <= 4
            targets = rewards + discounts*estimator.predict(following)
            assert np.allclose(targets, [s["target"] for s in update["samples"]], atol=1e-12, rtol=0)
            previous = np.r_[estimator.coef_, estimator.intercept_].copy()
            estimator.partial_fit(x, targets, sample_weight=1/(1+np.sum(x*x, axis=1)))
            snapshot = json.loads((root / "snapshots" / f"version_{expected_version:06d}.json").read_text())
            assert np.allclose(estimator.coef_, snapshot["weights"], atol=1e-12, rtol=0)
            assert abs(estimator.intercept_[0]-snapshot["intercept"]) < 1e-12
            nonzero += bool(np.linalg.norm(np.r_[estimator.coef_, estimator.intercept_]-previous) > 0)
            for sample in update["samples"]:
                key = (sample["phase"], sample["game_index"], sample["action_index"])
                assert sample["phase"] == "live_training" and key not in seen
                seen.add(key)
                assert sample["next_action_index"] > sample["action_index"]
                assert 0 <= sample["action_model_version"] < expected_version
                assert sample["terminal"] == (sample["discount"] == 0)
                if not sample["terminal"]:
                    assert sample["next_selected_index"] is not None
                    assert sample["next_action_model_version"] == expected_version-1
                    assert abs(sample["discount"]-.999**(sample["next_action_index"]-sample["action_index"])) < 1e-12
                by_game.setdefault(sample["game_index"], set()).add(sample["actor"])
                first_last.setdefault(sample["game_index"], []).append(sample["action_index"])
            transitions += len(update["samples"])
    assert transitions == report["fresh_training_transitions"] and len(seen) == transitions
    assert expected_version == report["central_updates"] and nonzero == report["updates_with_nonzero_parameter_change"]
    assert all(len(actors) == 1 for actors in by_game.values())
    assert all(actions == sorted(actions) for actions in first_last.values())
    results = {phase: [json.loads(line) for line in (root / phase / "results.jsonl").read_text().splitlines()]
               for phase in ("initial_evaluation", "live_training", "final_evaluation")}
    initial_boards = {r["initial_map_hash"] for r in results["initial_evaluation"]}
    train_boards = {r["initial_map_hash"] for r in results["live_training"]}
    final_boards = {r["initial_map_hash"] for r in results["final_evaluation"]}
    assert not train_boards.intersection(initial_boards) and initial_boards == final_boards
    ancestor = Path(config["resume_from"]) if config.get("resume_from") else None
    while ancestor:
        old_boards = {json.loads(line)["initial_map_hash"] for phase in results
                      for line in (ancestor / phase / "results.jsonl").read_text().splitlines()}
        assert not (train_boards | initial_boards).intersection(old_boards)
        parent = json.loads((ancestor / "config.json").read_text()).get("resume_from")
        ancestor = Path(parent) if parent else None
    for phase, rows in results.items():
        assert len({r["game_index"] for r in rows}) == len(rows)
        counts = [sum(r["learner_seat"] == i for r in rows) for i in range(4)]
        assert len(set(counts)) == 1
        assert all(r["learner_policy"] == "Linear_SARSA" and r["forbidden_engine_calls"] == 0 for r in rows)
        summary = json.loads((root / phase / "summary.json").read_text())
        assert summary["learner_wins"] == sum(r["learner_won"] is True for r in rows)
        assert summary["weights_updated"] == (phase == "live_training")
    for game_id, actors in by_game.items():
        result = next(r for r in results["live_training"] if r["game_index"] == game_id)
        assert actors == {result["learner_color"]}
    final = json.loads((root / "linear_model.json").read_text())
    assert np.allclose(estimator.coef_, final["weights"], atol=1e-12, rtol=0)
    assert normalizer["fit_split"] == "train"
    assert hashlib.sha256(Path(config["normalization_source"]).read_bytes()).hexdigest() == config["normalization_source_sha256"]
    verification = {"passed": True, "updates_recomputed": expected_version, "fresh_transitions_verified": transitions,
                    "transitions_used_once": True, "same_player_transitions_verified": True,
                    "actual_next_action_SARSA_targets_verified": True, "nonzero_updates": nonzero,
                    "train_evaluation_board_disjoint": True, "balanced_seats": True,
                    "zero_initial_weights_verified": not bool(config.get("resume_from")),
                    "saved_starting_weights_and_optimizer_verified": bool(config.get("resume_from")),
                    "ancestor_boards_excluded": bool(config.get("resume_from")),
                    "training_policy_was_linear_model": True}
    (root / "update_audit.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    audit(parser.parse_args().study)
