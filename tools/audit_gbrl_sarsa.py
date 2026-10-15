"""Recompute tree SARSA updates and check predictions on replayed real states."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from gbrl.models import GBTModel
from train_gbrl_sarsa import new_tree_model, tree_predict, board_hashes, REPO
from catanzero.online_linear import shaped_transition


def prefix_predict(model, x, version):
    if version == 0:
        return np.zeros(len(x), dtype=np.float32)
    return model(np.ascontiguousarray(x, dtype=np.float32), requires_grad=False,
                 stop_idx=version, tensor=False).reshape(-1)


def audit(root):
    read = lambda name: json.loads((root / name).read_text())
    config, report, normalizer = read("config.json"), read("report.json"), read("normalizer.json")
    assert config["gbrl_version"] == "1.1.9" and config["updater_processes"] == (0 if config.get("evaluation_only") else 1)
    for path, expected in config["source_sha256"].items():
        archived = root / "implementation_snapshot" / path
        assert hashlib.sha256((archived if archived.exists() else REPO / path).read_bytes()).hexdigest() == expected
    reference = Path(config["reference"])
    assert hashlib.sha256((reference / "model.joblib").read_bytes()).hexdigest() == config["reference_sha256"]
    assert json.loads((reference / "normalizer.json").read_text()) == normalizer and normalizer["fit_split"] == "train"
    final = GBTModel.load_learner(str(root / "model"), device="cpu")
    curriculum = config.get("curriculum", False)
    base_count = config.get("initial_tree_count", 0)
    if curriculum:
        source = Path(config["resume_from"])
        assert hashlib.sha256((source / "model.gbrl_model").read_bytes()).hexdigest() == config["resume_tree_sha256"]
        rebuilt = GBTModel.load_learner(str(source / "model"), device="cpu")
        assert rebuilt.get_iteration() == base_count
    else:
        rebuilt = new_tree_model()
    seen, transitions, changed, reward_checks = set(), 0, 0, 0
    cases = [json.loads(line) for line in gzip.open(root / "tree_prediction_cases.jsonl.gz", "rt", encoding="utf-8")]
    decisions = {(c["game_index"], c["action_index"]): c for c in cases if c["phase"] == "live_training"}
    phases = tuple(p for p in ("initial_evaluation", "live_training", "final_evaluation", "confirmation") if (root / p).exists()) if curriculum else ("linear_evaluation", "live_training", "tree_evaluation")
    results = {phase: [json.loads(line) for line in (root / phase / "results.jsonl").read_text().splitlines()] for phase in phases}
    train_results = {r["game_index"]: r for r in results.get("live_training", [])}
    version = 0
    with gzip.open(root / "updates.jsonl.gz", "rt", encoding="utf-8") as stream:
        for version, line in enumerate(stream, 1):
            row = json.loads(line)
            assert row["version"] == version
            samples = row["samples"]
            x = np.asarray([s["x"] for s in samples], dtype=np.float32)
            following = np.asarray([s["next_x"] for s in samples], dtype=np.float32)
            assert np.isfinite(x).all() and np.max(np.abs(x)) <= 4 and np.max(np.abs(following)) <= 4
            targets = np.asarray([s["reward"] for s in samples])+np.asarray([s["discount"] for s in samples])*tree_predict(rebuilt, following)
            assert np.allclose(targets, [s["target"] for s in samples], atol=2e-6, rtol=0)
            previous = tree_predict(rebuilt, x)
            gradients = np.clip(previous-targets, -1, 1).astype(np.float32).reshape(-1, 1)
            assert np.allclose(gradients[:, 0], [s["gradient"] for s in samples], atol=2e-6, rtol=0)
            rebuilt.step(x, gradients)
            after = tree_predict(rebuilt, x)
            assert np.allclose(after, prefix_predict(final, x, base_count+version), atol=3e-6, rtol=0)
            changed += bool(np.linalg.norm(after-previous) > 0)
            for s in samples:
                key = (s["game_index"], s["action_index"])
                assert key not in seen and s["phase"] == "live_training"
                seen.add(key)
                assert s["actor"] == train_results[s["game_index"]]["learner_color"]
                assert s["next_action_index"] > s["action_index"]
                assert 0 <= s["action_model_version"] < version
                assert s["terminal"] == (s["discount"] == 0)
                if not s["terminal"]:
                    assert s["next_action_model_version"] == version-1 and s["next_selected_index"] is not None
                    assert abs(s["discount"]-.999**(s["next_action_index"]-s["action_index"])) < 1e-12
                if key in decisions:
                    prev = decisions[key]
                    following_state = train_results[s["game_index"]] if s["terminal"] else decisions[(s["game_index"], s["next_action_index"])]
                    assert s["selected_index"] == prev["selected_index"] and s["action_model_version"] == prev["model_version"]
                    if not s["terminal"]:
                        assert s["next_selected_index"] == following_state["selected_index"]
                    r, d = shaped_transition(prev, following_state, terminal=s["terminal"], won=train_results[s["game_index"]]["learner_won"])
                    assert abs(r-s["reward"]) < 1e-12 and d == s["discount"]
                    reward_checks += 1
            transitions += len(samples)
    assert version == report["central_updates"] and final.get_iteration() == base_count+version
    assert transitions == len(seen) == report["fresh_training_transitions"]
    assert changed == report["updates_with_prediction_change"]
    scores_checked, frozen = 0, report["central_updates"]
    for case in cases:
        x = np.clip((np.asarray(case["features"])-normalizer["offset"])/normalizer["scale"], -4, 4)
        expected = prefix_predict(final, x, base_count+case["model_version"])
        assert np.allclose(expected, case["scores"], atol=3e-6, rtol=0)
        if case["phase"] in ("tree_evaluation", "final_evaluation", "confirmation"):
            assert case["model_version"] == frozen and not case["explored"]
        if not case["explored"]:
            assert abs(expected[case["selected_index"]]-max(expected)) < 3e-6
        scores_checked += len(x)
    boards = {phase: {r["initial_map_hash"] for r in rows} for phase, rows in results.items()}
    evaluation_phase = "final_evaluation" if curriculum else "tree_evaluation"
    if not curriculum:
        assert boards["linear_evaluation"] == boards["tree_evaluation"]
    assert not boards.get("live_training", set()).intersection(boards[evaluation_phase])
    ancestor = reference
    while ancestor:
        assert not (boards.get("live_training", set()) | boards[evaluation_phase]).intersection(board_hashes(ancestor, ("initial_evaluation", "live_training", "final_evaluation")))
        parent = json.loads((ancestor / "config.json").read_text()).get("resume_from")
        ancestor = Path(parent) if parent else None
    for phase, rows in results.items():
        assert len({r["game_index"] for r in rows}) == len(rows)
        assert len(set(sum(r["learner_seat"] == i for r in rows) for i in range(4))) == 1
        assert all(r["forbidden_engine_calls"] == 0 for r in rows)
        summary = report[phase]["summary"] if phase == "confirmation" else report[phase]
        assert summary["learner_wins"] == sum(r["learner_won"] is True for r in rows)
    verification = {"passed": True, "updates_recomputed": version, "fresh_transitions_verified": transitions,
                    "transitions_used_once": True, "same_player_SARSA_targets_verified": True,
                    "sampled_real_rewards_verified": reward_checks, "tree_candidate_predictions_verified": scores_checked,
                    "zero_tree_initialization_verified": not curriculum, "saved_tree_continuation_verified": curriculum,
                    "train_eval_and_prior_boards_disjoint": True,
                    "balanced_seats": True, "no_forward_simulation": True, "nonzero_updates": changed}
    (root / "update_audit.json").write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    audit(parser.parse_args().study.resolve())
