"""Replay sample games; verify public predecision vectors and frozen snapshots."""

import argparse
import gzip
import json
from math import isclose
from pathlib import Path

from catanatron.models.enums import ActionType
from catanatron.serialization import action_record_from_json, action_to_json, map_to_json, state_to_json

from catanzero.dataset import observation, ordered_actions
from catanzero.features import FeatureEncoder, VERSION
from catanzero.online_linear import LinearActionValues
from catanzero.regression_data import privacy_invariance_check
from catanzero.simulation import SimulationConfig, create_game
from catanzero.static_policies import forbid_forward_simulation


def audit(root):
    normalizer = json.loads((root / "normalizer.json").read_text())
    config = json.loads((root / "config.json").read_text())
    tree_run = "GBRL" in config["algorithm"]
    curriculum = config.get("curriculum", False)
    phases = (tuple(p for p in ("initial_evaluation", "live_training", "final_evaluation", "confirmation") if (root / p).exists())
              if curriculum else (("linear_evaluation", "live_training", "tree_evaluation") if tree_run else ("initial_evaluation", "live_training", "final_evaluation")))
    tree_cases, tree_candidates = 0, 0
    export = gzip.open(root / "tree_prediction_cases.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) if tree_run else None
    if tree_run:
        linear_path = Path(config["reference"]) / "linear_model.json" if curriculum else root / "linear_reference/snapshots/version_000000.json"
        linear = LinearActionValues.load(linear_path, root / "normalizer.json")
    snapshots, checked, predictions, privacy_checks = {}, 0, 0, 0
    counts = {}
    for phase in phases:
        phase_normalizer = normalizer
        phase_snapshot_root = root
        expanded_initial = phase == 'initial_evaluation' and config.get('feature_expansion_from')
        if expanded_initial:
            phase_snapshot_root = Path(config['resume_from'])
            phase_normalizer = json.loads((phase_snapshot_root/'normalizer.json').read_text())
        results = [json.loads(line) for line in (root / phase / "results.jsonl").read_text().splitlines()]
        selected_ids = {0, 1, 2, 3, len(results)-4, len(results)-3, len(results)-2, len(results)-1}
        counts[phase] = 0
        for result in results:
            if result["game_index"] not in selected_ids:
                continue
            directory = root / phase / f"game_{result['game_index']:04d}"
            document = json.loads((directory / "game.json").read_text())
            game = create_game(SimulationConfig(), result["seed"])
            encoder = FeatureEncoder(map_to_json(game.state.board.map), version=phase_normalizer.get('feature_version', VERSION))
            with gzip.open(directory / "decisions.jsonl.gz", "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            assert len(rows) == len(document["action_records"]) == result["actions"]
            strata = set()
            for index, (row, payload) in enumerate(zip(rows, document["action_records"])):
                record = action_record_from_json(payload)
                selected = record.action
                if selected.action_type in (ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD):
                    selected = selected._replace(value=None)
                inputs = row["inputs"]
                assert inputs["observation"] == observation(game, game.state.current_color())
                assert inputs["legal_actions"] == [action_to_json(a) for a in ordered_actions(game)]
                assert inputs["action"] == action_to_json(selected) and row["action_index"] == index
                assert row["forward_simulation_guard"]
                is_learner = selected.color.value == result["learner_color"]
                assert (row["policy"] == result["learner_policy"]) == is_learner
                if is_learner:
                    version = row["model_version"]
                    is_tree = row["policy"] == "GBRL_SARSA"
                    snapshot_key = (phase,version)
                    if not tree_run and snapshot_key not in snapshots:
                        model_path = (phase_snapshot_root/'linear_model.json' if expanded_initial else phase_snapshot_root/'snapshots'/f'version_{version:06d}.json')
                        snapshots[snapshot_key] = LinearActionValues(json.loads(model_path.read_text()), phase_normalizer)
                    with forbid_forward_simulation():
                        features = [encoder.encode({**inputs, "action": a}) for a in inputs["legal_actions"]]
                        expected = row["raw_scores"] if is_tree else [(linear if tree_run else snapshots[snapshot_key]).predict(x) for x in features]
                    assert len(expected) == len(row["raw_scores"])
                    assert all(isclose(a, b, abs_tol=1e-10, rel_tol=1e-10) for a, b in zip(expected, row["raw_scores"]))
                    chosen_index = inputs["legal_actions"].index(inputs["action"])
                    if is_tree:
                        own = next(p for p in inputs["observation"]["players"] if p["color"] == result["learner_color"])
                        export.write(json.dumps({"phase": phase, "game_index": result["game_index"],
                                                 "action_index": index, "model_version": version, "features": features,
                                                 "scores": row["raw_scores"], "selected_index": chosen_index,
                                                 "learner_color": result["learner_color"], "vp": own["actual_vp"],
                                                 "explored": row["explored"]})+"\n")
                        tree_cases += 1
                        tree_candidates += len(features)
                    if not row["explored"]:
                        assert isclose(expected[chosen_index], max(expected), abs_tol=1e-10, rel_tol=1e-10)
                    if phase != "live_training":
                        assert not row["explored"]
                        expected_version = (0 if phase in ("linear_evaluation", "initial_evaluation") else json.loads((root / "report.json").read_text())["central_updates"]) if tree_run else (0 if phase == "initial_evaluation" else json.loads((root / "linear_model.json").read_text())["model_version"])
                        assert version == expected_version
                    if not is_tree:
                        predictions += len(expected)
                    if selected.action_type.value not in strata:
                        assert privacy_invariance_check(encoder, {"inputs": inputs, "targets": {}, "heuristic": {}})
                        privacy_checks += 1
                        strata.add(selected.action_type.value)
                elif curriculum and row["policy"] == "Linear_SARSA":
                    with forbid_forward_simulation():
                        expected = [linear.predict(encoder.encode({**inputs, "action": a})) for a in inputs["legal_actions"]]
                    assert len(expected) == len(row["raw_scores"])
                    assert all(isclose(a, b, abs_tol=1e-10, rel_tol=1e-10) for a, b in zip(expected, row["raw_scores"]))
                    chosen_index = inputs["legal_actions"].index(inputs["action"])
                    assert isclose(expected[chosen_index], max(expected), abs_tol=1e-10, rel_tol=1e-10)
                    assert not row["explored"] and row["model_version"] is None
                    predictions += len(expected)
                game.execute(selected, action_record=record)
                checked += 1
            final = state_to_json(game)
            for field in ("board", "player_state", "num_turns", "action_records", "resource_freqdeck", "development_listdeck"):
                assert final[field] == document[field]
            winner = game.winning_color()
            assert result["winner"] == (winner.value if winner else None)
            counts[phase] += 1
    if export:
        export.close()
    verification = {"passed": True, "games_replayed": counts, "actions_verified": checked,
                    "candidate_predictions_verified": predictions, "privacy_invariance_checks": privacy_checks,
                    "tree_decision_cases_exported_for_model_audit": tree_cases,
                    "tree_candidate_vectors_exported": tree_candidates,
                    "evaluation_snapshots_frozen": True, "no_forward_simulation_during_scoring": True}
    (root / "game_audit.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    audit(parser.parse_args().study)
