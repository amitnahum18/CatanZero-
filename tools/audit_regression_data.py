"""Verify saved normalized X against independently replayed predecision sources."""

import argparse
from collections import Counter
import gzip
import json
from math import isclose, sqrt
from pathlib import Path
from statistics import fmean

from catanatron.models.enums import ActionType
from catanatron.serialization import action_record_from_json, action_to_json, state_to_json

from catanzero.dataset import observation, ordered_actions
from catanzero.features import FeatureEncoder, feature_schema
from catanzero.regression_data import canonical_hash
from catanzero.simulation import SimulationConfig, create_game


def audit(root):
    schema = json.loads((root / "feature_schema.json").read_text(encoding="utf-8"))
    report = json.loads((root / "data_report.json").read_text(encoding="utf-8"))
    normalizer = json.loads((root / "normalizer.json").read_text(encoding="utf-8"))
    assert schema == feature_schema()
    assert normalizer["fit_split"] == "train"
    assert normalizer["feature_names"] == schema["feature_names"]
    saved = {}
    split_vectors = {}
    for split in ("train", "validation", "test"):
        with gzip.open(root / f"{split}.jsonl.gz", "rt", encoding="utf-8") as stream:
            examples = [json.loads(line) for line in stream]
        metadata = [json.loads(line) for line in (root / f"{split}.metadata.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(examples) == len(metadata) == report["split_rows"][split]
        split_vectors[split] = set()
        for example, meta in zip(examples, metadata):
            assert set(example) == {"x", "y"} and len(example["x"]) == schema["input_columns"]
            assert meta["split"] == split
            identity = (meta["source"], meta["game_index"], meta["action_index"])
            assert identity not in saved
            saved[identity] = (example, meta)
            split_vectors[split].add(canonical_hash(example["x"]))
    assert not (split_vectors["train"] & split_vectors["validation"])
    assert not (split_vectors["train"] & split_vectors["test"])
    assert not (split_vectors["validation"] & split_vectors["test"])
    exclusions = json.loads((root / "excluded_feature_collisions.json").read_text(encoding="utf-8"))
    excluded_ids = {(m["source"], m["game_index"], m["action_index"]) for m in exclusions}
    assert not excluded_ids.intersection(saved)
    raw_train = []
    checked, games, restored_actions = 0, 0, 0
    raw_split_vectors = {s: set() for s in split_vectors}
    for source in report["sources"]:
        directory = Path(source["source"])
        maps = {m["map_ref"]: m["map"] for m in
                map(json.loads, (directory / "maps.jsonl").read_text(encoding="utf-8").splitlines())}
        encoders = {key: FeatureEncoder(document) for key, document in maps.items()}
        results = [json.loads(line) for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        for result in results:
            document = json.loads((directory / "games" / f"game_{result['game_index']:04d}.json").read_text(encoding="utf-8"))
            game = create_game(SimulationConfig(players=4), result["seed"])
            with gzip.open(directory / result["moves_file"], "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            assert len(rows) == len(document["action_records"])
            for row, payload in zip(rows, document["action_records"]):
                record = action_record_from_json(payload)
                selected = record.action
                if selected.action_type in (ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD):
                    selected = selected._replace(value=None)
                before = observation(game, game.state.current_color())
                # This directly proves the source observes the state BEFORE
                # executing the selected move or revealing its recorded result.
                assert row["inputs"]["observation"] == before
                assert row["inputs"]["legal_actions"] == [action_to_json(a) for a in ordered_actions(game)]
                assert row["inputs"]["action"] == action_to_json(selected)
                identity = (directory.name, row["game_index"], row["action_index"])
                if row["train_eligible"]:
                    assert identity in saved or identity in excluded_ids
                else:
                    assert identity not in saved
                if identity in saved:
                    example, meta = saved[identity]
                    vector = encoders[row["inputs"]["map_ref"]].encode(row["inputs"])
                    assert meta["split"] == result["split"]
                    assert example["y"] == row["targets"]["weighted_score"]
                    for i, value in enumerate(vector):
                        expected = (value - normalizer["offset"][i]) / normalizer["scale"][i]
                        assert isclose(example["x"][i], expected, rel_tol=1e-12, abs_tol=1e-12)
                    if meta["split"] == "train":
                        raw_train.append(vector)
                    raw_split_vectors[meta["split"]].add(canonical_hash(vector))
                    checked += 1
                game.execute(selected, action_record=record)
                restored_actions += 1
            final = state_to_json(game)
            for field in ("board", "player_state", "num_turns", "action_records", "resource_freqdeck", "development_listdeck"):
                assert final[field] == document[field]
            games += 1
    assert checked == len(saved) == report["rows"]
    assert len(raw_train) == normalizer["fit_rows"]
    binary = set(schema["binary_features"])
    for i, name in enumerate(schema["feature_names"]):
        values = [v[i] for v in raw_train]
        mean = fmean(values)
        std = sqrt(fmean((v - mean) ** 2 for v in values))
        offset = 0.0 if name in binary else mean
        scale = 1.0 if name in binary or std < 1e-12 else std
        assert isclose(normalizer["offset"][i], offset, rel_tol=1e-10, abs_tol=1e-10)
        assert isclose(normalizer["scale"][i], scale, rel_tol=1e-10, abs_tol=1e-10)
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not raw_split_vectors[a].intersection(raw_split_vectors[b])
    return {"passed": True, "input_columns": schema["input_columns"], "output_columns": 1,
            "verified_predecision_actions": restored_actions, "authoritative_games_verified": games,
            "normalized_rows_verified": checked, "fit_train_rows_verified": len(raw_train),
            "train_only_normalizer_independently_recomputed": True,
            "x_y_separated": True, "targets_match_source": True,
            "cross_split_feature_overlap": 0,
            "holdout_exclusions_verified": len(excluded_ids),
            "note": "Verified this dataset and feature pipeline, not a guarantee about future modified code."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    result = audit(args.dataset)
    with (args.dataset / "leakage_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
