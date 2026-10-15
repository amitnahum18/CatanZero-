"""Export numeric X/y with game splits and train-only standardization."""

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from math import sqrt, isclose
from pathlib import Path
from time import perf_counter

from catanzero.cli import write_json
from catanzero.features import FeatureEncoder, feature_schema, VERSION, STRATEGIC_VERSION, BUILD_VERSION


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fit_normalizer(vectors, schema):
    """Welford statistics. Caller must supply only train vectors."""
    count = 0
    means = [0.0] * schema["input_columns"]
    m2 = means.copy()
    for vector in vectors:
        count += 1
        for i, value in enumerate(vector):
            difference = value - means[i]
            means[i] += difference / count
            m2[i] += difference * (value - means[i])
    if count == 0:
        raise ValueError("no training vectors")
    stds = [sqrt(max(value / count, 0.0)) for value in m2]
    constant = [schema["feature_names"][i] for i, std in enumerate(stds) if std < 1e-12]
    binary = set(schema["binary_features"])
    offsets, scales = [], []
    for name, mean, std in zip(schema["feature_names"], means, stds):
        offsets.append(0.0 if name in binary else mean)
        scales.append(1.0 if name in binary or std < 1e-12 else std)
    return {"version": "train_only_zscore_v1", "fit_split": "train", "fit_rows": count,
            "feature_version": schema["version"], "feature_names": schema["feature_names"],
            "offset": offsets, "scale": scales, "train_mean": means, "train_std": stds,
            "constant_train_features": constant,
            "binary_handling": "preserve one-hot and binary flags as 0/1",
            "constant_handling": "scale=1; fixed schema; unseen nonconstant validation/test values preserved"}


def normalize(vector, normalizer):
    return [(value - offset) / scale for value, offset, scale in
            zip(vector, normalizer["offset"], normalizer["scale"])]


def privacy_invariance_check(encoder, row):
    """Falsification check: labels/provenance/unknown private fields don't affect X."""
    original = encoder.encode(row["inputs"])
    changed = deepcopy(row)
    changed["targets"] = {"weighted_score": 0.987654321, "terminal_outcome": 1}
    changed["heuristic"] = {"best_value": 12345, "heuristic_score": 0.0123456}
    changed.update(actor_bot="UNKNOWN", game_index=98765, action_index=54321, split="test",
                   winner="UNKNOWN", seed=999, random_state="future")
    inputs = changed["inputs"]
    inputs["map_ref"] = "another-local-id"
    inputs.update(winner="UNKNOWN", seed=888, future_roll=[6, 6])
    obs = inputs["observation"]
    obs.update(winner="UNKNOWN", random_state="future", development_listdeck=["VICTORY_POINT"] * 100)
    for player in obs["players"]:
        if player["color"] != obs["actor"]:
            player.update(resources={"ORE": 999}, development_cards={"VICTORY_POINT": 999},
                          actual_vp=999, playable_at_turn_start={"KNIGHT": True})
    assert encoder.encode(inputs) == original
    return True


def disjoint_feature_splits(rows):
    """Exclude identical compressed vectors from later holdouts, using X only.

    Original game assignments and training rows remain unchanged. Collisions
    need not indicate duplicate raw states; this compact encoding is lossy.
    """
    prior = set()
    retained, excluded = [], []
    for split in ("train", "validation", "test"):
        current = set()
        for row in rows:
            if row["split"] != split:
                continue
            digest = canonical_hash(row["x"])
            if digest in prior:
                excluded.append({**row["metadata"], "reason": "same compressed X in an earlier split",
                                 "feature_hash": digest})
            else:
                retained.append(row)
                current.add(digest)
        prior.update(current)
    return retained, excluded


def build(directories, output, feature_version=VERSION):
    started = perf_counter()
    schema = feature_schema(feature_version)
    rows = []
    game_splits, map_splits = {}, {}
    duplicate_ids, duplicate_inputs = set(), {}
    invariance_checked = Counter()
    source_notes = []
    for root in directories:
        if (root / "FAILED.json").exists():
            raise ValueError(f"failed source dataset: {root}")
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        maps = {m["map_ref"]: m["map"] for m in
                map(json.loads, (root / "maps.jsonl").read_text(encoding="utf-8").splitlines())}
        encoders = {key: FeatureEncoder(document, version=feature_version) for key, document in maps.items()}
        map_hashes = {key: canonical_hash(doc) for key, doc in maps.items()}
        source_notes.append({"source": str(root.resolve()), "heuristic_version": config["heuristic_version"]})
        results = [json.loads(line) for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        for result in results:
            split = result["split"]
            if split not in ("train", "validation", "test"):
                raise ValueError("unknown split")
            game_id = (str(root.resolve()), result["game_index"])
            assert game_splits.setdefault(game_id, split) == split
            with gzip.open(root / result["moves_file"], "rt", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if not row["train_eligible"]:
                        continue
                    if not row["has_choice"] or result["winner"] is None:
                        raise ValueError("forced/censored row marked train eligible")
                    identity = (*game_id, row["action_index"])
                    if identity in duplicate_ids:
                        raise ValueError("duplicate source decision")
                    duplicate_ids.add(identity)
                    target = row["targets"]["weighted_score"]
                    if not isinstance(target, (float, int)) or not 0 <= target <= 1:
                        raise ValueError("invalid target")
                    inputs = row["inputs"]
                    map_ref = inputs["map_ref"]
                    board_hash = map_hashes[map_ref]
                    previous_split = map_splits.setdefault(board_hash, split)
                    if previous_split != split:
                        raise ValueError("same initial board crosses dataset splits")
                    obs = inputs["observation"]
                    if any(set(p).intersection(("resources", "development_cards", "actual_vp", "playable_at_turn_start"))
                           for p in obs["players"] if p["color"] != obs["actor"]):
                        raise ValueError("private opponent information present in source inputs")
                    if set(obs).intersection(("winner", "seed", "random_state", "development_listdeck", "targets")):
                        raise ValueError("future/provenance fields present in source observation")
                    input_hash = canonical_hash({"map": board_hash,
                                                "observation": obs, "action": inputs["action"],
                                                "legal_actions": inputs["legal_actions"]})
                    if input_hash in duplicate_inputs:
                        raise ValueError("duplicate state/action input")
                    duplicate_inputs[input_hash] = split
                    encoder = encoders[map_ref]
                    vector = encoder.encode(inputs)
                    for player in obs["players"]:
                        if player["color"] == obs["actor"]:
                            assert "resources" in player and "development_cards" in player
                    # Every represented action type in each source gets an
                    # invariance test, without wasting full deep copies per row.
                    group = (root.name, inputs["action"][1])
                    if not invariance_checked[group]:
                        privacy_invariance_check(encoder, row)
                        invariance_checked[group] += 1
                    rows.append({"split": split, "x": vector, "y": float(target),
                                 "metadata": {"source": root.name, "game_index": row["game_index"],
                                              "action_index": row["action_index"], "actor_bot": row["actor_bot"],
                                              "split": split, "initial_board_hash": board_hash}})
    source_rows = len(rows)
    rows, excluded = disjoint_feature_splits(rows)
    normalizer = fit_normalizer((r["x"] for r in rows if r["split"] == "train"), schema)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "feature_schema.json", schema)
    write_json(output / "normalizer.json", normalizer)
    write_json(output / "excluded_feature_collisions.json", excluded)
    vectors_by_split = {split: set() for split in ("train", "validation", "test")}
    norm_means, norm_m2, train_count = [0.0] * schema["input_columns"], [0.0] * schema["input_columns"], 0
    for split in vectors_by_split:
        with gzip.open(output / f"{split}.jsonl.gz", "wt", encoding="utf-8") as stream, \
                (output / f"{split}.metadata.jsonl").open("x", encoding="utf-8") as meta:
            for row in rows:
                if row["split"] != split:
                    continue
                x = normalize(row["x"], normalizer)
                vectors_by_split[split].add(canonical_hash(x))
                stream.write(json.dumps({"x": x, "y": row["y"]}, allow_nan=False) + "\n")
                meta.write(json.dumps(row["metadata"]) + "\n")
                if split == "train":
                    train_count += 1
                    for i, value in enumerate(x):
                        difference = value - norm_means[i]
                        norm_means[i] += difference / train_count
                        norm_m2[i] += difference * (value - norm_means[i])
    cross_split_feature_collisions = {
        f"{a}:{b}": len(vectors_by_split[a] & vectors_by_split[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    binary = set(schema["binary_features"])
    checked_continuous = 0
    for i, name in enumerate(schema["feature_names"]):
        if name not in binary and name not in normalizer["constant_train_features"]:
            assert abs(norm_means[i]) < 1e-9 and isclose(norm_m2[i] / train_count, 1, abs_tol=1e-9)
            checked_continuous += 1
    assert not any(cross_split_feature_collisions.values())
    report = {"sources": source_notes, "source_rows": source_rows, "rows": len(rows),
              "holdout_rows_excluded_for_compressed_feature_collisions": len(excluded),
              "input_columns": schema["input_columns"], "output_columns": 1,
              "target": "weighted_score", "split_rows": dict(Counter(r["split"] for r in rows)),
              "split_games": dict(Counter(game_splits.values())),
              "by_bot": dict(Counter(r["metadata"]["actor_bot"] for r in rows)),
              "normalizer_fit_split": "train", "normalizer_fit_rows": train_count,
              "continuous_columns_verified_mean_0_variance_1": checked_continuous,
              "binary_columns_preserved_0_1": len(binary),
              "privacy_invariance_checks": sum(invariance_checked.values()),
              "source_private_opponent_information_rejected": True,
              "source_future_observation_fields_rejected": True,
              "sampled_roll_and_draw_outcomes_rejected": True,
              "game_and_initial_board_splits_disjoint": True,
              "duplicate_source_rows": 0, "duplicate_state_action_inputs": 0,
              "cross_split_feature_collisions": cross_split_feature_collisions,
              "spatial_encoding": "compressed summaries; distinct boards/states can map to same features",
              "limitations": ["weak heuristic/outcome labels, not calibrated confidence",
                              "v2 single-belief and v3 eight-belief label sources retained",
                              "linear baseline; spatial representation is not lossless",
                              "no player-to-player trade negotiation"],
              "elapsed_seconds": perf_counter() - started}
    write_json(output / "data_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature-version", choices=("v1", "v2", "v3"), default="v1")
    args = parser.parse_args(argv)
    build(args.sources, args.output, {"v1": VERSION, "v2": STRATEGIC_VERSION, "v3": BUILD_VERSION}[args.feature_version])


if __name__ == "__main__":
    main()
