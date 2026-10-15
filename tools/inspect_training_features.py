"""Training-only structural diagnostics, never selecting on game-test wins."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    norm = json.loads((args.source / "normalizer.json").read_text())
    names = norm["feature_names"]
    vectors = []
    games = set()
    with gzip.open(args.source / "updates.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            for sample in json.loads(line)["samples"]:
                assert sample["phase"] == "live_training"
                vectors.append(sample["x"])
                games.add(sample["game_index"])
    x = np.asarray(vectors, dtype=np.float64)
    assert x.shape[1] == len(names) and np.isfinite(x).all()
    std = x.std(axis=0)
    constant = np.flatnonzero(std < 1e-12)
    active = np.flatnonzero(std >= 1e-12)
    correlations = np.corrcoef(x[:, active], rowvar=False)
    pairs = []
    for i, j in zip(*np.where(np.triu(np.abs(correlations) >= .995, 1))):
        pairs.append({"first": names[active[i]], "second": names[active[j]], "correlation": float(correlations[i,j])})
    pairs.sort(key=lambda p: -abs(p["correlation"]))
    duplicate = []
    hashes = {}
    for index in active:
        digest = hashlib.sha256(np.ascontiguousarray(x[:, index]).tobytes()).hexdigest()
        if digest in hashes:
            duplicate.append({"first": names[hashes[digest]], "second": names[index]})
        else:
            hashes[digest] = index
    coefficients = np.asarray(json.loads((args.source / "linear_model.json").read_text())["weights"])
    contributions = std*np.abs(coefficients)
    order = np.argsort(contributions)
    report = {"source": str(args.source.resolve()), "training_games": len(games), "training_rows": len(x),
              "input_columns": len(names), "constant_feature_count": len(constant),
              "constant_features": [names[i] for i in constant],
              "exact_duplicate_columns_in_training": duplicate,
              "high_correlation_pair_count": len(pairs), "high_correlation_pairs": pairs,
              "lowest_observed_contribution": [{"name": names[i], "coefficient": float(coefficients[i]),
                    "training_std": float(std[i]), "abs_coefficient_times_std": float(contributions[i])} for i in order[:30]],
              "limitations": ["constant/duplicate/correlated in these training samples does not imply harmful gameplay",
                              "coefficient signs are not feature quality", "state-only columns can still support SARSA value learning",
                              "no features removed; independent ablation and retraining needed before pruning",
                              "all diagnostics use training data only, not the prior 200-game comparison"]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("training_games","training_rows","input_columns","constant_feature_count","high_correlation_pair_count")}))
    print(json.dumps({"constant_features":report["constant_features"], "first_correlated_pairs":pairs[:8], "duplicate_count":len(duplicate)}))


if __name__ == "__main__":
    main()
