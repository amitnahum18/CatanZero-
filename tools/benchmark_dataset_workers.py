"""Benchmark worker counts sequentially on identical games and verify the data."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def benchmark(output, games, seed, worker_counts):
    output.mkdir(parents=True, exist_ok=False)
    measurements = []
    reference_hashes = None
    reference_results = None
    for workers in worker_counts:
        directory = output / f"workers-{workers}"
        command = [sys.executable, "-m", "catanzero.dataset", "--games", str(games),
                   "--seed", str(seed), "--workers", str(workers), "--output", str(directory),
                   "--save-games"]
        started = perf_counter()
        with (output / f"workers-{workers}.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        wall_seconds = perf_counter() - started
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        hashes = []
        for index in range(games):
            with gzip.open(directory / "games" / f"game_{index:04d}.moves.jsonl.gz", "rb") as stream:
                hashes.append(hashlib.sha256(stream.read()).hexdigest())
        results = [json.loads(line) for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        for result in results:
            result.pop("elapsed_seconds")
            result.pop("label_seconds")
        if reference_hashes is None:
            reference_hashes, reference_results = hashes, results
        elif hashes != reference_hashes or results != reference_results:
            raise RuntimeError(f"Worker count {workers} changed gameplay or labels")
        measurement = {
            "workers": summary["workers"], "wall_seconds": wall_seconds,
            "collector_seconds": summary["elapsed_seconds"], "games": summary["games"],
            "completed_games": summary["completed_games"], "eligible_rows": summary["eligible_rows"],
            "eligible_rows_per_second": summary["eligible_rows"] / wall_seconds,
            "same_gameplay_and_labels": True,
        }
        measurements.append(measurement)
        print(json.dumps(measurement), flush=True)
    for measurement in measurements:
        measurement["speedup_vs_first"] = measurements[0]["wall_seconds"] / measurement["wall_seconds"]
    report = {"games": games, "base_seed": seed, "save_games": True,
              "measurements": measurements,
              "note": "Timed sequentially on identical seeds; label quality is not assessed."}
    (output / "benchmark_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    args = parser.parse_args()
    if args.games < 1 or not args.workers or any(n < 1 for n in args.workers) or len(set(args.workers)) != len(args.workers):
        parser.error("positive games and distinct positive worker counts required")
    benchmark(args.output, args.games, args.seed, args.workers)
