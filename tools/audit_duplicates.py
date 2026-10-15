"""Read-only duplicate audit of authoritative row files and their subset exports."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class Index:
    def __init__(self):
        self.seen = {}
        self.rows = self.duplicates = self.groups = self.cross_split = 0
        self.examples = []

    def add(self, key, location, split):
        self.rows += 1
        if key not in self.seen:
            self.seen[key] = [location, {split}, 1]
            return
        first, splits, count = self.seen[key]
        self.duplicates += 1
        self.groups += count == 1
        self.cross_split += split not in splits
        if len(self.examples) < 10:
            self.examples.append({"first": first, "repeat": location})
        splits.add(split)
        self.seen[key][2] += 1

    def report(self):
        return {"rows": self.rows, "unique": len(self.seen),
                "duplicate_extra_rows": self.duplicates, "duplicate_groups": self.groups,
                "groups_crossing_splits": sum(len(value[1]) > 1 for value in self.seen.values()),
                "examples": self.examples}


def input_key(row, map_hash, ignore_turn=False):
    inputs = row["inputs"]
    obs = dict(inputs["observation"])
    if ignore_turn:
        obs.pop("turn", None)
    # The engine stores undirected roads in both orientations. Represent each
    # physical edge once for state comparisons; don't change the recorded rows.
    obs["roads"] = sorted({(min(edge), max(edge), color) for edge, color in obs["roads"]})
    return fingerprint({"map": map_hash, "observation": obs,
                        "legal_actions": inputs["legal_actions"], "action": inputs["action"]})


def scan(directories):
    started = perf_counter()
    global_inputs, global_decisions, global_positions = Index(), Index(), Index()
    datasets = []
    for root in directories:
        if (root / "FAILED.json").exists():
            raise ValueError(f"refusing failed dataset: {root}")
        results = [json.loads(line) for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        maps = {m["map_ref"]: fingerprint(m["map"]) for m in
                map(json.loads, (root / "maps.jsonl").read_text(encoding="utf-8").splitlines())}
        full_rows, ids, inputs, decisions, positions = (Index() for _ in range(5))
        choices, split_counts = Counter(), Counter()
        candidate_source = {}
        for result in results:
            split = result["split"]
            with gzip.open(root / result["moves_file"], "rt", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    location = {"dataset": root.name, "game_index": row["game_index"],
                                "action_index": row["action_index"]}
                    row_id = (row["game_index"], row["action_index"])
                    ids.add(row_id, location, split)
                    full_rows.add(fingerprint(row), location, split)
                    key = input_key(row, maps[row["inputs"]["map_ref"]])
                    position_key = input_key(row, maps[row["inputs"]["map_ref"]], ignore_turn=True)
                    inputs.add(key, location, split)
                    positions.add(position_key, location, split)
                    global_inputs.add(key, location, split)
                    global_positions.add(position_key, location, split)
                    if row["train_eligible"]:
                        decisions.add(key, location, split)
                        global_decisions.add(key, location, split)
                        choices[row["actor_bot"]] += 1
                    split_counts[split] += 1
                    if row.get("imitation_eligible"):
                        candidate_source[row_id] = fingerprint(row)
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        assert full_rows.rows == summary["actions"]
        assert decisions.rows == summary["eligible_rows"]
        entry = {"dataset": str(root.resolve()), "files": len(results),
                 "full_row_duplicates": full_rows.report(), "row_id_duplicates": ids.report(),
                 "equivalent_inputs_and_action": inputs.report(),
                 "eligible_inputs_and_action": decisions.report(),
                 "same_position_and_action_ignoring_turn": positions.report(),
                 "eligible_by_bot": dict(choices), "split_action_counts": dict(split_counts)}
        export = root / "candidate.moves.jsonl.gz"
        if export.exists():
            export_rows, export_ids, export_inputs = Index(), Index(), Index()
            verified = 0
            with gzip.open(export, "rt", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    row_id = (row["game_index"], row["action_index"])
                    location = {"game_index": row_id[0], "action_index": row_id[1]}
                    split = row["split"]
                    digest = fingerprint(row)
                    export_rows.add(digest, location, split)
                    export_ids.add(row_id, location, split)
                    export_inputs.add(input_key(row, maps[row["inputs"]["map_ref"]]), location, split)
                    assert candidate_source[row_id] == digest
                    verified += 1
            assert verified == len(candidate_source) == summary["candidate_rows"]
            entry["candidate_export"] = {
                "full_row_duplicates": export_rows.report(), "row_id_duplicates": export_ids.report(),
                "equivalent_inputs_and_action": export_inputs.report(),
                "intentional_overlap_with_source_rows": verified,
                "note": "Subset export, not extra training examples; do not concatenate with its source files."}
        datasets.append(entry)
    return {"datasets": datasets, "combined": {
        "equivalent_inputs_and_action": global_inputs.report(),
        "eligible_inputs_and_action": global_decisions.report(),
        "same_position_and_action_ignoring_turn": global_positions.report()},
        "elapsed_seconds": perf_counter() - started,
        "definitions": {
            "full_rows": "Canonical parsed JSON, including all metadata and targets.",
            "equivalent_inputs": "Actual map contents + preaction observation + legal actions + action; no game/map IDs or targets; roads normalized to unique undirected edges.",
            "position_repeats": "Same as equivalent inputs with only observation.turn omitted; possible legitimate repeated positions, not automatically corrupt rows.",
            "exports": "Authoritative games read once. candidate and first-five files are exports, not additional examples."}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = scan(args.directories)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    for item in report["datasets"]:
        print(json.dumps({"dataset": item["dataset"],
                          "rows": item["full_row_duplicates"]["rows"],
                          "full_duplicates": item["full_row_duplicates"]["duplicate_extra_rows"],
                          "id_duplicates": item["row_id_duplicates"]["duplicate_extra_rows"],
                          "equivalent_inputs": item["equivalent_inputs_and_action"]["duplicate_extra_rows"],
                          "position_repeats": item["same_position_and_action_ignoring_turn"]["duplicate_extra_rows"],
                          "export": item.get("candidate_export")}), flush=True)
    print(json.dumps(report["combined"], indent=2), flush=True)
