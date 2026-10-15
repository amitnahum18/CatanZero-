"""Read-only diagnostics of legal-action score ties in saved gameplay."""
import gzip
import json
from pathlib import Path

root = Path("runs/gbrl-linear-curriculum-v1/batch_01")
groups = {"opening": ("initial_evaluation", [0, 1, 2, 3]),
          "late_training": ("live_training", [240, 241, 242, 243])}
result = {}
for name, (phase, ids) in groups.items():
    rows = []
    for index in ids:
        path = root / phase / f"game_{index:04d}" / "decisions.jsonl.gz"
        if path.exists():
            rows.extend(json.loads(line) for line in gzip.open(path, "rt"))
    decisions = [r for r in rows if r["policy"] == "GBRL_SARSA" and len(r["raw_scores"]) > 1]
    result[name] = {"decisions": len(decisions),
                    "all_legal_actions_tied": sum(max(r["raw_scores"]) == min(r["raw_scores"]) for r in decisions),
                    "multiple_best_actions": sum(r["raw_scores"].count(max(r["raw_scores"])) > 1 for r in decisions)}
print(json.dumps(result, indent=2))
