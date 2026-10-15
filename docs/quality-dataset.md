# Replay-derived AB:2 dataset

`python -m catanzero.tournament_dataset runs/bot-comparison-12 --output runs/quality-AB2-12 --belief-samples 8 --workers 4`

This converts the 12 existing AB:2-vs-three-F matches without calling the bots again.
Each recorded outcome is replayed exactly and the persisted final state is verified.
All actions, including forced actions, remain in each game's `.moves.jsonl.gz`.
`candidate.moves.jsonl.gz` contains only non-forced, completed-game decisions by
the designated AB:2 player, including decisions from its losses.

Each row contains `inputs` (pre-action observation, local map reference, all legal
actions, chosen action), separate `heuristic` and `targets`, and provenance metadata.
Resolve `inputs.map_ref` in `maps.jsonl`. Only `inputs` should enter the predictor;
bot identity, game IDs, split, outcome and scoring diagnostics stay outside it.
For imitation learning, the observation/map and legal actions are inputs and the
chosen action is the target. For move scoring, chosen action is also an input and
`targets.weighted_score` is the target.

All legal actions are evaluated under the same eight sampled live, public-count
beliefs. Each action's mean heuristic value is computed first; regret is the
best mean minus the chosen mean. H = exp(-regret / 3). The final target is the
previously agreed 0.5 H + 0.5 acting-player terminal win/loss. Candidate mean and
standard deviation are retained; this standard deviation describes sensitivity
to hidden-state sampling, not uncertainty about expert correctness.

This is a more promising behavior source than random bots: AB:2 won 5/12 in the
matched pilot, compared with 2/12 for the designated F player. That small pilot
does not establish statistical superiority or human-level strength. Averaging
beliefs can reduce sampling noise, but does not remove heuristic bias. These
scores are weak labels, not calibrated confidence, optimal-action probability,
or terminal rollout estimates. No player-to-player trading is modeled.

Splits are grouped by initial seed across all source tournament groups. With
12 distinct seeds this gives 8 training, 2 validation, and 2 test games. That is
too little for a reliable broad generalization claim. Splits and every pre-action
observation are checked by `tools/audit_quality_dataset.py`.

Map IDs and game IDs are local to this dataset. Qualify/remap them before merging
with an older run. Pairwise search labels from other runs remain separate and
cannot be treated as the same score or assigned this game's outcome.

Read the compact export:

```python
import gzip, json
from pathlib import Path

root = Path("runs/quality-AB2-12")
maps = {m["map_ref"]: m for m in map(json.loads, (root / "maps.jsonl").read_text().splitlines())}
with gzip.open(root / "candidate.moves.jsonl.gz", "rt", encoding="utf-8") as stream:
    rows = [json.loads(line) for line in stream]
train = [r for r in rows if r["split"] == "train"]
```
