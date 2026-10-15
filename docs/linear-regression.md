# Normalized predecision linear regression

The first regression dataset uses the 32 F-vs-F games and 12 AB:2-vs-three-F
games. It contains 9,172 retained decisions: 7,278 train, 938 validation, 956 test.
Original splits remain at whole-game level: 34 / 5 / 5 games. Two validation
opening decisions were excluded because the compact features matched training
vectors, despite belonging to different boards. No source dataset was edited.

There are **164 numeric input features and one output, weighted_score**. Raw
metadata is stored in separate files, and is never passed to the fitting tool.

| Group | Inputs |
|---|---:|
| Turn, phase, bank, prompt, relative turn owner | 25 |
| Acting player's current hand, cards, public state, production, ports | 42 |
| Three opponents' public summaries | 24 |
| Action type one-hot | 13 |
| Deterministic resource costs/gains and requested monopoly resource | 15 |
| Candidate location: production, ports, nearby buildings/roads | 21 |
| Known production/point effects and public robber victim information | 9 |
| Action-resource/state interactions for linear scoring | 15 |
| Total | 164 |

This spatial summary is intentionally compact, not a full graph encoding.
It preserves useful properties of the candidate location, without treating node
IDs as quantities. Undirected roads stored in both orientations are counted once.
Opponents are ordered relative to the acting seat, not by absolute color.

Nonbinary features use z = (x - train_mean) / train_std. Means and population
standard deviations are fitted only to the 7,278 training decisions. One-hot
and binary flags (58 columns) remain 0/1. Constant training columns use scale 1
and are retained, so the schema does not depend on the validation/test data.
The target already lies in [0,1] and is not fitted or rescaled.

The stored output is the agreed 50% heuristic + 50% actor terminal win/loss.
That label is weak supervision, not calibrated confidence. The F32 source uses
the single-belief v2 heuristic; the AB12 source uses the eight-belief v3 heuristic.
No action's heuristic value, winner, terminal label, bot identity, game/map ID,
future dice/draw, or hidden opponent hand enters X. Current own private cards and
current own actual VP are legitimate observations. Deterministic action effects
are computed from the predecision public board and intent, never by stepping the
simulator or reading a recorded outcome.

`leakage_audit.json` independently verifies all 14,539 original predecision
actions and 44 authoritative game replays, recomputes every retained feature
and normalized value, recomputes the normalizer from train only, and verifies
there is no cross-split vector overlap. Invariance tests also inject changed
targets, metadata, future stream/deck, and hidden opponent fields and check that
X is unchanged. Sampled roll/card values in the action field are rejected.

The model is linear Ridge regression with fixed alpha=1 and an intercept. It
is fitted on train only, without hyperparameter selection on held-out scores.
Predictions are clipped to [0,1]. Test MAE is 0.1832, versus 0.1972 for predicting
the training mean; test RMSE is 0.2191 versus 0.2371. Validation RMSE is 0.2340,
slightly worse than the constant baseline's 0.2337. These are label prediction
metrics only: the model has not been evaluated as a game-playing bot.

## Commands

```powershell
# Use a new output directory each time.
.\.venv\Scripts\python.exe -m catanzero.regression_data runs/competitive-F-32 runs/quality-AB2-12 --output runs/regression-competitive-v2
.\.venv\Scripts\python.exe tools/audit_regression_data.py runs/regression-competitive-v2
# Host Anaconda Python already provides NumPy; no new package was installed.
python tools/train_linear_regression.py runs/regression-competitive-v2
```

## Saved files and inference

Each split's `.jsonl.gz` contains only `{"x": [164 normalized values], "y": score}`.
`feature_schema.json` defines the ordered column names. Provenance is in the
corresponding `.metadata.jsonl`, deliberately separate. `normalizer.json` freezes
the scaling parameters; `linear_model.json` stores coefficients and intercept.

Project-venv inference does not need NumPy:

```python
from catanzero.linear_model import LinearMoveScorer

scorer = LinearMoveScorer("runs/regression-competitive-v2/linear_model.json")
# map_document is the map assigned to inputs.map_ref in the source dataset.
# inputs is the same predecision observation/legal_actions/action structure.
ranked = scorer.rank(map_document, inputs)
chosen_action = ranked[0]["action"]
```
