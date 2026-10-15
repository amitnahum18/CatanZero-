# First classical regression comparison

Ten predefined configurations from six families were fitted on the original
7,278 training decisions, using the same frozen train-only normalizer and 164
features. Candidate configurations were written before fitting. No source data,
splits, features or labels were changed.

The candidates were Ridge alpha 1/10/100/1000, Lasso, ElasticNet, Random Forest,
Extra Trees and two shallow histogram Gradient Boosting configurations. Depth,
minimum leaf sizes, penalties and fixed iteration counts bound model complexity.
Gradient Boosting early stopping is disabled: it does not create a row-random
internal validation split across the recorded games.

The winner was selected by **validation RMSE**, with validation MAE as a tie
breaker. The selected model was frozen and saved before the test labels were
loaded for scoring. It was not refitted on validation. Only that model and the
original Ridge alpha=1 were evaluated on the test in this experiment.

| Model/configuration | Validation MAE | Validation RMSE |
|---|---:|---:|
| Gradient Boosting depth 3 | 0.169274 | 0.214331 |
| Gradient Boosting depth 5 | 0.172757 | 0.219708 |
| Ridge alpha=1000 | 0.175533 | 0.220775 |
| Random Forest | 0.170202 | 0.224483 |
| Lasso | 0.180101 | 0.227878 |
| Ridge alpha=100 | 0.181137 | 0.229307 |
| ElasticNet | 0.181569 | 0.230103 |
| Ridge alpha=10 | 0.183699 | 0.233347 |
| Extra Trees | 0.185542 | 0.233470 |
| Original Ridge alpha=1 | 0.184044 | 0.233963 |

Selected: Gradient Boosting depth 3, 150 iterations, learning rate 0.05,
min_samples_leaf=50, l2_regularization=10, random_state=42.

| Frozen test model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Selected Gradient Boosting | 0.173730 | 0.231902 | 0.042505 |
| Original Ridge alpha=1 | 0.183227 | 0.219149 | 0.144926 |
| Constant train mean | 0.197208 | 0.237090 | -0.000810 |

Gradient Boosting reduced test MAE by about 5.2% relative to original Ridge,
but increased test RMSE by about 5.8%. It is not uniformly better. The per-game
results expose substantial variation, including a worse first F test game.
Only five games contribute to each held-out split, and their rows are correlated.
The existing test was reported previously, so this is exploratory confirmation
on existing data, not a new sealed final test. No further tuning followed its
result in this experiment.

The target remains the heuristic/outcome blend. Lower label error does not prove
better move selection, calibrated confidence or a higher game win rate. No game
strength evaluation has been performed for these models.

All ten estimators were saved, reloaded, and checked. A separate artifact audit
recomputes their train/validation metrics, verifies the validation-only selection,
verifies the selected saved model and test predictions, and confirms that all
source split files, normalization parameters and schemas have unchanged hashes.

```powershell
python tools/compare_regression_models.py runs/regression-competitive-v2 --output runs/model-comparison-v1
python tools/audit_model_comparison.py runs/model-comparison-v1
```

Use the host Python with the existing installed NumPy/sklearn/joblib packages.
The training tool does not require additional packages or API calls. The
project's minimal simulator venv does not include sklearn.

The saved `selected_model.joblib` expects the normalized X array, in the exact
feature order from `feature_schema.json`. Apply the saved normalizer to raw
features once; do not normalize the already-normalized dataset a second time.
Clip predictions to [0,1], as used in this comparison.

```python
import joblib
import numpy as np

model = joblib.load("runs/model-comparison-v1/selected_model.joblib")
scores = np.clip(model.predict(normalized_x), 0, 1)
```
