# Frozen champion 402

This folder contains the selected model, not training data or checkpoints.

- `best_model/ranker.model`: 114 LightGBM ranking trees, maximum depth 10.
- `best_model/monte_carlo.joblib`: 64 ExtraTrees regression trees, maximum depth 10.
- `best_model/policy.json`: score scale and correction weight.
- `normalizer.json`: feature version/order and historical training metadata.
- `portable.json`: numeric export of exactly the same 178 trees, used in the
  browser without installing either ML library. Raw 264-element feature vectors
  go directly into both components, as in the selected native policy; the
  historical normalizer is not applied to 402 inference.

Score: `ranker(raw) / 1.4757017551196938 + 0.2 * critic(raw)`.
This is an action preference, not a calibrated win probability. Inference does
not run simulations or update weights. sklearn's float32 input conversion is
preserved for the critic; LightGBM uses float64 candidate features.

The files were copied unchanged from
`runs/tree-policy-trade-aware-v3/best_model` (the historical local experiment),
then exported with `tools/export_champion402.py`. Original model hashes are
embedded in `portable.json`. `tools/verify_champion_portable.py` compares both
native components against the export on real game states.

Latest separate confirmation: 570/1,000 wins against one native F; exact 95%
interval 53.8647%–60.0940%. It does not establish a win rate above 55% with the
specified confidence. See `docs/402-confirmation-f-1000-20261015.md`.
