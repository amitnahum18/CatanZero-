# Frozen v4 versus one F opponent: two-player search comparison

v4 with bounded search won **24/100** against native F; the same frozen v4 without search won **30/100**. The paired difference is not statistically significant (12 gained wins, 18 lost wins, exploratory two-sided p=0.3616). This pilot provides no evidence to replace direct v4 scoring with this search wrapper.

| v4 search | Opponent | v4 wins | F wins | Mean v4 decision |
|---|---|---:|---:|---:|
| No | F_static | 31/100 | 69/100 | 10.80 ms |
| No | native F (one-action search) | 30/100 | 70/100 | 11.25 ms |
| Yes | F_static | 29/100 | 71/100 | 49.34 ms |
| Yes | native F (one-action search) | 24/100 | 76/100 | 53.35 ms |


## Fixed protocol

Four conditions use the same 100 fresh boards, initial environment streams and learner seats (50 in each relative seat), with exactly two real players and a target of 10 VP. All 400 games reached victory. No training, exploration, hyperparameter tuning or registry alias promotion occurred. Weights and train-fitted normalization from `runs/online-sarsa-features-100-v4` remain unchanged. Canonical tie-breaking is used consistently in the direct and searched v4 variants; this differs from randomized ties in the previous four-player evaluator.

The simulator uses standard Catanatron base rules with two players, without additional official two-player variant rules. v4 was trained with four players. An explicit `allow_fewer_players` transfer flag zero-pads the raw features for absent relative seats/opponents while retaining the original 264-feature order, offsets, scales and clipping. Public/own features and strategic calculations use only the actual two players; no phantom players are introduced. Default four-player encoding stays unchanged. These two-player rates cannot be compared directly with v4's earlier 32/100 against three F_static opponents.

## What depth means here

Direct v4 ranks all current legal actions by the original frozen SARSA Q. The searched v4 expands the top three Q-ranked root actions, with two sampled public-count beliefs per choice, continuing the actual opponent policy until the next v4 decision. It then backs up terminal win/loss or the original .05-own-VP potential reward plus discounted maximum Q (gamma .999 per simulated table action). An entire decision falls back to direct Q if any branch cannot reach its endpoint within 24 actions; zero such fallbacks occurred in these games. This is an approximate next-own-decision search, not two full turns or full minimax.

Native F is the installed, unmodified `ValueFunctionPlayer`: it executes each legal candidate once on a copy and evaluates its hand-coded state value. Its synthetic view has detached randomness, so candidate simulations cannot consume the real future dice/card stream. Native F is a different heuristic from F_static; changing F_static to F changes both evaluation logic and forward simulation, so that contrast is not a pure depth-only ablation. The two sides use different depth definitions and computational budgets. Internal candidate simulations inside F are omitted from v4's branch-action statistics.

Both search policies receive only their own known cards and public observations through synthetic beliefs. Opponent exact private cards and actual future randomness are not used. Real future draws may diverge after policies choose different moves, despite identical initial seeds.

## Recovery and verification

The initial run finished 399 games but failed to sample a live belief in one late-game decision of `both/game_0028` within 128 rejection attempts. The same deterministic sample sequence now permits 4096 attempts, preserving choices where any of the first 128 samples succeeded. The missing game was rerun from the same original seed; the 399 complete games were retained. The failed partial trace is preserved under `both/game_0028_failed_attempt`. A regression test reproduces the old failure and verifies the new sampler returns a live view without changing real state/RNG. No failed game is counted as a loss or omitted from the 100-game denominator.

All **400 games / 102,204 actions** were independently replayed through the engine, checking predecision whitelisted observations, complete legal-action lists, selected actions, final state and winners. In 12 games (first, last and index 28 in all four conditions), **1,955 choices** were recomputed exactly, including bounded search. **80 privacy-invariance checks** and 43 distinct focused tests passed. Initial boards were checked against all prior recorded boards and normalization-source boards under both hash conventions. Final-state comparison excludes the independently generated UUID; all simulator state and game rule metadata match.

Measured execution time, including the initial failed batch and recovery: **249.9 seconds**; independent replay/scoring audit: **63.1 seconds**. Preparation, debugging pause, tests and final documentation are not included. The recovery-only 5.5 seconds is not the full benchmark duration.

## Interpretation and saved artifacts

Search changes many decisions but did not improve measured wins in this pilot. An inaccurate Q continuation, beam omissions, sparse belief sampling and transfer from four-player training can limit or degrade search. This experiment does not isolate which factor is responsible, nor prove every deeper search would fail. The five paired comparisons are exploratory and unadjusted for multiple testing.

All traces, complete games, protocol, copied model/normalization, results, audit and source snapshots are in `runs/v4-vs-f-duel-search-100-v1`. Small reports, source and documentation are attached to MLflow experiment `CatanZero_V4_F_Duel_Search`, completed run `deeef5730ec743aa942d1ba04f2e1746`; the original technical failure remains recorded as failed run `4dd298bbb6454bc1866b737c7016349d`. No new trained model was produced; v4's existing registered model and aliases are preserved.
