# Preserved champion and fresh Monte Carlo RL continuation

## Outcome

The original registry version3 remains best_pilot, in accordance with the user request to retain the previous successful method. The continuation did not establish a reliable improvement. The original 282/500 (56.4%) result remains a historical independent test,
not a target attached to every future sample. This experiment compares both frozen
policies against the same one-native-F, two-player base-engine conditions, target10VP.

New independent paired test:

| Policy | Wins | Games | Win rate |
|---|---:|---:|---:|
| Preserved champion | 107 | 200 | 53.5% |
| Continued critic, mix0.4 | 111 | 200 | 55.5% |

Discordant pairs: 9 continuation-only wins,
5 baseline-only wins. Exact two-sided paired p=0.42395.
The two runs share initial board, seat and chance stream; outcomes can diverge after decisions.
The final test was never used to fit trees or choose the critic mixture. A positive
raw win difference alone is insufficient for promotion. Paired p<.05, positive win
difference, validation selection and all audits must pass.

## Saved method

The deployed policy is still **two classic tree models**, 114 frozen LightGBM
ranking trees plus64 ExtraTrees return-regression trees, maximum depth10, raw264
known-predecision features. Score is `ranker / scale + mix * critic`.
Depth10 is tree depth, not ten-game-move lookahead. Deployment performs no forward
simulation or neural computation. Teacher trade-to-build simulations occurred only
in the earlier offline annotation stage. Scores are not calibrated win probabilities.

Original method: `docs/tree-policy-depth10-native-f.md`.
Original immutable registry model: `CatanZero_TreePolicy_NativeF` version3,
alias `baseline_56_4`, native backup at `C:\project_ML\CatanZero\runs\tree-policy-rl-continuation-v4\baseline_model`.
Continued model: registry version4, alias `rl_continuation_pilot`.
Active best_pilot after audits: version3.
Original model file hashes and the copied backup match exactly; original weights
were never updated in place. The earlier four-player linear v4 is also preserved.

## Actual continuation and selection

The current champion played100 **new actual training games** against native F
with epsilon.08, winning45/100. These exploratory collection
wins are separate from frozen policy evaluation. All100 completed and yielded
10,497 actual same-actor transitions.

One ExtraTrees refit combined the earlier200 actual training games with these100:
32,158 chosen-action examples from300 complete games.
Each complete game has total sample weight1, preventing long episodes from
dominating solely because they contain more rows. Trees are refitted in a batch;
ExtraTrees has no incremental partial_fit. Only ExtraTrees changed; LightGBM stayed
identical, as verified by native artifact hashes. This is Monte Carlo policy
improvement with replay, not SARSA or pure teacher imitation. Old exploratory
policy data introduces a replay-policy mismatch, not future-input leakage.

Two mixtures were prespecified, .2 and .4, and compared to the baseline on100
new validation games, whole-game split. Earliest version wins ties.

| Version | Critic mixture | Validation wins |
|---|---:|---:|
| 500 | 0.2 | 51/100 |
| 501 | 0.2 | 52/100 |
| 502 | 0.4 | 53/100 |

One continuation was selected before200 new test games per policy. No additional
mixture was chosen after seeing the test. The training/validation/test seed ranges
are56m/57m/58m respectively. Both historical map hash conventions were excluded.

## Reward and ranking assessment

Terminal win=+1, loss=-1, gamma1. Potential uses only the actor's legitimately
known actual points and the opponent's **public** points:

`Phi(s)=.05*ownVP+.02*(ownVP-opponentPublicVP)`

`r=outcome+Phi(next)-Phi(current)` with terminal Phi0.

Own gained point yields+.07, opponent gained public point yields-.02, unchanged
gap yields0. The complete Monte Carlo return is exactly `outcome-Phi(current)`.
For a fixed decision, Phi is identical for all candidate actions, so shaping
does not by itself provide a counterfactual ordering of actions or alter the
true full-return optimal action. Gamma1 does not reward postponing a negative
discounted endpoint. Unknown/censored outcomes are not losses, and completed
long games are retained. No win-streak or repeated absolute-gap bonus is used.

Thus the endpoint objective is appropriate for maximizing victory; the main
limitation is the **weak action-specific learning signal**. One chosen action is
observed per state; no dataset row proves an unchosen action's true win return.
LightGBM's ordinal3/2/1 teacher labels are a prior relative ranking, not ground-truth
victory probability. We assess success by heldout actual victories, not teacher
agreement or regression error alone.

On2,000 multi-action next-decision sets from new TRAINING ONLY:

* Original critic changed 2/2,000 choices relative
  to LightGBM alone (0.10%).
* Continued critic at selected mixture changed 6/2,000
  (0.30%).
* Original median weighted critic spread was 0.00139523,
  versus teacher top-two score margin 0.967005.
* About7.0% of original
  critic impurity importance lies on features that varied among candidate actions
  in this sample. Most importance describes common state value.

These are sample diagnostics, not a claim about all possible states or proof that
the critic never matters. Even rare changed actions can alter whole trajectories.
Importance is descriptive and affected by correlated features, not causal.

A justified next research direction is **action advantage learning**: estimate a
state-only baseline V(s) with whole-game cross-fitting, learn actual-return minus
that baseline, and update the action policy with controlled exploration and a
regularizer retaining useful teacher behavior. Subtracting a common constant
at prediction time alone would not change rankings; the learning target/objective
must change. This direction is not implemented or proven better in this pilot.
Merely collecting more episodes or increasing critic mixture cannot guarantee
maximum ability. Further candidates need fresh heldout evaluation.

## Leakage and verification

No leakage was found in the checks performed. Numeric inputs are264 explicitly
whitelisted predecision features. Seed, game identity, winner, rewards, teacher
labels, actual future rolls/deck order and opponents' actual private cards are
not prediction columns. Outcome is used as a training target only. Visible bank
and card counts are legitimate public information. Deterministic after-action
resource/graph arithmetic uses known information, not sampled hidden futures.

The previous champion passed a fresh replay of800 matches/164,417 actions,
771 decisions/3,814 candidate scores and33 feature-privacy invariance checks.
It also passed a separate isolated MLflow load without repo source imports.

This pilot passed full independent replay of800 matches/166,612
actions; all10,497 new causal rewards; all32,158
return targets; all64 fitted forest weighted leaf values; whole-game train/val/test
disjointness and source provenance. Raw feature and scoring reconstruction was
sampled (1268 decisions/6510 candidates),
with44 private/future-field invariance checks. Maximum score
error0.0; isolated reload error0.0.
Scope distinguishes full replay/reward checks from sampled feature verification.

Fit seconds: 11.1.
Complete gameplay/fit/registration seconds: 510.6,
excluding subsequent audits. Source, native models, target/weight rows, actual
decision traces, validation selection, independent results, audit reports and
this method are saved locally and in MLflow. No human/four-player/global-optimum
claim is made.
