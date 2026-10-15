# Depth-10 classical tree policy versus native F

## Result and model

The frozen selected policy won **282/500 (56.4%)** against exactly **one native F opponent** in the Catanatron two-player base simulator. All 500 games completed. Wilson 95% interval: **52.02%–60.68%**; auxiliary two-sided binomial p against 50%: **0.00479157**. Seats were balanced: first relative seat 146/250 wins, second 136/250. This establishes evidence above equal-strength 50% for this benchmark, not full four-player/human-level Catan strength or global optimality.

The model combines **114 LightGBM ranking trees** and **64 ExtraTrees Monte Carlo value trees**, each with maximum depth 10 (actual maximum verified 10). It takes **264 raw predecision numeric features**. No neural network, Game construction/copy/execute, forward rollout or opponent-private cards occur in learned-policy scoring. Existing deterministic resource/board feature arithmetic is guarded against simulator calls.

The action score is:

`LightGBM_rank_score(raw_features) / 1.4757017551196938 + 0.2 * MC_tree_value(raw_features)`

Choose the maximum among current legal actions; canonical first tie. Scores are **preferences, not calibrated 0–1 win probabilities**. Numeric scaling/clipping from the old linear normalizer is not applied to these trees; its feature names/version provide the existing encoding contract. Absent opponent slots are zero-padded for the two-player adapter. True future environment randomness is not consumed during policy deliberation.

## Training and real reward

Warm policy training uses **500 complete training-game trajectories / 29,362 decision groups / 241,779 legal candidate rows**: 300 earlier learner games plus 200 real games played by the stronger ranker against native F. Native-F preferences were generated only for training states on detached public-information beliefs. The teacher was strengthened to consider an immediate legal city/settlement after a maritime trade. This extra deterministic backup is an **offline annotation operation**, not search by the deployed model. It evaluated 3,423 such building continuations. A focused regression test reproduces a real state where native F ended its turn despite a trade enabling a city, and verifies the stronger label changes while its input features remain identical.

Ranking relevance is best/second/third distinct teacher value = 3/2/1, others = 0. LightGBM LambdaRank depth10, 127 leaves, min-child16, L2=1, learning rate .06; iteration count selected through whole-game label holdouts (last20 games of each100-game source batch), then refitted on all500 training games. The final fit used114 boosting iterations and took **29.6 seconds**. Label aggregation/reconstruction/serialization took **303.7 seconds** with8 processes.

The RL correction is independently learned from **200 actual new games against native F**, collected by frozen ranker version101 with epsilon .08. These are actual simulated matches; they are not playback-only wins or teacher labels. The final version402 itself was evaluated frozen, rather than being the collector in all200 training games. The Monte Carlo forest was refitted after the first100 games, then after200 games using all prior real experience (11,071 then21,661 actual same-actor transitions). ExtraTrees64, depth10, min-leaf16, max-features .75; each complete game contributes equal total weight via `1 / learner_decisions_in_game`.

Base terminal reward: **win +1, loss -1**. Gamma = **1** so terminal penalties do not incentivize delaying a loss. Potential:

`Phi(s) = .05 * own_known_actual_VP + .02 * (own_known_actual_VP - opponent_PUBLIC_VP)`

`r = terminal_outcome + Phi(next) - Phi(current)`, with terminal Phi=0.

Own point gain gives+.07 and opponent public point gain gives-.02 before terminal reset. No full gap penalty is repeatedly charged when nothing changes. The full episode return is exactly `outcome - Phi(initial)`, and the Monte Carlo target at a recorded action is `outcome - Phi(current)`. Endpoints that are censored/unknown are not labelled as losses. Completed long games are retained. Reward math, actor/time ordering, all21,661 causal transitions, all128 fitted MC trees' weighted leaf means, and source lineage were independently verified. These checks prove implementation consistency, not an optimal reward/model guarantee.

## Selection and heldout evaluation

Three fixed policies were evaluated on the same **new100 validation boards** (seed54000000 onward), balanced seats and initial chance streams. Actual futures may diverge once decisions differ:

| Candidate | Validation wins against native F |
|---|---:|
| Previous strongest ranker (400) | 47/100 |
| Trade-aware ranker (401) | 47/100 |
| Trade-aware ranker plus0.2 actual-return MC correction (402) | **51/100** |

Version402 was selected **before** a separate500-game test (seed55000000 onward). No final-test examples/labels entered fitting or selection. Training-source public maps, all previously recorded evaluation maps, validation maps and final maps were checked for overlap under both historic JSON hash conventions. The validation advantage does not isolate a statistically established causal effect for each component; the independent final result supports the complete selected hybrid.

Native F is the installed unmodified `ValueFunctionPlayer`, which evaluates one simulated action per legal candidate with its fixed hand-coded value function. It receives a synthetic public-count belief and its own exact known cards. F_static was not used as the opponent in these experiments. Both sides have their own legitimate private hand; neither is given the other's actual private cards or future deck/RNG. Target is10VP, standard base engine with2 players, no added official two-player variant rules.

## Verification, timing and saved artifacts

Full replay of all **800 final-stage matches / 164,417 actual actions** checks whitelisted observations, complete legal lists, selected moves, final states and winners. UUIDs generated independently are excluded from final-state equality. Feature/scoring checks on the first/last game in each phase verified **771 decisions / 3,814 candidates**, exact score error0, **33 private/future-field invariance checks**, and460 independently recomputed native-F choices. The earlier real-RL stage also passed full replay of1,050 matches/226,229 actions, all21,661 rewards, and independent MC target/leaf checks. Scope is explicit: raw numeric feature re-encoding is sampled, reward and engine replay checks are complete.

MLflow load/predict matches exactly, including a separate `python -I` process without repository source imports. All model depths/selection lineage and copied MC predictions match. Existing v4 model weights and its `feature_pilot` registry alias remain unchanged.

The complete final-stage annotation/fit/validation/test/registration run took **1087.2 seconds**; its500-game test took **428.9 seconds**, and final ranker fitting took29.6seconds. These are not the duration of the entire multi-experiment session. Prior model trials and verification added time; CatBoost and MC-mixed evaluation were materially slower. No claim that the whole project took30seconds.

Registered model: **`CatanZero_TreePolicy_NativeF` version3**, alias **`best_pilot`**. URI **`models:/m-4fef44a7a4784c0cb2558987216350fb`**; run **`b18980f8d5f4443789408c2fce7efb8d`**. Native ranker, MC forest, score-combination metadata, raw input signature, inference code/dependencies, results, audit and documentation are saved. Local files: `runs/tree-policy-trade-aware-v3/best_model`, `registration.json`, `report.json`, full game/decision traces and teacher dataset.

## Earlier trials and interpretation

The initial residual ExtraTrees FQI depth10 pilot trained300 games and9 fitting updates, but its new100-game test won11 games versus v4's23 on the same boards. That model remains an archived pilot, not the champion. Next, native-F ranker distillation plus two actual MC learning batches tested7 candidates: the selected warm-start-only ranker won245/500 (49%); RL corrections did not improve its validation selection. Dataset aggregation and a CatBoost trial tested5 more candidates: selected LightGBM won246/500 (49.2%), again without MC in the winner. Those old tests are retired and were excluded from each later heldout test. They are not combined into a single success denominator or reused to select the final model.

The final focused trade-aware annotation changed which prior actions are considered promising, and the MC correction then won validation selection. The current champion is a **hybrid of learned teacher ranking and real-outcome RL value correction**, not pure from-scratch RL and not a lookup table. It has evidence of beating native F in this two-player benchmark. Generalization to other opponents, four players, other rules, or humans remains unmeasured.

## Leakage and whether to continue RL

No leakage was found in the saved audits. Training, validation, and final-test boards were checked for overlap; the selected policy was frozen before the independent 500-game test; and no final-test examples or outcomes were used to fit or select it. Teacher labels were made from the current recorded training state and legal actions, using a detached public-information belief. The 264 numeric policy inputs are predecision features; the recorded winner, future deck/RNG and opponent private hand do not enter them. The audit replayed every actual game action and checked the selected action against the current legal list. It also ran 33 explicit private/future-field invariance checks and independently recomputed 460 native-F choices.

Scope limit: policy feature re-encoding and live-score reproduction were sampled (3,814 candidate rows), not exhaustively recomputed over every numeric row in the 241,779-row teacher table. The previous RL source stage did exhaustively verify all 21,661 reward transitions and its game replay, with sampled feature/prediction checks. Thus the evidence supports “no leakage found under these audits,” not a formal proof over every serialized training value.

The reward matches the stated two-player objective: maximize wins against F. A terminal +1 for a win and -1 for a loss directly optimizes head-to-head outcome; gamma=1 ensures the loss penalty does not reward prolonging a game. The potential is based only on the learner's known VP and the opponent's public VP. Because rewards use `Phi(next)-Phi(current)` and terminal potential is zero, the episode return telescopes exactly to `outcome-Phi(initial)`. For a fixed starting state, `Phi(initial)` is identical for every action, so this shaping does not change the ideal action ranking. It can make intermediate learning targets easier to fit, but it is not extra proof that an action is good. Optimizing win rate, rather than average VP margin, remains the right model-selection metric for this goal.

Continuing RL is technically sound, but blindly adding more fits to the current forest is unlikely to be the best route. Its 21,661 return targets came from 200 games played by ranker 101, while selected policy 402 is a later policy; that distribution gap can weaken the MC correction. A focused next round should collect fresh complete games using frozen policy 402 against the same native F, with a small amount of recorded exploration (for example epsilon .03-.08), then refit the MC value forest on old plus new actual-game returns. Preserve the LightGBM ranker and compare the old and updated mixtures on a new, fixed validation set; keep the update only if it improves wins, then use a separate untouched 500-game test. This targets the actual objective and trains the value correction on states the current champion visits. Report win rate and confidence interval as the primary score; track illegal moves, unfinished games, seat balance and VP margin as diagnostics. This is a proposed next experiment, not a result already measured.

## Continuation attempt toward a 60% win rate

A fresh RL continuation was run from the frozen 402 policy. It collected 200 new complete games against native F with epsilon=.05 exploration (91 wins, 20,908 actual return transitions), refit an ExtraTrees depth-10 Monte Carlo critic on the old 21,661 return rows plus the new experience, and compared five critic mix weights on a new 100-game validation set. The best validation candidate was version403, mix=.10, with61/100 wins; incumbent402 had58/100.

On a separate paired test of 500 new boards and balanced seats, incumbent402 won286/500 (57.2%, Wilson95% 52.82%–61.47%) and challenger403 won279/500 (55.8%, Wilson95% 51.42%–60.09%). Both policies played the same boards and seats. Challenger-only wins were8, incumbent-only wins15 (exact paired binomial p=.210). The challenger did not reach the 60% target and was not promoted; MLflow alias `best_pilot` remains on version3 / policy402. The fresh data did not demonstrate that refitting the current MC forest improved performance. The gap between validation (61%) and heldout test (55.8%) also shows why validation is only for selection.

The next useful RL attempt needs stronger or more targeted learning than simply pooling another batch into the same critic: e.g. compare a fresh-state-only critic against the pooled critic, tune the forest's leaf size/tree count on a sufficiently large validation set, and keep an independent paired final test. Any promoted policy should meet the desired heldout win rate; the 60% goal has not yet been achieved.
