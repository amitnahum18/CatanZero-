"""Document and attach audited depth10 experiments and the final champion."""
import os
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
import hashlib
import json
from pathlib import Path
import shutil
import mlflow
from scipy.stats import binomtest

REPO=Path(__file__).resolve().parents[1]


def main():
    root=REPO/'runs/tree-policy-trade-aware-v3';parent=REPO/'runs/tree-policy-native-f-max-v1'
    report=json.loads((root/'report.json').read_text());engine=json.loads((root/'engine_audit.json').read_text())
    audit=json.loads((root/'model_audit.json').read_text());reload=json.loads((root/'isolated_reload.json').read_text())
    assert engine['status']==audit['status']==reload['status']=='passed'
    assert report['final_test']['completed']==500 and report['final_test']['wins']==282
    assert report['best_contains_MC_RL_correction']
    tracking=json.loads((root/'tracking.json').read_text());registration=json.loads((root/'registration.json').read_text())
    mlflow.set_tracking_uri(tracking['tracking_uri']);client=mlflow.MlflowClient()
    kept=client.get_model_version_by_alias('CatanZero_Online_Linear_SARSA','feature_pilot');assert str(kept.version)=='4'
    best=client.get_model_version_by_alias(registration['family'],registration['alias']);assert str(best.version)==registration['version']
    rows=[json.loads(s) for s in (root/'final_test/results.jsonl').read_text().splitlines()]
    seats={str(seat):{'games':sum(r['learner_seat']==seat for r in rows),
                     'wins':sum(r['learner_seat']==seat and r['learner_won'] for r in rows)} for seat in (0,1)}
    report['final_seats']=seats
    report['two_sided_binomial_p_vs_50']=float(binomtest(282,500,.5).pvalue)
    report['verification']={'engine_replay':engine,'model_audit':audit,'isolated_reload':reload,
                             'v4_four_player_best_alias_preserved':True}
    report['model_hashes']={name:hashlib.sha256((root/'best_model'/name).read_bytes()).hexdigest()
                            for name in ('ranker.model','monte_carlo.joblib','policy.json')}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    fit=json.loads((root/'fit.json').read_text());teacher=json.loads((root/'teacher_report.json').read_text())
    mc_audit=json.loads((parent/'model_audit.json').read_text());prior=json.loads((parent/'report.json').read_text())
    history=json.loads((root/'selection_history.json').read_text())
    text=f"""# Depth-10 classical tree policy versus native F

## Result and model

The frozen selected policy won **282/500 (56.4%)** against exactly **one native F opponent** in the Catanatron two-player base simulator. All 500 games completed. Wilson 95% interval: **52.02%–60.68%**; auxiliary two-sided binomial p against 50%: **{report['two_sided_binomial_p_vs_50']:.6g}**. Seats were balanced: first relative seat 146/250 wins, second 136/250. This establishes evidence above equal-strength 50% for this benchmark, not full four-player/human-level Catan strength or global optimality.

The model combines **114 LightGBM ranking trees** and **64 ExtraTrees Monte Carlo value trees**, each with maximum depth 10 (actual maximum verified 10). It takes **264 raw predecision numeric features**. No neural network, Game construction/copy/execute, forward rollout or opponent-private cards occur in learned-policy scoring. Existing deterministic resource/board feature arithmetic is guarded against simulator calls.

The action score is:

`LightGBM_rank_score(raw_features) / 1.4757017551196938 + 0.2 * MC_tree_value(raw_features)`

Choose the maximum among current legal actions; canonical first tie. Scores are **preferences, not calibrated 0–1 win probabilities**. Numeric scaling/clipping from the old linear normalizer is not applied to these trees; its feature names/version provide the existing encoding contract. Absent opponent slots are zero-padded for the two-player adapter. True future environment randomness is not consumed during policy deliberation.

## Training and real reward

Warm policy training uses **500 complete training-game trajectories / {teacher['decision_groups']:,} decision groups / {teacher['candidate_rows']:,} legal candidate rows**: 300 earlier learner games plus 200 real games played by the stronger ranker against native F. Native-F preferences were generated only for training states on detached public-information beliefs. The teacher was strengthened to consider an immediate legal city/settlement after a maritime trade. This extra deterministic backup is an **offline annotation operation**, not search by the deployed model. It evaluated {teacher['trade_build_followups']:,} such building continuations. A focused regression test reproduces a real state where native F ended its turn despite a trade enabling a city, and verifies the stronger label changes while its input features remain identical.

Ranking relevance is best/second/third distinct teacher value = 3/2/1, others = 0. LightGBM LambdaRank depth10, 127 leaves, min-child16, L2=1, learning rate .06; iteration count selected through whole-game label holdouts (last20 games of each100-game source batch), then refitted on all500 training games. The final fit used114 boosting iterations and took **{fit['seconds']:.1f} seconds**. Label aggregation/reconstruction/serialization took **{teacher['seconds']:.1f} seconds** with8 processes.

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

Full replay of all **800 final-stage matches / {engine['actions']:,} actual actions** checks whitelisted observations, complete legal lists, selected moves, final states and winners. UUIDs generated independently are excluded from final-state equality. Feature/scoring checks on the first/last game in each phase verified **{audit['decisions_verified']:,} decisions / {audit['prediction_candidates_verified']:,} candidates**, exact score error0, **{engine['privacy_checks']} private/future-field invariance checks**, and460 independently recomputed native-F choices. The earlier real-RL stage also passed full replay of1,050 matches/226,229 actions, all21,661 rewards, and independent MC target/leaf checks. Scope is explicit: raw numeric feature re-encoding is sampled, reward and engine replay checks are complete.

MLflow load/predict matches exactly, including a separate `python -I` process without repository source imports. All model depths/selection lineage and copied MC predictions match. Existing v4 model weights and its `feature_pilot` registry alias remain unchanged.

The complete final-stage annotation/fit/validation/test/registration run took **{report['elapsed_seconds']:.1f} seconds**; its500-game test took **{report['final_test']['elapsed_seconds']:.1f} seconds**, and final ranker fitting took29.6seconds. These are not the duration of the entire multi-experiment session. Prior model trials and verification added time; CatBoost and MC-mixed evaluation were materially slower. No claim that the whole project took30seconds.

Registered model: **`CatanZero_TreePolicy_NativeF` version{registration['version']}**, alias **`best_pilot`**. URI **`{registration['model_uri']}`**; run **`{tracking['run_id']}`**. Native ranker, MC forest, score-combination metadata, raw input signature, inference code/dependencies, results, audit and documentation are saved. Local files: `runs/tree-policy-trade-aware-v3/best_model`, `registration.json`, `report.json`, full game/decision traces and teacher dataset.

## Earlier trials and interpretation

The initial residual ExtraTrees FQI depth10 pilot trained300 games and9 fitting updates, but its new100-game test won11 games versus v4's23 on the same boards. That model remains an archived pilot, not the champion. Next, native-F ranker distillation plus two actual MC learning batches tested7 candidates: the selected warm-start-only ranker won245/500 (49%); RL corrections did not improve its validation selection. Dataset aggregation and a CatBoost trial tested5 more candidates: selected LightGBM won246/500 (49.2%), again without MC in the winner. Those old tests are retired and were excluded from each later heldout test. They are not combined into a single success denominator or reused to select the final model.

The final focused trade-aware annotation changed which prior actions are considered promising, and the MC correction then won validation selection. The current champion is a **hybrid of learned teacher ranking and real-outcome RL value correction**, not pure from-scratch RL and not a lookup table. It has evidence of beating native F in this two-player benchmark. Generalization to other opponents, four players, other rules, or humans remains unmeasured.
"""
    document=REPO/'docs/tree-policy-depth10-native-f.md';document.write_text(text,encoding='utf-8');(root/'README.md').write_text(text,encoding='utf-8')
    # Archive a concise record of the initial requested RL pilot as well.
    initial=REPO/'runs/residual-extratrees-depth10-f-duel-v1';initial_report=json.loads((initial/'report.json').read_text())
    initial_doc=REPO/'docs/residual-extratrees-depth10-f-duel-v1.md'
    initial_doc.write_text('# Archived depth10 residual FQI pilot\n\n300 real two-player training games against native F; 36,379 causal transitions, 9 batch fits, 64 ExtraTrees per fit, maximum depth10. Terminal +1/-1, gamma1, exact potential shaping as documented in the final experiment. Frozen final test: 11/100 versus v4 23/100 on paired boards (p=.01690). No claim of improvement or promotion over v4.\n\nThis pilot led to teacher ranking/real-return MC/tree-policy optimization, documented in `tree-policy-depth10-native-f.md`. Local native and MLflow models remain saved. Initial fitting/scoring source tests passed; independent complete replay/MC numerical audits apply to the later final policy and are not retroactively claimed for this archived FQI pilot.\n',encoding='utf-8')
    for file in ('tools/train_trade_aware_tree.py','tools/audit_saved_policy.py','tools/audit_residual_tree_games.py',
                 'tools/finalize_tree_experiments.py','tools/verify_tree_mlflow_isolated.py','tests/test_trade_teacher.py'):
        dest=root/'implementation_snapshot'/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/file,dest)
    with mlflow.start_run(run_id=tracking['run_id']):
        mlflow.set_tags({'audit':'passed','beats_native_F_two_player':'true','neural_network':'false','learner_forward_simulation':'false',
                         'training_type':'teacher-ranking plus real-game MonteCarlo RL correction'})
        mlflow.log_metrics({'engine_replay_games':engine['games'],'engine_replay_actions':engine['actions'],
                            'prediction_candidates_verified':audit['prediction_candidates_verified'],
                            'isolated_reload_error':reload['max_abs_error'],'binomial_p_vs_50':report['two_sided_binomial_p_vs_50'],
                            'win_rate_95_lower':report['wilson_95'][0],'win_rate_95_upper':report['wilson_95'][1]})
        for file in (root/'report.json',root/'engine_audit.json',root/'model_audit.json',root/'isolated_reload.json',document,root/'README.md'):
            mlflow.log_artifact(str(file),'audit_and_documentation')
        mlflow.log_artifact(str(root/'report.json'),'experiment')
        mlflow.log_artifacts(str(root/'implementation_snapshot'),'implementation_snapshot')
        mlflow.log_artifact(str(parent/'engine_audit.json'),'MC_training_audit');mlflow.log_artifact(str(parent/'model_audit.json'),'MC_training_audit')
    # Preserve earlier saved runs as finished, attach their audit/documentation.
    for path in (parent,initial):
        track=json.loads((path/'tracking.json').read_text())
        with mlflow.start_run(run_id=track['run_id']):
            if path==parent:
                for name in ('engine_audit.json','model_audit.json','isolated_reload.json'):mlflow.log_artifact(str(path/name),'audit_and_documentation')
            else:mlflow.log_artifact(str(initial_doc),'documentation')
    assert client.get_run(tracking['run_id']).info.status=='FINISHED'
    final={'wins':282,'games':500,'win_rate':.564,'CI95':report['wilson_95'],'MC_RL_in_selected_model':True,
           'registered_family':registration['family'],'registered_version':registration['version'],'run_id':tracking['run_id'],
           'doc':str(document),'all_required_audits_passed':True}
    write=root/'final_verification.json';write.write_text(json.dumps(final,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(final),flush=True)


if __name__=='__main__':main()
