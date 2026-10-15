"""Save the audited method and decide promotion from the prespecified gate."""
import os
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
import json
from pathlib import Path
import shutil
import mlflow
from continue_champion_tree_rl import REPO,SOURCE,sha


def main():
    root=REPO/'runs/tree-policy-rl-continuation-v4'
    read=lambda name:json.loads((root/name).read_text())
    report=read('report.json');engine=read('engine_audit.json');model=read('model_audit.json')
    reload=read('isolated_reload.json');ranking=read('ranking_assessment.json')
    assert engine['status']==model['status']==reload['status']=='passed'
    assert json.loads((SOURCE/'engine_audit.json').read_text())['status']=='passed'
    assert json.loads((SOURCE/'model_audit.json').read_text())['status']=='passed'
    assert json.loads((SOURCE/'isolated_reload.json').read_text())['status']=='passed'
    config=read('config.json');selection=read('selection.json');tracking=read('tracking.json');registration=read('registration.json')
    mlflow.set_tracking_uri(tracking['tracking_uri']);client=mlflow.MlflowClient()
    assert str(client.get_model_version_by_alias(registration['family'],'baseline_56_4').version)=='3'
    before=report['baseline_test'];after=report['continued_test'];history=read('selection_history.json')
    assert before['completed']==after['completed']==200
    qualify=(after['wins']>before['wins'] and report['paired']['two_sided_p']<.05
             and selection['best_validation_version']==selection['continued_version'])
    assert qualify==report['qualifies_for_promotion_after_audits']
    retention_path=root/'retention_decision.json'
    keep_requested=retention_path.exists() and json.loads(retention_path.read_text())['keep_original']
    promote=qualify and not keep_requested
    if promote:
        client.set_registered_model_alias(registration['family'],'best_pilot',registration['version'])
    else:
        assert str(client.get_model_version_by_alias(registration['family'],'best_pilot').version)=='3'
    report['promoted']=bool(promote)
    report['baseline_retained_by_user_preference']=bool(keep_requested)
    report['active_best_registry_version']=registration['version'] if promote else '3'
    report['all_required_audits_passed']=True
    report['verification']={'engine':engine,'model':model,'isolated_reload':reload}
    report['baseline_checkpoint_alias']='baseline_56_4'
    report['continued_model_hashes']={n:sha(root/'continued_model'/n) for n in ('ranker.model','monte_carlo.joblib','policy.json')}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    p=ranking['policies'];source_signal=p['500'];new_signal=p[str(selection['continued_version'])]
    validation='\n'.join(f"| {h['version']} | {h['mix']} | {h['validation']['wins']}/100 |" for h in history)
    status=('The continued policy passed the prespecified promotion gate and is the active best_pilot.' if promote else
            'The original registry version3 remains best_pilot, in accordance with the user request to retain the previous successful method. '
            +('The continuation met the statistical gate but is retained as an archived pilot.' if qualify else
              'The continuation did not establish a reliable improvement.'))
    doc=f"""# Preserved champion and fresh Monte Carlo RL continuation

## Outcome

{status} The original 282/500 (56.4%) result remains a historical independent test,
not a target attached to every future sample. This experiment compares both frozen
policies against the same one-native-F, two-player base-engine conditions, target10VP.

New independent paired test:

| Policy | Wins | Games | Win rate |
|---|---:|---:|---:|
| Preserved champion | {before['wins']} | 200 | {before['wins']/2:.1f}% |
| Continued critic, mix{new_signal['mix']} | {after['wins']} | 200 | {after['wins']/2:.1f}% |

Discordant pairs: {report['paired']['candidate_only_wins']} continuation-only wins,
{report['paired']['baseline_only_wins']} baseline-only wins. Exact two-sided paired p={report['paired']['two_sided_p']:.6g}.
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
alias `baseline_56_4`, native backup at `{root/'baseline_model'}`.
Continued model: registry version{registration['version']}, alias `rl_continuation_pilot`.
Active best_pilot after audits: version{report['active_best_registry_version']}.
Original model file hashes and the copied backup match exactly; original weights
were never updated in place. The earlier four-player linear v4 is also preserved.

## Actual continuation and selection

The current champion played100 **new actual training games** against native F
with epsilon.08, winning{report['training']['wins']}/100. These exploratory collection
wins are separate from frozen policy evaluation. All100 completed and yielded
{report['training']['transitions']:,} actual same-actor transitions.

One ExtraTrees refit combined the earlier200 actual training games with these100:
{model['MC_training_rows']:,} chosen-action examples from300 complete games.
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
{validation}

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

* Original critic changed {source_signal['changed_actions']}/2,000 choices relative
  to LightGBM alone ({100*source_signal['fraction_changed_from_teacher_only']:.2f}%).
* Continued critic at selected mixture changed {new_signal['changed_actions']}/2,000
  ({100*new_signal['fraction_changed_from_teacher_only']:.2f}%).
* Original median weighted critic spread was {source_signal['median_weighted_MC_spread']:.6g},
  versus teacher top-two score margin {source_signal['median_teacher_top_two_margin']:.6g}.
* About{100*source_signal['MC_importance_on_decision_varying_features']:.1f}% of original
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

This pilot passed full independent replay of{engine['games']} matches/{engine['actions']:,}
actions; all{engine['reward_transitions']:,} new causal rewards; all{model['MC_training_rows']:,}
return targets; all64 fitted forest weighted leaf values; whole-game train/val/test
disjointness and source provenance. Raw feature and scoring reconstruction was
sampled ({model['decisions_rescored']} decisions/{model['candidates_rescored']} candidates),
with{engine['privacy_checks']} private/future-field invariance checks. Maximum score
error{model['max_score_error']}; isolated reload error{reload['max_abs_error']}.
Scope distinguishes full replay/reward checks from sampled feature verification.

Fit seconds: {json.loads((root/'fit.json').read_text())['fit_seconds']:.1f}.
Complete gameplay/fit/registration seconds: {report['elapsed_seconds']:.1f},
excluding subsequent audits. Source, native models, target/weight rows, actual
decision traces, validation selection, independent results, audit reports and
this method are saved locally and in MLflow. No human/four-player/global-optimum
claim is made.
"""
    doc_path=REPO/'docs/tree-policy-rl-continuation-v4.md';doc_path.write_text(doc,encoding='utf-8')
    (root/'README.md').write_text(doc,encoding='utf-8')
    for file in ('tools/continue_champion_tree_rl.py','tools/audit_tree_rl_continuation.py',
                 'tools/analyze_tree_rl_scores.py','tools/finalize_tree_rl_continuation.py',
                 'tools/audit_residual_tree_games.py','tests/test_tree_rl_continuation.py',
                 'src/catanzero/features.py','src/catanzero/dataset.py','src/catanzero/tree_duel_reward.py'):
        destination=root/'implementation_snapshot'/file;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO/file,destination)
    with mlflow.start_run(run_id=tracking['run_id']):
        mlflow.log_metrics({'engine_games_audited':engine['games'],'reward_transitions_audited':engine['reward_transitions'],
              'max_score_error':model['max_score_error'],'isolated_reload_error':reload['max_abs_error'],
              'promoted':float(promote),'baseline_MC_decision_change_fraction':source_signal['fraction_changed_from_teacher_only']})
        for path in [doc_path]+[root/n for n in ('report.json','engine_audit.json','model_audit.json','isolated_reload.json','ranking_assessment.json')]:
            mlflow.log_artifact(str(path),'audit_and_method')
        for name in ('engine_audit.json','model_audit.json','isolated_reload.json'):
            mlflow.log_artifact(str(SOURCE/name),'baseline_reverification')
        if retention_path.exists():mlflow.log_artifact(str(retention_path),'audit_and_method')
        mlflow.log_artifacts(str(root/'implementation_snapshot'),'implementation_snapshot')
    assert client.get_run(tracking['run_id']).info.status=='FINISHED'
    verification={'status':'passed','all_required_audits_passed':True,'best_registry_version':report['active_best_registry_version'],
              'baseline_test_wins':before['wins'],'continued_test_wins':after['wins'],'test_games_per_policy':200,
              'promoted':bool(promote),'doc':str(doc_path),'baseline_preserved':True}
    (root/'final_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verification),flush=True)


if __name__=='__main__':main()
