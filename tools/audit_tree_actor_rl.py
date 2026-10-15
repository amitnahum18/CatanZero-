"""Verify real-game advantages, behavior likelihoods and actor tree updates."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
from continue_champion_tree_rl import load_policy,sha,SOURCE,paired_stats
from optimize_tree_actor_rl import ClippedActorObjective
from audit_tree_policy_model import tree_depth
from train_residual_tree_fqi import write


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    root=parser.parse_args().root.resolve();started=perf_counter()
    config=json.loads((root/'config.json').read_text());normalizer=json.loads((root/'normalizer.json').read_text())
    engine=json.loads((root/'engine_audit.json').read_text());assert engine['status']=='passed' and engine['games']==1000
    data=dict(np.load(root/'actor_data.npz'));experience=dict(np.load(root/'RL_round_1/experience.npz'))
    metadata=[json.loads(s) for s in (root/'RL_round_1/transitions.jsonl').read_text().splitlines()]
    results={r['game_index']:r for r in map(json.loads,(root/'RL_round_1/results.jsonl').read_text().splitlines())}
    models={v:load_policy(root/f'candidate_{v}',normalizer) for v in (700,701,702)}
    reference=models[700];temperature=config['temperature'];raw=experience['current_x']
    np.testing.assert_array_equal(data['x'],raw.astype(np.float32))
    scores=reference.predict(raw);critic=reference.mc.predict(raw);phi=experience['phi']
    last={};reward_returns=np.empty(len(metadata));counts={}
    for i,m in enumerate(metadata):
        counts[m['game_index']]=counts.get(m['game_index'],0)+1
    for indexes in ([i for i,m in enumerate(metadata) if m['game_index']==g] for g in counts):
        value=0.
        for i in reversed(indexes):
            value=experience['reward'][i]+experience['discount'][i]*value;reward_returns[i]=value
    probability_error=0.;game_mass={};sampled_raw=0
    for i,m in enumerate(metadata):
        a,b=experience['current_ranges'][i];selected=a+m['selected_index']
        assert a==data['starts'][i] and b-a==data['lengths'][i] and selected==data['selected_rows'][i]
        np.testing.assert_array_equal(raw[selected],experience['x'][i])
        previous=last.get(m['game_index'])
        if previous is not None:
            c,d=experience['ranges'][previous];np.testing.assert_array_equal(raw[a:b],experience['next_x'][c:d])
        last[m['game_index']]=i
        p=np.exp((scores[a:b]-scores[a:b].max())/temperature);p/=p.sum()
        error=abs(p[m['selected_index']]-experience['behavior_probability'][i]);probability_error=max(probability_error,error)
        np.testing.assert_allclose(p,data['old_probabilities'][a:b],rtol=0,atol=1e-12)
        assert error<1e-12 and abs(m['behavior_probability']-p[m['selected_index']])<1e-12
        baseline=float(p@critic[a:b]);expected=(1. if results[m['game_index']]['learner_won'] else -1.)-phi[i]
        assert abs(expected-data['return'][i])<1e-12 and abs(expected-reward_returns[i])<1e-10
        assert abs(baseline-data['baseline'][i])<1e-12
        assert abs(expected-baseline-data['advantage'][i])<1e-12
        game_mass[m['game_index']]=game_mass.get(m['game_index'],0)+data['weight'][i]
    assert max(game_mass.values())-min(game_mass.values())<1e-8
    np.testing.assert_allclose(data['fixed_critic'],reference.mix*critic,rtol=0,atol=1e-12)
    # Match sampled causal replay exports to both current and next group matrices.
    seen=set()
    with gzip.open(root/'audit_transitions.jsonl.gz','rt',encoding='utf-8') as stream:
        for case in map(json.loads,stream):
            i=case['position'];seen.add(i)
            assert abs(case['reward']-experience['reward'][i])<1e-12
            if case['x'] is not None:
                np.testing.assert_array_equal(case['x'],experience['x'][i]);a,b=experience['ranges'][i]
                np.testing.assert_array_equal(np.asarray(case['next_x']).reshape(-1,264),experience['next_x'][a:b]);sampled_raw+=1
    assert len(seen)==len(metadata)
    # Replay every derivative and verify all added leaf outputs against their
    # assigned samples. LightGBM consumes float32 gradients/Hessians internally.
    objective=ClippedActorObjective(data,reference.scale,temperature)
    actor=models[702].teacher;trees=actor.dump_model()['tree_info'];base=reference.teacher.num_trees()
    pred=reference.teacher.predict(data['x'],num_threads=1);leaf_error=0.;leaves_checked=0
    for iteration in range(base,len(trees)):
        grad,hess=objective(pred,None);grad=grad.astype(np.float32).astype(np.float64);hess=hess.astype(np.float32).astype(np.float64)
        assigned=actor.predict(data['x'],start_iteration=iteration,num_iteration=1,pred_leaf=True,num_threads=1).reshape(-1)
        gs=np.bincount(assigned,weights=grad);hs=np.bincount(assigned,weights=hess);ns=np.bincount(assigned)
        def check(node):
            nonlocal leaf_error,leaves_checked
            if 'leaf_index' not in node:
                check(node['left_child']);check(node['right_child']);return
            j=node['leaf_index'];expected=.05*np.clip(-gs[j]/(hs[j]+3.),-.5,.5)
            error=abs(node['leaf_value']-expected);assert error<2e-5
            assert node['leaf_count']==ns[j];leaf_error=max(leaf_error,error);leaves_checked+=1
        check(trees[iteration]['tree_structure'])
        pred+=actor.predict(data['x'],start_iteration=iteration,num_iteration=1,num_threads=1)
    np.testing.assert_allclose(pred,actor.predict(data['x'],num_threads=1),rtol=0,atol=1e-10)
    cases={};score_error=0.;decisions=candidates=0
    with gzip.open(root/'audit_predictions.jsonl.gz','rt',encoding='utf-8') as stream:
        for case in map(json.loads,stream):cases.setdefault(case['version'],[]).append(case)
    for version,group in cases.items():
        x=np.concatenate([np.asarray(c['raw']) for c in group]);actual=models[version].predict(x)
        expected=np.concatenate([np.asarray(c['scores']) for c in group]);error=float(np.max(np.abs(actual-expected)))
        assert error<1e-10;score_error=max(score_error,error);decisions+=len(group);candidates+=len(x)
        for c in group:
            if not c['explored']:assert np.argmax(c['scores'])==c['selected']
    for version,model in models.items():
        expected_count=114+({700:0,701:8,702:24}[version]);assert model.teacher.num_trees()==expected_count
        ts=model.teacher.dump_model()['tree_info'];assert max(tree_depth(t['tree_structure']) for t in ts)<=10
        for a,b in zip(models[700].teacher.dump_model()['tree_info'],ts[:114]):assert a==b
        np.testing.assert_array_equal(model.mc.predict(raw[::100]),reference.mc.predict(raw[::100]))
    for n,h in config['source_hashes'].items():assert sha(SOURCE/'best_model'/n)==h
    history=json.loads((root/'selection_history.json').read_text());selection=json.loads((root/'selection.json').read_text())
    best=max(history,key=lambda h:(h['validation']['wins'],-h['version']))
    continued=max(history[1:],key=lambda h:(h['validation']['wins'],-h['version']))
    assert selection['model_version']==best['version'] and selection['continued_version']==continued['version']
    assert not selection['test_inspected_for_selection']
    for n in ('ranker.model','monte_carlo.joblib','policy.json'):
        assert sha(root/'continued_model'/n)==sha(root/f"candidate_{continued['version']}"/n)
    read=lambda p:sorted([json.loads(s) for s in (root/p/'results.jsonl').read_text().splitlines()],key=lambda r:r['game_index'])
    report=json.loads((root/'report.json').read_text());assert paired_stats(read('baseline_test'),read('candidate_test'))==report['paired']
    assert not report['test_used_for_training_or_selection']
    audit={'status':'passed','actual_training_games':100,'actual_decision_groups':len(metadata),'actor_candidate_rows':len(raw),
          'behavior_probabilities_verified':True,'max_behavior_probability_error':probability_error,
          'all_actual_return_advantage_targets_verified':True,'added_trees_audited':len(trees)-base,
          'added_leaves_audited':leaves_checked,'max_added_leaf_output_error':leaf_error,
          'sampled_raw_training_transitions':sampled_raw,'decisions_rescored':decisions,'candidates_rescored':candidates,
          'max_score_error':score_error,'critic_frozen':True,'original114trees_unchanged':True,
          'test_not_used_for_fitting_or_checkpoint_selection':True,'source_unchanged':True,
          'elapsed_seconds':perf_counter()-started,'scope':'all rewards/likelihoods/advantages/added leaves; sampled feature/scoring checks'}
    write(root/'model_audit.json',audit);print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
