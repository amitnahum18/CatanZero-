"""Independently verify continuation targets, leaf values and frozen scoring."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import gzip
import json
from time import perf_counter
from pathlib import Path
import joblib
import numpy as np
from continue_champion_tree_rl import critic_data,load_policy,sha,SOURCE,OLD_MC,paired_stats
from train_residual_tree_fqi import write
from audit_tree_policy_model import tree_depth


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    root=parser.parse_args().root.resolve();started=perf_counter()
    config=json.loads((root/'config.json').read_text());normalizer=json.loads((root/'normalizer.json').read_text())
    engine=json.loads((root/'engine_audit.json').read_text());assert engine['status']=='passed'
    assert engine['games']==800 and engine['unique_training_boards']==100
    assert json.loads((OLD_MC/'model_audit.json').read_text())['status']=='passed'
    x,y,w,ids=critic_data(root);saved=dict(np.load(root/'MC_update.npz'))
    for actual,key in ((x,'x'),(y,'target'),(w,'weight')):np.testing.assert_array_equal(actual,saved[key])
    metadata=[json.loads(s) for s in (root/'MC_update.jsonl').read_text().splitlines()];assert metadata==ids
    games={}
    for m,weight in zip(ids,w):
        key=(m['source_root'],m['phase'],m['game_index']);games[key]=games.get(key,0)+weight
        assert m['phase'] in ('RL_round_1','RL_round_2')
    assert len(games)==300 and max(abs(v-1) for v in games.values())<1e-12
    # Independently sum the recorded causal rewards from each current action to
    # termination, instead of trusting the direct outcome-minus-potential label.
    data=dict(np.load(root/'RL_round_1/experience.npz'))
    transitions=[json.loads(s) for s in (root/'RL_round_1/transitions.jsonl').read_text().splitlines()]
    per_game={}
    for i,m in enumerate(transitions):per_game.setdefault(m['game_index'],[]).append(i)
    returns=np.empty(len(transitions))
    for indexes in per_game.values():
        assert transitions[indexes[-1]]['terminal'];value=0.
        for i in reversed(indexes):
            value=data['reward'][i]+data['discount'][i]*value;returns[i]=value
    np.testing.assert_allclose(returns,y[len(y)-len(returns):],rtol=0,atol=1e-10)
    raw_verified=0;positions=set();max_feature_error=0.
    with gzip.open(root/'audit_transitions.jsonl.gz','rt',encoding='utf-8') as stream:
        for case in map(json.loads,stream):
            assert case['phase']=='RL_round_1';i=case['position'];assert i not in positions;positions.add(i)
            assert abs(data['reward'][i]-case['reward'])<1e-12 and data['discount'][i]==case['discount']
            if case['x'] is not None:
                a,b=data['ranges'][i];error=float(np.max(np.abs(data['x'][i]-case['x'])))
                if b>a:error=max(error,float(np.max(np.abs(data['next_x'][a:b]-case['next_x']))))
                assert error<1e-12;max_feature_error=max(max_feature_error,error);raw_verified+=1
    assert len(positions)==len(transitions)
    forest=joblib.load(root/'MC_update.joblib');assert len(forest.estimators_)==64 and not forest.bootstrap
    leaf_error=0.;leaves=forest.apply(x)
    for j,estimator in enumerate(forest.estimators_):
        tree=estimator.tree_;assert tree.max_depth<=10
        counts=np.bincount(leaves[:,j],minlength=tree.node_count)
        mass=np.bincount(leaves[:,j],weights=w,minlength=tree.node_count)
        sums=np.bincount(leaves[:,j],weights=w*y,minlength=tree.node_count);mask=mass>0
        np.testing.assert_array_equal(counts[mask],tree.n_node_samples[mask])
        np.testing.assert_allclose(mass[mask],tree.weighted_n_node_samples[mask],rtol=0,atol=1e-10)
        error=float(np.max(np.abs(sums[mask]/mass[mask]-tree.value[mask,0,0])));assert error<1e-10
        leaf_error=max(leaf_error,error)
    models={};depths={}
    for version in (500,501,502):
        path=root/f'candidate_{version}';model=load_policy(path,normalizer);models[version]=model
        ds=[tree_depth(t['tree_structure']) for t in model.teacher.dump_model()['tree_info']]
        assert len(ds)==114 and max(ds)<=10
        assert sha(path/'ranker.model')==sha(root/'candidate_500/ranker.model')
        assert len(model.mc.estimators_)==64 and max(t.tree_.max_depth for t in model.mc.estimators_)<=10
        depths[str(version)]={'LightGBM_trees':114,'ExtraTrees_trees':64,'max_depth':10}
        if version!=500:np.testing.assert_array_equal(model.mc.predict(x[::100]),forest.predict(x[::100]))
    groups={};score_error=0.;decisions=candidates=0
    with gzip.open(root/'audit_predictions.jsonl.gz','rt',encoding='utf-8') as stream:
        for case in map(json.loads,stream):groups.setdefault(case['version'],[]).append(case)
    phi_i=normalizer['feature_names'].index('own_actual_vp')
    enemy_i=normalizer['feature_names'].index('opponent_1_VICTORY_POINTS')
    for version,cases in groups.items():
        raw=np.concatenate([np.asarray(c['raw']) for c in cases]);actual=models[version].predict(raw)
        expected=np.concatenate([np.asarray(c['scores']) for c in cases])
        error=float(np.max(np.abs(actual-expected)));assert error<1e-10;score_error=max(score_error,error)
        cursor=0
        for c in cases:
            n=len(c['raw']);state=np.asarray(c['raw'])
            assert np.ptp(state[:,phi_i])==np.ptp(state[:,enemy_i])==0
            if not c['explored']:
                assert np.argmax(c['scores'])==c['selected']
                assert actual[cursor+c['selected']]>=actual[cursor:cursor+n].max()-1e-10
            cursor+=n
        decisions+=len(cases);candidates+=len(raw)
    history=json.loads((root/'selection_history.json').read_text());selection=json.loads((root/'selection.json').read_text())
    best=max(history,key=lambda h:(h['validation']['wins'],-h['version']))
    continued=max(history[1:],key=lambda h:(h['validation']['wins'],-h['version']))
    assert selection['model_version']==best['version'] and selection['continued_version']==continued['version']
    assert not selection['test_inspected_for_selection']
    for name in ('ranker.model','monte_carlo.joblib','policy.json'):
        assert sha(root/'continued_model'/name)==sha(root/f"candidate_{continued['version']}"/name)
        assert sha(SOURCE/'best_model'/name)==config['source_hashes'][name]
        assert sha(root/'baseline_model'/name)==config['source_hashes'][name]
    read_rows=lambda p:sorted([json.loads(s) for s in (root/p/'results.jsonl').read_text().splitlines()],key=lambda r:r['game_index'])
    reference=read_rows(config['validation_phase'])
    for version in (501,502):
        for a,b in zip(reference,read_rows(f'validation_candidate_{version}')):
            assert all(a[k]==b[k] for k in ('seed','initial_map_hash','initial_stream_hash','learner_seat'))
    report=json.loads((root/'report.json').read_text())
    assert paired_stats(read_rows('baseline_test'),read_rows('candidate_test'))==report['paired']
    assert not report['test_used_for_fitting_or_mixture_selection']
    audit={'status':'passed','MC_games':len(games),'MC_training_rows':len(y),'all_new_reward_transitions':len(transitions),
           'all_MC_leaf_values_verified':True,'max_leaf_value_error':leaf_error,
           'sampled_raw_training_transitions':raw_verified,'max_raw_feature_error':max_feature_error,
           'decisions_rescored':decisions,'candidates_rescored':candidates,'max_score_error':score_error,
           'source_model_unchanged':True,'ranker_unchanged':True,'validation_selection_only':True,
           'reward_potential_is_state_only':True,'depths':depths,'elapsed_seconds':perf_counter()-started,
           'scope':'all new causal rewards/MC targets/weighted leaves, all engine games, sampled raw features/scores'}
    write(root/'model_audit.json',audit);print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
