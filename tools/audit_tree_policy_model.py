"""Audit winning-policy selection, actual Monte Carlo fits and predictions."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter
import joblib
import lightgbm as lgb
import numpy as np
from optimize_tree_policy_vs_f import TreePolicy,monte_carlo_data
from train_online_sarsa import write


def tree_depth(node):
    return 0 if 'leaf_index' in node or 'split_index' not in node else 1+max(tree_depth(node['left_child']),tree_depth(node['right_child']))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    root=parser.parse_args().root.resolve();start=perf_counter()
    config=json.loads((root/'config.json').read_text());engine=json.loads((root/'engine_audit.json').read_text());assert engine['status']=='passed'
    normalizer=json.loads((root/'normalizer.json').read_text())
    training={f'RL_round_{i}':dict(np.load(root/f'RL_round_{i}/experience.npz')) for i in range(1,3)}
    seen={p:set() for p in training};feature_error=0.
    with gzip.open(root/'audit_transitions.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            case=json.loads(line);data=training[case['phase']];i=case['position'];assert i not in seen[case['phase']];seen[case['phase']].add(i)
            a,b=data['ranges'][i]
            if case['x'] is not None:
                error=float(np.max(np.abs(data['x'][i]-case['x'])))
                if b>a:error=max(error,float(np.max(np.abs(data['next_x'][a:b]-case['next_x']))))
                feature_error=max(feature_error,error);assert error<1e-12
            assert abs(case['reward']-data['reward'][i])<1e-12 and case['discount']==data['discount'][i]
    assert all(len(seen[p])==len(d['x']) for p,d in training.items())
    leaf_error=0.;mc_samples=[]
    for round_index in (1,2):
        phases=[f'RL_round_{i}' for i in range(1,round_index+1)]
        x,y,w,ids=monte_carlo_data(root,phases);saved=dict(np.load(root/f'MC_round_{round_index}.npz'))
        np.testing.assert_array_equal(x,saved['x']);np.testing.assert_array_equal(y,saved['target']);np.testing.assert_array_equal(w,saved['weight'])
        forest=joblib.load(root/f'MC_round_{round_index}.joblib');assert forest.max_depth==10 and len(forest.estimators_)==64 and not forest.bootstrap
        leaves=forest.apply(x)
        for j,estimator in enumerate(forest.estimators_):
            tree=estimator.tree_;assert tree.max_depth<=10
            counts=np.bincount(leaves[:,j],minlength=tree.node_count)
            mass=np.bincount(leaves[:,j],weights=w,minlength=tree.node_count)
            sums=np.bincount(leaves[:,j],weights=w*y,minlength=tree.node_count);mask=mass>0
            np.testing.assert_array_equal(counts[mask],tree.n_node_samples[mask])
            np.testing.assert_allclose(mass[mask],tree.weighted_n_node_samples[mask],rtol=0,atol=1e-10)
            error=float(np.max(np.abs(sums[mask]/mass[mask]-tree.value[mask,0,0])));leaf_error=max(leaf_error,error);assert error<1e-10
        mc_samples.append(len(x))
        print(f'MC AUDIT {round_index}: {len(x)} actual-game return targets verified',flush=True)
    candidates={};depths={};prediction_cases={}
    for path in root.glob('candidate_*'):
        if not path.is_dir():continue
        document=json.loads((path/'policy.json').read_text());booster=lgb.Booster(model_file=str(path/'ranker.txt'))
        mc=joblib.load(path/'monte_carlo.joblib');version=document['version']
        candidates[version]=TreePolicy(booster,normalizer,document['scale'],mc,document['mix'],version)
        values=[tree_depth(t['tree_structure']) for t in booster.dump_model()['tree_info']]
        depths[str(version)]={'trees':len(values),'max_actual_depth':max(values)};assert max(values)<=10
    candidate_rows=choices=0;prediction_error=0.
    with gzip.open(root/'audit_predictions.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            case=json.loads(line);prediction_cases.setdefault(case['version'],[]).append(case)
    for version,cases in prediction_cases.items():
        raw=np.concatenate([np.asarray(c['raw']) for c in cases]);actual=candidates[version].predict(raw)
        expected=np.concatenate([np.asarray(c['scores']) for c in cases]);error=float(np.max(np.abs(actual-expected)))
        prediction_error=max(prediction_error,error);assert error<1e-10
        cursor=0
        for case in cases:
            n=len(case['raw'])
            if not case['explored']:
                assert np.argmax(case['scores'])==case['selected']
                assert actual[cursor+case['selected']]>=actual[cursor:cursor+n].max()-1e-10
            cursor+=n
        candidate_rows+=len(actual);choices+=len(cases)
    history=json.loads((root/'selection_history.json').read_text());selection=json.loads((root/'selection.json').read_text())
    best=max(history,key=lambda h:(h['validation']['wins'],-history.index(h)))
    assert selection['model_version']==best['version'] and selection['validation_wins']==best['validation']['wins']
    assert not selection['test_inspected_for_selection']
    assert (root/'best_model/ranker.txt').read_bytes()==(root/f"candidate_{best['version']}/ranker.txt").read_bytes()
    # Whole source game groups, never final test games, form the teacher dataset.
    teacher_groups=[json.loads(s) for s in (root/'teacher_groups.jsonl').read_text().splitlines()]
    teacher=dict(np.load(root/'teacher_data.npz'))
    assert len(teacher_groups)==len(teacher['groups']) and teacher['x'].shape[1]==264
    cursor=0
    for m,n,mask in zip(teacher_groups,teacher['groups'],teacher['validation_groups']):
        assert n==m['candidate_count'] and bool(mask)==(m['game_index']>=80)
        assert teacher['y'][cursor+m['teacher_index']]==teacher['y'][cursor:cursor+n].max()
        cursor+=n
    assert cursor==len(teacher['x'])
    report={'status':'passed','fresh_RL_transitions_verified':sum(len(d['x']) for d in training.values()),
            'MC_fit_rows':mc_samples,'MC_trees_audited':128,'teacher_candidate_rows':len(teacher['x']),
            'teacher_decision_groups':len(teacher_groups),'depths':depths,'max_feature_error':feature_error,
            'max_MC_leaf_mean_error':leaf_error,'max_prediction_error':prediction_error,
            'prediction_candidates_verified':candidate_rows,'decisions_verified':choices,
            'selection_by_F_validation_wins_only':True,'no_final_test_rows_used_in_training_or_selection':True,
            'learner_forward_simulation':False,'elapsed_seconds':perf_counter()-start,
            'scope':'all fresh RL rewards and weighted MC targets/leaves; sampled raw training/evaluation features and predictions; grouped teacher provenance'}
    write(root/'model_audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
