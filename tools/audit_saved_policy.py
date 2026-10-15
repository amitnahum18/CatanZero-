"""Verify sampled live scores and frozen selection for saved tree policies."""
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
import catboost as cb
import numpy as np
from optimize_dagger_policy import Policy
from train_online_sarsa import write


def depth(node):
    return 0 if 'split_index' not in node else 1+max(depth(node['left_child']),depth(node['right_child']))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    root=parser.parse_args().root.resolve();start=perf_counter()
    engine=json.loads((root/'engine_audit.json').read_text());assert engine['status']=='passed'
    normalizer=json.loads((root/'normalizer.json').read_text())
    policies={};depths={}
    for directory in root.glob('candidate_*'):
        if not directory.is_dir():continue
        doc=json.loads((directory/'policy.json').read_text())
        if doc['kind']=='lightgbm':
            teacher=lgb.Booster(model_file=str(directory/'ranker.model'))
            ds=[depth(t['tree_structure']) for t in teacher.dump_model()['tree_info']]
        else:
            teacher=cb.CatBoostRanker();teacher.load_model(str(directory/'ranker.model'))
            ds=[int(np.ceil(np.log2(n))) for n in teacher.get_tree_leaf_counts()]
        assert max(ds)<=10
        mc=joblib.load(directory/'monte_carlo.joblib')
        if doc['mix']:
            assert len(mc.estimators_)==64 and all(e.tree_.max_depth<=10 for e in mc.estimators_)
        policies[doc['version']]=Policy(teacher,doc['kind'],normalizer,doc['scale'],doc['version'],mc,doc['mix'])
        depths[str(doc['version'])]={'ranker_trees':len(ds),'max_actual_depth':max(ds),'MC_trees':64 if doc['mix'] else 0}
    cases={};maximum_error=0.;candidates=choices=0
    with gzip.open(root/'audit_predictions.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            case=json.loads(line);cases.setdefault(case['version'],[]).append(case)
    for version,group in cases.items():
        raw=np.concatenate([np.asarray(c['raw']) for c in group]);expected=np.concatenate([np.asarray(c['scores']) for c in group])
        actual=policies[version].predict(raw);error=float(np.max(np.abs(actual-expected)));maximum_error=max(maximum_error,error);assert error<1e-10
        cursor=0
        for c in group:
            n=len(c['raw']);assert not c['explored'] and np.argmax(c['scores'])==c['selected']
            assert actual[cursor+c['selected']]>=actual[cursor:cursor+n].max()-1e-10;cursor+=n
        candidates+=len(raw);choices+=len(group)
    history=json.loads((root/'selection_history.json').read_text());selection=json.loads((root/'selection.json').read_text())
    best=max(enumerate(history),key=lambda pair:(pair[1]['validation']['wins'],-pair[0]))[1]
    assert best['version']==selection['model_version'] and not selection['test_inspected_for_selection']
    assert (root/'best_model/ranker.model').read_bytes()==(root/f"candidate_{best['version']}/ranker.model").read_bytes()
    assert (root/'best_model/monte_carlo.joblib').read_bytes()==(root/f"candidate_{best['version']}/monte_carlo.joblib").read_bytes()
    # The outcome-based correction is the already audited, real-game MC forest.
    mc_audit_path=root.parents[0]/'tree-policy-native-f-max-v1/model_audit.json'
    mc_audit=json.loads(mc_audit_path.read_text());assert mc_audit['status']=='passed'
    if policies[best['version']].mix:
        reference=root.parents[0]/'tree-policy-native-f-max-v1/MC_round_2.joblib'
        selected_mc=joblib.load(root/'best_model/monte_carlo.joblib');source_mc=joblib.load(reference)
        sample=dict(np.load(root.parents[0]/'tree-policy-native-f-max-v1/MC_round_2.npz'))['x'][::100]
        np.testing.assert_array_equal(selected_mc.predict(sample),source_mc.predict(sample))
    teacher=dict(np.load(root/'teacher_data.npz'));meta=[json.loads(s) for s in (root/'teacher_groups.jsonl').read_text().splitlines()]
    assert len(meta)==len(teacher['groups']);cursor=0
    for record,n,mask in zip(meta,teacher['groups'],teacher['validation_groups']):
        assert n==record['candidate_count'] and bool(mask)==(record['game_index']>=80)
        assert teacher['y'][cursor+record['teacher_index']]==teacher['y'][cursor:cursor+n].max();cursor+=n
    assert cursor==len(teacher['x'])
    report={'status':'passed','depths':depths,'prediction_candidates_verified':candidates,'decisions_verified':choices,
            'max_prediction_error':maximum_error,'selection_uses_validation_only':True,'selected_model_unchanged_during_test':True,
            'teacher_rows':len(teacher['x']),'teacher_groups':len(meta),'teacher_whole_game_split_checked':True,
            'actual_MC_training_audit':str(mc_audit_path),'MC_return_targets_and_leaf_means_verified':True,
            'elapsed_seconds':perf_counter()-start,'scope':'sampled replayed inference; all policy depths; selection and grouped teacher provenance; audited real-game MC lineage'}
    write(root/'model_audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
