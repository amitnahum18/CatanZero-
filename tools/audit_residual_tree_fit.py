"""Verify every tree's targets and leaves from audited real training inputs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
from train_residual_tree_fqi import ResidualTrees,combine
from train_online_sarsa import write


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    root=parser.parse_args().root.resolve();start=perf_counter()
    config=json.loads((root/'config.json').read_text());base=json.loads((root/'base_model.json').read_text())
    norm=json.loads((root/'normalizer.json').read_text());model=ResidualTrees(base,norm)
    engine=json.loads((root/'engine_audit.json').read_text());assert engine['status']=='passed'
    datasets={f'training_round_{i}':dict(np.load(root/f'training_round_{i}/experience.npz')) for i in range(1,config['rounds']+1)}
    seen={p:set() for p in datasets};max_feature_error=max_target_error=max_leaf_error=max_prediction_error=0.
    with gzip.open(root/'audit_transitions.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            case=json.loads(line);data=datasets[case['phase']];i=case['position'];assert i not in seen[case['phase']]
            seen[case['phase']].add(i)
            a,b=data['ranges'][i]
            assert b-a==len(case['next_x'])
            x=np.asarray(case['x']);nx=np.asarray(case['next_x']).reshape(-1,264)
            error=max(float(np.max(np.abs(data['x'][i]-x))),float(np.max(np.abs(data['next_x'][a:b]-nx))) if b>a else 0.)
            max_feature_error=max(max_feature_error,error);assert error<1e-12
            assert abs(data['reward'][i]-case['reward'])<1e-12 and data['discount'][i]==case['discount']
    assert all(len(seen[p])==len(data['x']) for p,data in datasets.items())
    fits=[];data_sets=[];ver=0
    for round_index in range(1,config['rounds']+1):
        data_sets.append(datasets[f'training_round_{round_index}']);data=combine(data_sets)
        np.testing.assert_allclose(data['phi'],model.phi(data['x']),atol=1e-12,rtol=0)
        for iteration in range(config['iterations_per_round']):
            ver+=1;directory=root/f'update_{ver:03d}';saved=dict(np.load(directory/'targets.npz'))
            q=model.predict(data['next_x'])
            future=np.asarray([q[a:b].max() if b>a else 0. for a,b in data['ranges']])
            future=np.clip(future,-1-data['next_phi'],1-data['next_phi'])
            targets=np.clip(data['reward']+data['discount']*future,-1-data['phi'],1-data['phi'])
            old=model.predict(data['x'])
            residual=(1-config['damping'])*old+config['damping']*targets-model.anchor(data['x'])
            for key,expected in [('target',targets),('previous_q',old),('residual_target',residual)]:
                error=float(np.max(np.abs(saved[key]-expected)));max_target_error=max(max_target_error,error);assert error<1e-12
            forest=joblib.load(directory/'forest.joblib')
            assert forest.max_depth==10 and len(forest.estimators_)==64 and not forest.bootstrap
            assert forest.random_state==2026+ver
            normalized=model.normalize(data['x'])
            leaves=forest.apply(normalized)
            for tree_index,estimator in enumerate(forest.estimators_):
                tree=estimator.tree_;idx=leaves[:,tree_index]
                counts=np.bincount(idx,minlength=tree.node_count)
                sums=np.bincount(idx,weights=residual,minlength=tree.node_count)
                mask=counts>0;means=sums[mask]/counts[mask]
                np.testing.assert_array_equal(counts[mask],tree.n_node_samples[mask])
                error=float(np.max(np.abs(means-tree.value[mask,0,0])));max_leaf_error=max(max_leaf_error,error);assert error<1e-10
                assert tree.max_depth<=10
            model.forest=forest;model.version=ver
            fits.append({'version':ver,'training_rows':len(data['x']),'all_tree_leaf_means_verified':True})
            print(f'FIT AUDIT {ver}: {len(data["x"])} real training rows',flush=True)
    caches={0:ResidualTrees(base,norm)}
    candidate_counts=0;choices=0
    batches={}
    with gzip.open(root/'audit_predictions.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            case=json.loads(line);version=case['version']
            raw=np.asarray(case['raw']);scores=np.asarray(case['scores'])
            batches.setdefault(version,[]).append((raw,scores,case['selected'],case['explored']))
    for version,cases in batches.items():
        if version not in caches:caches[version]=ResidualTrees(base,norm,joblib.load(root/f'update_{version:03d}/forest.joblib'),version)
        matrices=[c[0] for c in cases];expected=np.concatenate([c[1] for c in cases]);raw=np.concatenate(matrices)
        actual=caches[version].predict(raw)
        error=float(np.max(np.abs(actual-expected)));max_prediction_error=max(max_prediction_error,error);assert error<1e-10
        cursor=0
        for matrix,scores,selected,explored in cases:
            if not explored:
                assert int(np.argmax(scores))==selected
                local=actual[cursor:cursor+len(matrix)]
                assert local[selected]>=local.max()-1e-10
            cursor+=len(matrix)
        candidate_counts+=len(actual);choices+=len(cases)
    selection=json.loads((root/'selection.json').read_text());validations=selection['tree_validation']
    best=max(range(len(validations)),key=lambda i:(validations[i]['wins'],-i))
    assert selection['selected_round']==best+1 and selection['model_version']==validations[best]['model_version']
    assert (root/'forest.joblib').read_bytes()==(root/f"update_{selection['model_version']:03d}/forest.joblib").read_bytes()
    for name,digest in config['source_hashes'].items():assert hashlib.sha256((Path(config['source'])/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((root/'normalizer.json').read_bytes()).hexdigest()==config['source_hashes']['normalizer.json']
    report={'status':'passed','updates_verified':len(fits),'training_transitions_verified':sum(len(d['x']) for d in datasets.values()),
            'trees_checked':len(fits)*64,'prediction_candidates_verified':candidate_counts,'decisions_verified':choices,
            'max_feature_error':max_feature_error,'max_target_error':max_target_error,'max_leaf_mean_error':max_leaf_error,
            'max_prediction_error':max_prediction_error,'source_model_unchanged':True,'normalizer_unchanged':True,
            'validation_only_selection':True,'no_validation_or_test_rows_used_in_fitting':True,
            'scope':'all causal training transitions, all Bellman/EMA residual targets, all fitted leaf counts/means; predictions on all training decisions and sampled frozen evaluations',
            'elapsed_seconds':perf_counter()-start}
    write(root/'fit_audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
