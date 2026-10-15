"""Independent audit of low-rate SARSA and public relative-point rewards."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def own_rows(root,phase,result):
    marker='"policy": '+json.dumps(result['learner_policy'])
    count=0
    with gzip.open(root/phase/f"game_{result['game_index']:04d}"/'decisions.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            # Logs use the established JSON serializer. Verify the policy and
            # decision count after this inexpensive filter; no labels used.
            if marker not in line:continue
            row=json.loads(line)
            if row['policy']!=result['learner_policy']:continue
            count+=1
            yield row
    assert count==result['learner_decisions']


def audit(root):
    config=read(root/'config.json');norm=read(root/'normalizer.json')
    source=Path(config['resume_from'])
    assert norm==read(source/'normalizer.json')
    assert hashlib.sha256((source/'model.joblib').read_bytes()).hexdigest()==config['source_model_sha256']
    estimator=joblib.load(root/'initial_model.joblib');old=joblib.load(source/'model.joblib')
    np.testing.assert_array_equal(estimator.coef_,old.coef_)
    np.testing.assert_array_equal(estimator.intercept_,old.intercept_)
    assert estimator.t_==old.t_==config['initial_optimizer_t']
    assert estimator.get_params()==config['estimator_parameters'] and estimator.eta0==config['eta0']
    assert {k for k,v in old.get_params().items() if estimator.get_params()[k]!=v}=={'eta0'}
    phases=('initial_evaluation','live_training','validation_before','validation_50','validation_100','final_evaluation')
    results={p:[json.loads(s) for s in (root/p/'results.jsonl').read_text().splitlines()] for p in phases}
    training={r['game_index']:r for r in results['live_training']}
    actions={};next_indices={}
    for number,(game,result) in enumerate(training.items(),1):
        indices=[]
        for row in own_rows(root,'live_training',result):
            obs=row['inputs']['observation'];actor=obs['actor']
            assert actor==result['learner_color']
            own=next(p for p in obs['players'] if p['color']==actor)
            others=[p for p in obs['players'] if p['color']!=actor]
            assert all(not set(p)&{'resources','development_cards','actual_vp'} for p in others)
            leader=max(p['VICTORY_POINTS'] for p in others)
            phi=config['vp_weight']*own['actual_vp']+config['gap_weight']*(own['actual_vp']-leader)
            key=(game,row['action_index']);assert key not in actions
            actions[key]={'actor':actor,'phi':phi,'model_version':row['model_version'],
                'selected_index':row['inputs']['legal_actions'].index(row['inputs']['action'])}
            indices.append(row['action_index'])
        assert indices==sorted(indices)
        next_indices.update({(game,index):(indices[n+1] if n+1<len(indices) else result['actions']) for n,index in enumerate(indices)})
        if number%50==0:print(f'AUDIT: {number}/100 training trajectories read',flush=True)
    seen=set();count=0;maximum_error=0.
    with gzip.open(root/'updates.jsonl.gz','rt',encoding='utf-8') as stream:
        for version,line in enumerate(stream,1):
            update=json.loads(line);assert update['version']==version
            samples=update['samples'];assert len(samples)==update['transitions']
            x=np.asarray([s['x'] for s in samples]);next_x=np.asarray([s['next_x'] for s in samples])
            assert x.shape==next_x.shape and x.shape[1]==len(norm['offset']) and np.isfinite(x).all() and np.isfinite(next_x).all()
            assert np.max(np.abs(x))<=4 and np.max(np.abs(next_x))<=4
            for sample in samples:
                key=(sample['game_index'],sample['action_index']);assert key not in seen and sample['phase']=='live_training'
                seen.add(key);actual=actions[key]
                assert actual['actor']==sample['actor'] and actual['selected_index']==sample['selected_index']
                assert actual['model_version']==sample['action_model_version'] and 0<=sample['action_model_version']<version
                following=sample['next_action_index'];assert following==next_indices[key]
                terminal=following==training[key[0]]['actions'];assert terminal==sample['terminal']
                gap=following-key[1];assert gap>0
                discount=0. if terminal else config['gamma']**gap
                if terminal:
                    assert training[key[0]]['termination']=='victory'
                    reward=config['gamma']**(gap-1)*float(training[key[0]]['learner_won'])-actual['phi']
                    np.testing.assert_array_equal(sample['next_x'],np.zeros(len(norm['offset'])))
                else:
                    nxt=actions[(key[0],following)]
                    assert nxt['actor']==sample['actor'] and nxt['selected_index']==sample['next_selected_index']
                    assert nxt['model_version']==sample['next_action_model_version'] and 0<=sample['next_action_model_version']<version
                    reward=discount*nxt['phi']-actual['phi']
                assert abs(sample['reward']-reward)<1e-12 and abs(sample['discount']-discount)<1e-12
            targets=np.asarray([s['reward'] for s in samples])+np.asarray([s['discount'] for s in samples])*estimator.predict(next_x)
            np.testing.assert_allclose(targets,[s['target'] for s in samples],atol=1e-12,rtol=0)
            estimator.partial_fit(x,targets,sample_weight=1/(1+np.sum(x*x,axis=1)))
            snapshot=read(root/'snapshots'/f'version_{version:06d}.json')
            error=float(np.max(np.abs(estimator.coef_-snapshot['weights'])));maximum_error=max(maximum_error,error)
            assert error<1e-12 and abs(estimator.intercept_[0]-snapshot['intercept'])<1e-12
            count+=len(samples)
    report=read(root/'report.json')
    assert version==report['central_updates'] and count==report['fresh_training_transitions']
    assert seen==set(actions)
    final=joblib.load(root/'model.joblib')
    np.testing.assert_allclose(estimator.coef_,final.coef_,atol=1e-12,rtol=0)
    np.testing.assert_allclose(estimator.intercept_,final.intercept_,atol=1e-12,rtol=0)
    assert estimator.t_==final.t_
    boards={p:{r['initial_map_hash'] for r in rows} for p,rows in results.items()}
    assert len(boards['live_training'])==100 and len(boards['initial_evaluation'])==config['evaluation_games']
    assert boards['initial_evaluation']==boards['final_evaluation']
    assert boards['validation_before']==boards['validation_50']==boards['validation_100']
    assert not boards['live_training']&(boards['initial_evaluation']|boards['validation_before'])
    assert not boards['initial_evaluation']&boards['validation_before']
    assert not (boards['live_training']|boards['initial_evaluation']|boards['validation_before'])&set(config['excluded_board_hashes'])
    versions={'initial_evaluation':0,'validation_before':0,'validation_50':read(root/'checkpoint_50/linear_model.json')['model_version'],
        'validation_100':version,'final_evaluation':version}
    decisions={}
    for phase,expected in versions.items():
        decisions[phase]=0
        for result in results[phase]:
            for row in own_rows(root,phase,result):
                assert row['model_version']==expected and not row['explored']
                decisions[phase]+=1
    verification={'passed':True,'updates_recomputed':version,'training_transitions_verified':count,
        'public_opponent_points_only':True,'all_relative_rewards_and_targets_recomputed':True,
        'each_sample_used_once':True,'evaluation_and_validation_samples_in_updates':0,
        'train_validation_test_boards_disjoint':True,'starting_weights_and_optimizer_preserved':True,
        'only_estimator_parameter_changed':'eta0','learning_rate':estimator.eta0,
        'normalization_unchanged':True,'snapshot_max_abs_weight_error':maximum_error,
        'final_optimizer_exact_match':True,'all_frozen_decisions_checked':decisions,
        'scope':'All updates and action provenance; independent feature/state reconstruction covered by sampled engine replay.'}
    (root/'update_audit.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verification),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    audit(parser.parse_args().root)
