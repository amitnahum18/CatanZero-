"""Recompute all SGD updates and independently check episode-update cadence."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np


def audit(root):
    config=json.loads((root/'config.json').read_text())
    norm=json.loads((root/'normalizer.json').read_text())
    source=Path(config['resume_from'])
    assert norm==json.loads((source/'normalizer.json').read_text())
    assert hashlib.sha256((source/'model.joblib').read_bytes()).hexdigest()==config['source_model_sha256']
    model=joblib.load(root/'initial_model.joblib')
    old=joblib.load(source/'model.joblib')
    np.testing.assert_array_equal(model.coef_,old.coef_)
    np.testing.assert_array_equal(model.intercept_,old.intercept_)
    assert model.t_==old.t_
    batches=json.loads((root/'update_game_batches.json').read_text())
    assert len(batches)==config['planned_updates']
    finished=set()
    for version,batch in enumerate(batches,1):
        assert batch['update_version']==version and batch['completed_games']==3*version
        assert len(batch['newly_finished_games'])==3
        for item in batch['newly_finished_games']:
            assert item['game_index'] not in finished
            finished.add(item['game_index'])
    assert finished==set(range(config['training_games']))
    results={json.loads(s)['game_index']:json.loads(s) for s in (root/'live_training/results.jsonl').read_text().splitlines()}
    assert set(results)==finished
    for batch in batches:
        for item in batch['newly_finished_games']:
            assert item['censored']==(results[item['game_index']]['termination']!='victory')
    # Build independent predecision provenance from actual logged actions.
    actions={}
    for game,result in results.items():
        with gzip.open(root/'live_training'/f'game_{game:04d}'/'decisions.jsonl.gz','rt',encoding='utf-8') as stream:
            own_rows=[json.loads(s) for s in stream]
        for row in own_rows:
            if row['policy']==result['learner_policy']:
                obs=row['inputs']['observation']
                assert obs['actor']==result['learner_color']
                own=next(p for p in obs['players'] if p['color']==obs['actor'])
                actions[(game,row['action_index'])]={'actor':obs['actor'],'vp':own['actual_vp'],
                    'model_version':row['model_version'],'selected_index':row['inputs']['legal_actions'].index(row['inputs']['action'])}
    next_indices={}
    for game in results:
        indices=sorted(i for g,i in actions if g==game)
        next_indices.update({(game,i):(indices[n+1] if n+1<len(indices) else results[game]['actions']) for n,i in enumerate(indices)})
    seen=set();count=0
    with gzip.open(root/'updates.jsonl.gz','rt',encoding='utf-8') as stream:
        for version,line in enumerate(stream,1):
            update=json.loads(line)
            assert update['version']==version and update['transitions']==batches[version-1]['transitions']
            samples=update['samples']
            x=np.asarray([s['x'] for s in samples]);nxt=np.asarray([s['next_x'] for s in samples])
            assert x.shape==nxt.shape and x.shape[1]==len(norm['offset']) and np.isfinite(x).all() and np.isfinite(nxt).all()
            assert np.max(np.abs(x))<=4 and np.max(np.abs(nxt))<=4
            for s in samples:
                key=(s['game_index'],s['action_index'])
                assert key not in seen and s['phase']=='live_training'
                seen.add(key)
                logged=actions[key]
                assert logged['actor']==s['actor'] and logged['selected_index']==s['selected_index']
                assert logged['model_version']==s['action_model_version'] and 0<=s['action_model_version']<version
                ni=s['next_action_index'];assert ni==next_indices[key]
                terminal=ni==results[key[0]]['actions']
                assert terminal==s['terminal']
                gap=ni-key[1]
                discount=0 if terminal else config['gamma']**gap
                if terminal:
                    assert results[key[0]]['termination']=='victory'
                    np.testing.assert_array_equal(s['next_x'],np.zeros(len(norm['offset'])))
                    reward=config['gamma']**(gap-1)*float(results[key[0]]['learner_won'])-config['vp_weight']*logged['vp']
                else:
                    following=actions[(key[0],ni)]
                    assert following['actor']==s['actor'] and following['selected_index']==s['next_selected_index']
                    assert following['model_version']==s['next_action_model_version'] and 0<=s['next_action_model_version']<version
                    reward=discount*config['vp_weight']*following['vp']-config['vp_weight']*logged['vp']
                assert abs(reward-s['reward'])<1e-12 and abs(discount-s['discount'])<1e-12
            target=np.asarray([s['reward'] for s in samples])+np.asarray([s['discount'] for s in samples])*model.predict(nxt)
            np.testing.assert_allclose(target,[s['target'] for s in samples],atol=1e-12,rtol=0)
            model.partial_fit(x,target,sample_weight=1/(1+np.sum(x*x,axis=1)))
            saved=json.loads((root/'snapshots'/f'version_{version:06d}.json').read_text())
            np.testing.assert_allclose(model.coef_,saved['weights'],atol=1e-12,rtol=0)
            assert abs(model.intercept_[0]-saved['intercept'])<1e-12
            count+=len(samples)
    assert version==config['planned_updates'] and count==sum(b['transitions'] for b in batches)
    final=joblib.load(root/'model.joblib')
    np.testing.assert_array_equal(model.coef_,final.coef_)
    np.testing.assert_array_equal(model.intercept_,final.intercept_)
    assert model.t_==final.t_
    all_phases={phase:[json.loads(s) for s in (root/phase/'results.jsonl').read_text().splitlines()]
        for phase in ('initial_evaluation','live_training','final_evaluation')}
    boards={k:{r['initial_map_hash'] for r in v} for k,v in all_phases.items()}
    assert not boards['live_training']&boards['initial_evaluation'] and boards['initial_evaluation']==boards['final_evaluation']
    excluded=set(config['excluded_board_hashes'])
    assert not (boards['live_training']|boards['initial_evaluation'])&excluded
    assert len(boards['live_training'])==config['training_games']
    decision_counts={}
    for phase in ('initial_evaluation','final_evaluation'):
        expected=0 if phase=='initial_evaluation' else version
        decision_counts[phase]=0
        for result in all_phases[phase]:
            with gzip.open(root/phase/f"game_{result['game_index']:04d}"/'decisions.jsonl.gz','rt',encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line)
                    if row['policy']==result['learner_policy']:
                        assert row['model_version']==expected and not row['explored']
                        decision_counts[phase]+=1
    report={'passed':True,'updates_recomputed':version,'training_transitions_verified':count,
        'exactly_three_finished_games_per_update':True,'each_sample_used_once':True,
        'rewards_actual_next_actions_and_targets_verified':True,'final_weights_and_optimizer_exact_match':True,
        'source_model_unchanged':True,'normalization_unchanged':True,'evaluation_samples_used_in_updates':0,
        'all_frozen_evaluation_versions_checked':decision_counts,'train_evaluation_boards_disjoint':True,
        'scope':'All updates and action provenance; raw feature encoding and engine states independently checked by separate sampled game replay.'}
    (root/'update_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    audit(parser.parse_args().root)
