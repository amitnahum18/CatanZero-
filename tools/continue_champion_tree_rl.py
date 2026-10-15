"""Preserve the champion; refit its outcome critic on fresh actual interaction."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
import joblib
import lightgbm as lgb
import catboost as cb
import mlflow
from mlflow.models import infer_signature
import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import ExtraTreesRegressor
from optimize_dagger_policy import Policy, PortablePolicy
from optimize_tree_policy_vs_f import monte_carlo_data
from train_residual_tree_fqi import REPO, play_phase, write
from train_three_game_sarsa import exclusions
from catanzero.duel_protocol import wilson_interval

SOURCE=REPO/'runs/tree-policy-trade-aware-v3'
OLD_MC=REPO/'runs/tree-policy-native-f-max-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy(path,normalizer,version=None):
    doc=json.loads((path/'policy.json').read_text())
    return Policy(lgb.Booster(model_file=str(path/'ranker.model')),doc['kind'],normalizer,
                  doc['scale'],doc['version'] if version is None else version,
                  joblib.load(path/'monte_carlo.joblib'),doc['mix'])


def critic_data(root):
    old_x,old_y,old_w,old_ids=monte_carlo_data(OLD_MC,['RL_round_1','RL_round_2'])
    new_x,new_y,new_w,new_ids=monte_carlo_data(root,['RL_round_1'])
    saved=dict(np.load(OLD_MC/'MC_round_2.npz'))
    for actual,key in ((old_x,'x'),(old_y,'target'),(old_w,'weight')):
        np.testing.assert_array_equal(actual,saved[key])
    ids=[{**m,'source_root':str(OLD_MC)} for m in old_ids]
    ids.extend({**m,'source_root':str(root)} for m in new_ids)
    return np.r_[old_x,new_x],np.r_[old_y,new_y],np.r_[old_w,new_w],ids


def paired_stats(before,after):
    assert len(before)==len(after)
    plus=minus=0
    for a,b in zip(before,after):
        for key in ('game_index','seed','learner_seat','initial_map_hash','initial_stream_hash'):
            assert a[key]==b[key]
        if a['termination']!='victory' or b['termination']!='victory':
            raise ValueError('incomplete comparison: unknown tails must not become losses')
        plus+=bool(b['learner_won'] and not a['learner_won'])
        minus+=bool(a['learner_won'] and not b['learner_won'])
    return {'candidate_only_wins':plus,'baseline_only_wins':minus,
            'two_sided_p':float(binomtest(plus,plus+minus,.5).pvalue) if plus+minus else 1.,
            'paired_games':len(before),'pairing':'same initial map, seat and chance stream; futures may diverge'}


def influence_report(root,models):
    groups=[]
    for index in (0,1,98,99):
        path=root/'RL_round_1'/f'game_{index:04d}'/'decisions.jsonl.gz'
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for row in map(json.loads,stream):
                if row['policy']=='F':continue
                # Feature reconstruction is independently done by the engine audit.
                # The following actual-next-decision candidates are recorded before
                # real actions; no synthetic future rollout is used.
                groups.append((index,row['action_index']))
    data=dict(np.load(root/'RL_round_1/experience.npz'))
    ranges=[(a,b) for a,b in data['ranges'] if b-a>1][:2000]
    raw=np.concatenate([data['next_x'][a:b] for a,b in ranges])
    prior=models[0].teacher.predict(raw,num_threads=1)/models[0].scale
    values={m.version:m.predict(raw) for m in models}
    results={};cursor=0
    for model in models:
        changed=0;cursor=0;mc_margin=[]
        for a,b in ranges:
            n=b-a;changed+=int(np.argmax(values[model.version][cursor:cursor+n])!=np.argmax(prior[cursor:cursor+n]))
            mc_margin.append(float(np.ptp(values[model.version][cursor:cursor+n]-prior[cursor:cursor+n])))
            cursor+=n
        results[str(model.version)]={'multi_action_decisions':len(ranges),'changes_vs_teacher_only':changed,
                                    'change_fraction':changed/len(ranges),'median_MC_score_spread':float(np.median(mc_margin))}
    return {'scope':'first 2000 actual next learner decision sets in new training only',
            'scores_are_calibrated_probabilities':False,'policies':results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=REPO/'runs/tree-policy-rl-continuation-v4')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    normalizer=json.loads((SOURCE/'normalizer.json').read_text());write(root/'normalizer.json',normalizer)
    excluded,manifest=exclusions(REPO/'runs/online-sarsa-features-100-v4')
    source_hashes={name:sha(SOURCE/'best_model'/name) for name in ('ranker.model','monte_carlo.joblib','policy.json')}
    shutil.copytree(SOURCE/'best_model',root/'baseline_model')
    phases=['RL_round_1','validation_candidate_500','validation_candidate_501','validation_candidate_502',
            'baseline_test','candidate_test']
    config={'algorithm':'champion on-policy Monte Carlo ExtraTrees continuation with frozen LightGBM',
            'source':str(SOURCE),'source_hashes':source_hashes,'fresh_training_games':100,
            'phase_plan':phases,'validation_phase':'validation_candidate_500','test_phase':'baseline_test',
            'training_seed':56000000,'validation_seed':57000000,'test_seed':58000000,
            'validation_games':100,'test_games_per_policy':200,'players':2,'opponent':'native F',
            'excluded_board_hashes':excluded,'workers':args.workers,'epsilon':.08,
            'max_depth':10,'trees':{'LightGBM':114,'ExtraTrees':64},'MC_mixes':[.2,.4],
            'reward':'terminal win +1/loss -1, gamma1; Phi=.05 own known VP+.02 public gap; terminal Phi0',
            'MC_target':'actual completed own-game outcome minus CURRENT state Phi',
            'replay':'200 prior actual training games plus 100 current-champion games, equal total weight per game',
            'selection':'one continued model selected by validation only; baseline retained on ties',
            'promotion':'requires independent paired test p<.05, positive win delta, and all audits passed',
            'learner_forward_simulation':False,'test_rows_enter_training':False}
    write(root/'config.json',config)
    mlflow.set_tracking_uri(manifest['tracking_uri']);client=mlflow.MlflowClient()
    client.set_registered_model_alias('CatanZero_TreePolicy_NativeF','baseline_56_4','3')
    mlflow.set_experiment(experiment_id=client.get_experiment_by_name('CatanZero_TreePolicy_NativeF').experiment_id)
    started=perf_counter();baseline=load_policy(root/'baseline_model',normalizer,500)
    baseline.save(root/'candidate_500')
    with mlflow.start_run(run_name='champion_fresh_MC_RL_continuation_100') as run:
        write(root/'tracking.json',{'run_id':run.info.run_id,'tracking_uri':manifest['tracking_uri']})
        mlflow.log_params({'training_games':100,'MC_games_total':300,'opponent':'F','players':2,
                           'max_depth':10,'ranker_frozen':True,'MC_trees':64,'epsilon':.08})
        mlflow.log_artifact(str(root/'config.json'),'protocol_before_training')
        print('STAGE: 100 new real games collected by current champion, epsilon .08',flush=True)
        train_rows,train,_=play_phase(root,'RL_round_1',baseline,100,args.workers,training=True,
                                     base_seed=config['training_seed'],excluded=excluded,epsilon=.08)
        assert train['completed']==100
        fit_start=perf_counter();x,y,w,ids=critic_data(root)
        forest=ExtraTreesRegressor(n_estimators=64,max_depth=10,min_samples_leaf=16,max_features=.75,
                                  n_jobs=4,random_state=20261014)
        forest.fit(x,y,sample_weight=w);forest.n_jobs=1
        np.savez_compressed(root/'MC_update.npz',x=x,target=y,weight=w)
        (root/'MC_update.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in ids),encoding='utf-8')
        joblib.dump(forest,root/'MC_update.joblib',compress=3)
        write(root/'fit.json',{'fit_seconds':perf_counter()-fit_start,'rows':len(y),'new_rows':train['transitions'],
              'games':300,'updates':1,'old_MC_prediction_mean_change':float(np.mean(np.abs(forest.predict(x)-baseline.mc.predict(x))))})
        print(f'FIT: {len(y)} true-outcome labels; {perf_counter()-fit_start:.1f}s',flush=True)
        candidates=[Policy(baseline.teacher,'lightgbm',normalizer,baseline.scale,501+i,forest,mix)
                    for i,mix in enumerate(config['MC_mixes'])]
        history=[];validation_boards=[]
        for model in [baseline]+candidates:
            model.save(root/f'candidate_{model.version}')
            rows,summary,_=play_phase(root,f'validation_candidate_{model.version}',model,100,args.workers,
                 training=False,base_seed=config['validation_seed'],excluded=excluded+[r['initial_map_hash'] for r in train_rows])
            if not validation_boards:validation_boards=[r['initial_map_hash'] for r in rows]
            assert summary['completed']==100
            history.append({'version':model.version,'mix':model.mix,'validation':summary})
            write(root/'selection_history.json',history)
        # Test the strongest continuation even if baseline wins validation, so a
        # negative result is concrete. Do not use the final test to pick mixtures.
        continued=max(candidates,key=lambda m:(history[m.version-500]['validation']['wins'],-m.version))
        best=max([baseline]+candidates,key=lambda m:(history[m.version-500]['validation']['wins'],-m.version))
        continued.save(root/'continued_model');best.save(root/'best_validation_model')
        write(root/'selection.json',{'continued_version':continued.version,'best_validation_version':best.version,
              'model_version':best.version,'validation_wins':history[best.version-500]['validation']['wins'],
              'test_inspected_for_selection':False})
        write(root/'score_diagnostics.json',influence_report(root,[baseline]+candidates))
        print(f'STAGE: frozen independent paired tests; continuation version {continued.version}',flush=True)
        test_excluded=excluded+validation_boards+[r['initial_map_hash'] for r in train_rows]
        before_rows,before,_=play_phase(root,'baseline_test',baseline,200,args.workers,training=False,
                                     base_seed=config['test_seed'],excluded=test_excluded)
        after_rows,after,_=play_phase(root,'candidate_test',continued,200,args.workers,training=False,
                                   base_seed=config['test_seed'],excluded=test_excluded)
        paired=paired_stats(before_rows,after_rows)
        qualifies=after['wins']>before['wins'] and paired['two_sided_p']<.05 and best.version==continued.version
        sample=x[::max(1,len(x)//512)].astype(np.float64);expected=continued.predict(sample)
        np.savez_compressed(root/'isolated_predictions.npz',raw=sample,expected=expected)
        info=mlflow.pyfunc.log_model(name='model',python_model=PortablePolicy(),
             artifacts={'ranker':str(root/'continued_model/ranker.model'),'mc':str(root/'continued_model/monte_carlo.joblib'),
                        'policy':str(root/'continued_model/policy.json')},input_example=sample[:3],
             signature=infer_signature(sample,expected),pip_requirements=['mlflow=='+mlflow.__version__,
             'numpy=='+np.__version__,'lightgbm=='+lgb.__version__,'catboost=='+cb.__version__,
             'scikit-learn==1.7.1','joblib=='+joblib.__version__],
             code_paths=[str(REPO/'tools/optimize_dagger_policy.py'),str(REPO/'tools/optimize_tree_policy_vs_f.py'),
                         str(REPO/'tools/train_residual_tree_fqi.py'),str(REPO/'tools/train_online_sarsa.py'),
                         str(REPO/'tools/train_three_game_sarsa.py'),str(REPO/'src/catanzero')])
        actual=mlflow.pyfunc.load_model(info.model_uri).predict(sample)
        error=float(np.max(np.abs(expected-actual)));assert error<1e-12
        registered=mlflow.register_model(info.model_uri,'CatanZero_TreePolicy_NativeF')
        client.set_registered_model_alias('CatanZero_TreePolicy_NativeF','rl_continuation_pilot',str(registered.version))
        write(root/'registration.json',{'family':'CatanZero_TreePolicy_NativeF','version':str(registered.version),
              'alias':'rl_continuation_pilot','model_uri':info.model_uri,'reload_max_abs_error':error})
        report={'training':train,'baseline_test':before,'continued_test':after,'paired':paired,
                'baseline_CI95':wilson_interval(before['wins'],200),'continued_CI95':wilson_interval(after['wins'],200),
                'qualifies_for_promotion_after_audits':bool(qualifies),'promoted':False,
                'baseline_registry_version':'3','continued_registry_version':str(registered.version),
                'test_used_for_fitting_or_mixture_selection':False,'elapsed_seconds':perf_counter()-started,
                'source_unchanged':all(sha(SOURCE/'best_model'/n)==h for n,h in source_hashes.items())}
        assert report['source_unchanged'];write(root/'report.json',report)
        for name in ('config.json','report.json','selection.json','selection_history.json','fit.json','score_diagnostics.json','registration.json'):
            mlflow.log_artifact(str(root/name),'experiment')
        mlflow.log_metrics({'baseline_test_wins':before['wins'],'continued_test_wins':after['wins'],
                            'paired_p':paired['two_sided_p'],'new_training_games':100,'new_transitions':train['transitions']})
        print('RESULT: '+json.dumps(report),flush=True)


if __name__=='__main__':main()
