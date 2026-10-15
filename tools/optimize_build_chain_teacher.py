"""Distill stronger deterministic build chains; maximize actual wins against F."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
from concurrent.futures import ThreadPoolExecutor,as_completed
import gzip
import json
from pathlib import Path
import pickle
import subprocess
from time import perf_counter
import numpy as np
import lightgbm as lgb
import catboost as cb
import mlflow
from mlflow.models import infer_signature
from continue_champion_tree_rl import SOURCE,load_policy,sha,paired_stats
from optimize_dagger_policy import Policy,PortablePolicy
from optimize_tree_policy_vs_f import fit_ranker
from train_residual_tree_fqi import REPO,play_phase,write
from train_three_game_sarsa import exclusions
from catanzero.duel_protocol import wilson_interval

ROOT=REPO/'runs/tree-policy-build-chain-v6'
SOURCES=[REPO/'runs/residual-extratrees-depth10-f-duel-v1'/f'training_round_{i}' for i in (1,2,3)]
SOURCES+=[REPO/'runs/tree-policy-native-f-max-v1'/f'RL_round_{i}' for i in (1,2)]
SOURCES+=[REPO/'runs/tree-policy-actor-rl-v5/RL_round_1']


def label_job(batch,index):
    source=SOURCES[batch-1]/f'game_{index:04d}';output=ROOT/'labels'/f'batch_{batch}_game_{index:04d}'
    with (ROOT/'labels'/f'worker_{batch}_{index:04d}.log').open('w',encoding='utf-8') as log:
        subprocess.run([str(REPO/'.venv/Scripts/python.exe'),'-u','-m','catanzero.label_f_policy',
              '--source',str(source),'--output',str(output),'--trade-followup','--build-followup'],
              cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240,
              env={**os.environ,'PYTHONHASHSEED':'0'})
    return json.loads((output/'result.json').read_text())


def collect():
    ROOT.mkdir(exist_ok=False);(ROOT/'labels').mkdir();started=perf_counter();info=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(label_job,b,i) for b in range(1,7) for i in range(100)]
        for job in as_completed(jobs):
            info.append(job.result())
            if len(info)%50==0:print(f'BUILD CHAIN LABELS {len(info)}/600, {perf_counter()-started:.1f}s',flush=True)
    xs=[];ys=[];groups=[];masks=[];metadata=[]
    for batch in range(1,7):
        for i in range(100):
            with gzip.open(ROOT/'labels'/f'batch_{batch}_game_{i:04d}'/'labels.pkl.gz','rb') as stream:records=pickle.load(stream)
            for record in records:
                scores=record['score'];unique=sorted(set(scores));ranks={v:j for j,v in enumerate(unique)}
                xs.append(np.asarray(record['features'],dtype=np.float32))
                ys.append(np.asarray([max(0,3-(len(unique)-1-ranks[v])) for v in scores],dtype=np.int32))
                groups.append(len(scores));masks.append(i>=80)
                metadata.append({'batch':batch,'game_index':i,'seed':record['seed'],'action_index':record['action_index'],
                                 'teacher_index':record['teacher_index'],'candidate_count':len(scores)})
    data={'x':np.concatenate(xs),'y':np.concatenate(ys),'groups':np.asarray(groups),'validation_groups':np.asarray(masks,dtype=bool)}
    np.savez_compressed(ROOT/'teacher_data.npz',**data)
    (ROOT/'teacher_groups.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in metadata),encoding='utf-8')
    report={'games':600,'candidate_rows':len(data['x']),'decision_groups':len(groups),
         'road_build_followups':sum(i['road_build_followups'] for i in info),
         'double_trade_build_followups':sum(i['double_trade_build_followups'] for i in info),
         'all_immediate_build_followups':sum(i['trade_build_followups'] for i in info),
         'source_board_hashes':sorted({i['initial_map_hash'] for i in info}),'source_phases':[str(p) for p in SOURCES],
         'seconds':perf_counter()-started,'test_states_used':False,'label_workers':4}
    write(ROOT/'teacher_report.json',report);return data


def main():
    started=perf_counter();data=collect()
    normalizer=json.loads((SOURCE/'normalizer.json').read_text());write(ROOT/'normalizer.json',normalizer)
    excluded,manifest=exclusions(REPO/'runs/online-sarsa-features-100-v4')
    config={'algorithm':'stronger build-chain teacher distillation, frozen actual-return critic',
       'fresh_training_games':0,'teacher_source_games':600,'learner_forward_simulation':False,'players':2,'opponent':'native F',
       'source':str(SOURCE),'source_hashes':{n:sha(SOURCE/'best_model'/n) for n in ('ranker.model','monte_carlo.joblib','policy.json')},
       'phase_plan':['validation_candidate_800','validation_candidate_801','validation_candidate_802','baseline_test','candidate_test'],
       'validation_phase':'validation_candidate_800','test_phase':'baseline_test','validation_seed':62000000,'test_seed':63000000,
       'validation_games':100,'test_games_per_policy':300,'max_depth':10,'critic_frozen':True,
       'teacher_paths':['trade to city/settlement','road to city/settlement','two trades to city/settlement'],
       'teacher_path_simulation':'offline only on detached public beliefs, all added paths deterministic',
       'rankers':[{'num_leaves':127,'min_child_samples':16,'reg_lambda':1.},
                  {'num_leaves':255,'min_child_samples':8,'reg_lambda':1.}],
       'MC_mix':.2,'excluded_board_hashes':excluded,
       'selection':'highest validation wins; baseline retained on ties; final test never used to pick parameters',
       'promotion':'paired positive delta p<.05, wins validation selection, all audits pass'}
    write(ROOT/'config.json',config)
    baseline=load_policy(SOURCE/'best_model',normalizer,800);baseline.save(ROOT/'candidate_800')
    mlflow.set_tracking_uri(manifest['tracking_uri']);client=mlflow.MlflowClient()
    mlflow.set_experiment(experiment_id=client.get_experiment_by_name('CatanZero_TreePolicy_NativeF').experiment_id)
    history=[];validation_boards=[];fits=[];candidates=[]
    with mlflow.start_run(run_name='build_chain_teacher600_max_nativeF') as run:
        write(ROOT/'tracking.json',{'run_id':run.info.run_id,'tracking_uri':manifest['tracking_uri']})
        mlflow.log_params({'teacher_games':600,'ranker_candidates':2,'max_depth':10,'critic_frozen':True,'learner_simulation':False})
        mlflow.log_artifact(str(ROOT/'config.json'),'protocol_before_fitting')
        for i in range(3):
            if i==0:model=baseline
            else:
                fit_started=perf_counter();teacher,scale,fit=fit_ranker(data,config['rankers'][i-1])
                fits.append({**fit,'version':800+i,'seconds':perf_counter()-fit_started})
                model=Policy(teacher,'lightgbm',normalizer,scale,800+i,baseline.mc,.2);candidates.append(model)
            model.save(ROOT/f'candidate_{model.version}')
            rows,summary,_=play_phase(ROOT,f'validation_candidate_{model.version}',model,100,8,
                                     training=False,base_seed=config['validation_seed'],excluded=excluded)
            assert summary['completed']==100
            if not validation_boards:validation_boards=[r['initial_map_hash'] for r in rows]
            history.append({'version':model.version,'validation':summary})
            write(ROOT/'selection_history.json',history);write(ROOT/'fits.json',fits)
        best=max([baseline]+candidates,key=lambda m:(history[m.version-800]['validation']['wins'],-m.version))
        continued=max(candidates,key=lambda m:(history[m.version-800]['validation']['wins'],-m.version))
        continued.save(ROOT/'continued_model');best.save(ROOT/'best_validation_model')
        write(ROOT/'selection.json',{'model_version':best.version,'continued_version':continued.version,
              'validation_wins':history[best.version-800]['validation']['wins'],'test_inspected_for_selection':False})
        test_excluded=excluded+validation_boards
        print(f'STAGE: selected chain teacher {continued.version}; independent paired300-game tests',flush=True)
        before_rows,before,_=play_phase(ROOT,'baseline_test',baseline,300,8,training=False,
                         base_seed=config['test_seed'],excluded=test_excluded)
        after_rows,after,_=play_phase(ROOT,'candidate_test',continued,300,8,training=False,
                         base_seed=config['test_seed'],excluded=test_excluded)
        paired=paired_stats(before_rows,after_rows)
        sample=data['x'][::max(1,len(data['x'])//512)].astype(np.float64);expected=continued.predict(sample)
        np.savez_compressed(ROOT/'isolated_predictions.npz',raw=sample,expected=expected)
        info=mlflow.pyfunc.log_model(name='model',python_model=PortablePolicy(),
             artifacts={'ranker':str(ROOT/'continued_model/ranker.model'),'mc':str(ROOT/'continued_model/monte_carlo.joblib'),
                        'policy':str(ROOT/'continued_model/policy.json')},input_example=sample[:3],
             signature=infer_signature(sample,expected),pip_requirements=['mlflow=='+mlflow.__version__,'numpy=='+np.__version__,
             'lightgbm=='+lgb.__version__,'catboost=='+cb.__version__,'scikit-learn==1.7.1','joblib=='+joblib.__version__],
             code_paths=[str(REPO/'tools/optimize_dagger_policy.py'),str(REPO/'tools/optimize_tree_policy_vs_f.py'),
                         str(REPO/'tools/train_residual_tree_fqi.py'),str(REPO/'tools/train_online_sarsa.py'),
                         str(REPO/'tools/train_three_game_sarsa.py'),str(REPO/'src/catanzero')])
        error=float(np.max(np.abs(mlflow.pyfunc.load_model(info.model_uri).predict(sample)-expected)));assert error<1e-12
        version=mlflow.register_model(info.model_uri,'CatanZero_TreePolicy_NativeF')
        client.set_registered_model_alias('CatanZero_TreePolicy_NativeF','build_chain_pilot',str(version.version))
        write(ROOT/'registration.json',{'family':'CatanZero_TreePolicy_NativeF','version':str(version.version),
               'alias':'build_chain_pilot','model_uri':info.model_uri,'reload_max_abs_error':error})
        report={'baseline_test':before,'continued_test':after,'paired':paired,'teacher_games':600,
             'baseline_CI95':wilson_interval(before['wins'],300),'continued_CI95':wilson_interval(after['wins'],300),
             'qualifies_for_promotion_after_audits':bool(after['wins']>before['wins'] and paired['two_sided_p']<.05 and best.version==continued.version),
             'promoted':False,'test_used_for_training_or_selection':False,'elapsed_seconds':perf_counter()-started,
             'critic_frozen':True,'source_unchanged':all(sha(SOURCE/'best_model'/n)==h for n,h in config['source_hashes'].items())}
        assert report['source_unchanged'];write(ROOT/'report.json',report)
        for name in ('config.json','report.json','fits.json','selection.json','selection_history.json','registration.json','teacher_report.json'):
            mlflow.log_artifact(str(ROOT/name),'experiment')
        mlflow.log_metrics({'baseline_wins':before['wins'],'chain_teacher_wins':after['wins'],'paired_p':paired['two_sided_p']})
        print('RESULT: '+json.dumps(report),flush=True)


if __name__=='__main__':main()
