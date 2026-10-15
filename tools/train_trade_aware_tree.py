"""Focused depth10 policy distillation with trade-to-build teacher labels."""
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
import shutil
import subprocess
from time import perf_counter
import joblib
import lightgbm as lgb
import catboost as cb
import mlflow
from mlflow.models import infer_signature
import numpy as np
from optimize_dagger_policy import Policy,PortablePolicy
from optimize_tree_policy_vs_f import fit_ranker
from train_residual_tree_fqi import REPO,play_phase,write
from train_three_game_sarsa import exclusions
from catanzero.duel_protocol import wilson_interval

ROOT=REPO/'runs/tree-policy-trade-aware-v3'
PARENT=REPO/'runs/tree-policy-dagger-max-v2'


def label_job(round_index,index):
    sources=[REPO/'runs/residual-extratrees-depth10-f-duel-v1'/f'training_round_{i}' for i in (1,2,3)]
    sources+=[REPO/'runs/tree-policy-native-f-max-v1'/f'RL_round_{i}' for i in (1,2)]
    source=sources[round_index-1]/f'game_{index:04d}';output=ROOT/'labels'/f'round_{round_index}_game_{index:04d}'
    with (ROOT/'labels'/f'worker_{round_index}_{index:04d}.log').open('w',encoding='utf-8') as log:
        subprocess.run([str(REPO/'.venv/Scripts/python.exe'),'-u','-m','catanzero.label_f_policy','--source',str(source),
                        '--output',str(output),'--trade-followup'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,
                        check=True,timeout=180,env={**os.environ,'PYTHONHASHSEED':'0'})
    return json.loads((output/'result.json').read_text())


def collect():
    start=perf_counter();infos=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(label_job,r,i) for r in range(1,6) for i in range(100)]
        for future in as_completed(futures):
            infos.append(future.result())
            if len(infos)%50==0:print(f'TRADE LABELS {len(infos)}/500, {perf_counter()-start:.1f}s',flush=True)
    xs=[];ys=[];groups=[];masks=[];metadata=[]
    for r in range(1,6):
        for i in range(100):
            with gzip.open(ROOT/'labels'/f'round_{r}_game_{i:04d}'/'labels.pkl.gz','rb') as stream:records=pickle.load(stream)
            for record in records:
                scores=record['score'];unique=sorted(set(scores));ranks={s:k for k,s in enumerate(unique)}
                xs.append(np.asarray(record['features'],dtype=np.float32));ys.append(np.asarray([max(0,3-(len(unique)-1-ranks[s])) for s in scores],dtype=np.int32))
                groups.append(len(scores));masks.append(i>=80)
                metadata.append({'round':r,'game_index':i,'seed':record['seed'],'action_index':record['action_index'],
                                 'teacher_index':record['teacher_index'],'candidate_count':len(scores)})
    data={'x':np.concatenate(xs),'y':np.concatenate(ys),'groups':np.asarray(groups),'validation_groups':np.asarray(masks,dtype=bool)}
    np.savez_compressed(ROOT/'teacher_data.npz',**data)
    (ROOT/'teacher_groups.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in metadata),encoding='utf-8')
    report={'games':500,'candidate_rows':len(data['x']),'decision_groups':len(groups),
            'trade_build_followups':sum(i['trade_build_followups'] for i in infos),'test_states_used':False,
            'source_board_hashes':sorted({i['initial_map_hash'] for i in infos}),'seconds':perf_counter()-start,
            'teacher':'native F value with extra deterministic city/settlement after maritime trade only'}
    write(ROOT/'teacher_report.json',report);return data


def main():
    ROOT.mkdir();(ROOT/'labels').mkdir()
    excluded,manifest=exclusions(REPO/'runs/online-sarsa-features-100-v4')
    normalizer=json.loads((PARENT/'normalizer.json').read_text());write(ROOT/'normalizer.json',normalizer)
    config={'algorithm':'depth10 trade-aware teacher policy, selected by native-F wins','parent':str(PARENT),
            'max_depth':10,'players':2,'opponent':'native F','input_columns':264,'neural_network':False,'learner_forward_simulation':False,
            'teacher':'F plus deterministic trade-to-city/settlement backup; offline labels only',
            'candidate_protocol':['prior best','trade-aware LightGBM','trade-aware plus .2 MC'],
            'validation_seed':54000000,'validation_games':100,'final_test_seed':55000000,'final_test_games':500,
            'excluded_board_hashes':excluded,'selection':'highest new validation wins, earliest tie; new final test after selection',
            'reward':'existing MC correction has terminal +1/-1, gamma1 and potential .05 own VP+.02 public gap'}
    write(ROOT/'config.json',config);started=perf_counter()
    data=collect();mlflow.set_tracking_uri(manifest['tracking_uri']);client=mlflow.MlflowClient()
    mlflow.set_experiment(experiment_id=client.get_experiment_by_name('CatanZero_TreePolicy_NativeF').experiment_id)
    history=[];best=None;best_wins=-1;val_boards=[]

    def evaluate(model,label):
        nonlocal best,best_wins,val_boards
        model.save(ROOT/f'candidate_{model.version}')
        rows,summary,_=play_phase(ROOT,f'validation_candidate_{model.version}',model,100,8,training=False,
                                 base_seed=54000000,excluded=excluded)
        if not val_boards:val_boards=[r['initial_map_hash'] for r in rows]
        history.append({'version':model.version,'type':label,'kind':model.kind,'mix':model.mix,'validation':summary})
        if not summary['unfinished'] and summary['wins']>best_wins:best=model;best_wins=summary['wins']
        write(ROOT/'selection_history.json',history);print(f'BEST: {best_wins}/100, version={best.version}',flush=True)

    with mlflow.start_run(run_name='depth10_trade_aware_native_F_max') as run:
        write(ROOT/'tracking.json',{'run_id':run.info.run_id,'tracking_uri':manifest['tracking_uri']})
        mlflow.log_params({'max_depth':10,'teacher_games':500,'opponent':'F','players':2,'teacher_trade_followup':True,'learner_simulation':False})
        mlflow.log_artifact(str(ROOT/'config.json'),'protocol_before_training')
        prior=json.loads((PARENT/'best_model/policy.json').read_text())
        model=Policy(lgb.Booster(model_file=str(PARENT/'best_model/ranker.model')),'lightgbm',normalizer,prior['scale'],400)
        print('STAGE: prior best on fresh validation',flush=True);evaluate(model,'prior best')
        fit_start=perf_counter();booster,scale,fit=fit_ranker(data,{'num_leaves':127,'min_child_samples':16,'reg_lambda':1.})
        write(ROOT/'fit.json',{**fit,'seconds':perf_counter()-fit_start})
        trained=Policy(booster,'lightgbm',normalizer,scale,401)
        print('STAGE: trade-aware depth10 model',flush=True);evaluate(trained,'trade-aware ranker')
        mc=joblib.load(REPO/'runs/tree-policy-native-f-max-v1/MC_round_2.joblib')
        evaluate(Policy(booster,'lightgbm',normalizer,scale,402,mc,.2),'trade-aware with MC RL correction')
        best.save(ROOT/'best_model');write(ROOT/'selection.json',{'model_version':best.version,'validation_wins':best_wins,
                   'validation_games':100,'test_inspected_for_selection':False})
        print(f'STAGE: selected {best.version}, new 500-game final test',flush=True)
        _,test,_=play_phase(ROOT,'final_test',best,500,8,training=False,base_seed=55000000,excluded=excluded+val_boards)
        sample=data['x'][::max(1,len(data['x'])//512)].astype(np.float64);expected=best.predict(sample)
        np.savez_compressed(ROOT/'isolated_predictions.npz',raw=sample,expected=expected)
        info=mlflow.pyfunc.log_model(name='model',python_model=PortablePolicy(),artifacts={'ranker':str(ROOT/'best_model/ranker.model'),
             'mc':str(ROOT/'best_model/monte_carlo.joblib'),'policy':str(ROOT/'best_model/policy.json')},input_example=sample[:3],
             signature=infer_signature(sample,expected),pip_requirements=['mlflow=='+mlflow.__version__,'numpy=='+np.__version__,
             'lightgbm=='+lgb.__version__,'catboost=='+cb.__version__,'scikit-learn==1.7.1','joblib=='+joblib.__version__],
             code_paths=[str(REPO/'tools/optimize_dagger_policy.py'),str(REPO/'tools/optimize_tree_policy_vs_f.py'),
                         str(REPO/'tools/train_residual_tree_fqi.py'),str(REPO/'tools/train_online_sarsa.py'),
                         str(REPO/'tools/train_three_game_sarsa.py'),str(REPO/'src/catanzero')])
        restored=mlflow.pyfunc.load_model(info.model_uri);error=float(np.max(np.abs(restored.predict(sample)-expected)));assert error<1e-12
        version=mlflow.register_model(info.model_uri,'CatanZero_TreePolicy_NativeF')
        client.set_registered_model_alias('CatanZero_TreePolicy_NativeF','best_pilot',str(version.version))
        write(ROOT/'registration.json',{'family':'CatanZero_TreePolicy_NativeF','version':str(version.version),'alias':'best_pilot',
              'model_uri':info.model_uri,'reload_max_abs_error':error,'input_columns':264})
        report={'best_version':best.version,'best_validation_wins':best_wins,'validation_games':100,'final_test':test,
                'final_win_rate':test['wins']/500,'wilson_95':wilson_interval(test['wins'],500) if not test['unfinished'] else None,
                'reached_50_percent_observed':test['wins']>=250 and not test['unfinished'],'best_contains_MC_RL_correction':bool(best.mix),
                'teacher_games':500,'actual_MC_RL_games':200,'learner_forward_simulation':False,'neural_network':False,
                'max_depth':10,'reload_max_abs_error':error,'elapsed_seconds':perf_counter()-started,
                'final_test_used_for_selection':False,'limitations':['best among tested candidates, not global optimum',
                    'teacher search occurs only in offline labeling','if zero MC mix wins, selected model is learned imitation, not RL improvement',
                    'two-player base-engine result; does not establish strength in full four-player Catan']}
        write(ROOT/'report.json',report)
        mlflow.log_metrics({'final_wins':test['wins'],'final_games':500,'final_win_rate':report['final_win_rate'],'validation_wins':best_wins,'reload_error':error})
        for name in ('config.json','report.json','selection.json','selection_history.json','registration.json','teacher_report.json','fit.json'):
            mlflow.log_artifact(str(ROOT/name),'experiment')
        print('RESULT: '+json.dumps(report),flush=True)


if __name__=='__main__':main()
