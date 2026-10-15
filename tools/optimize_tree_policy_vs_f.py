"""Keep the strongest depth-10 tree policy by real wins vs native F."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
import argparse
import json
from pathlib import Path
import random
import shutil
from time import perf_counter
import joblib
import lightgbm as lgb
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from train_residual_tree_fqi import play_phase,REPO,write
from train_three_game_sarsa import exclusions
from catanzero.duel_protocol import wilson_interval


class TreePolicy:
    policy_name='Tree_Ranking_MonteCarlo'
    def __init__(self,teacher,normalizer,scale=1.,mc=None,mix=0.,version=0):
        self.teacher,self.normalizer,self.scale=teacher,normalizer,scale
        self.mc,self.mix,self.version=mc,mix,version

    def predict(self,raw):
        raw=np.asarray(raw,dtype=np.float64)
        if not len(raw):return np.empty(0)
        score=self.teacher.predict(raw,num_threads=1)/self.scale
        if self.mix and self.mc is not None:score+=self.mix*self.mc.predict(raw)
        return score

    def choose(self,response,rng,epsilon):
        raw=np.asarray(response['features'],dtype=np.float64);scores=self.predict(raw)
        explored=rng.random()<epsilon
        selected=rng.randrange(len(raw)) if explored else int(np.argmax(scores))
        return {'index':selected,'raw':raw[selected],'scores':scores.tolist(),'version':self.version,'explored':explored}

    def save(self,directory):
        directory.mkdir(exist_ok=True)
        self.teacher.save_model(str(directory/'ranker.txt'))
        joblib.dump(self.mc,directory/'monte_carlo.joblib',compress=3)
        write(directory/'policy.json',{'scale':self.scale,'mix':self.mix,'version':self.version,
              'input_columns':264,'forward_simulation':False,'max_depth':10,'scores':'action preferences, not calibrated probabilities'})


class TreePolicyModel(mlflow.pyfunc.PythonModel):
    def load_context(self,context):
        self.teacher=lgb.Booster(model_file=context.artifacts['ranker'])
        self.mc=joblib.load(context.artifacts['mc'])
        self.document=json.loads(Path(context.artifacts['policy']).read_text())

    def predict(self,context,model_input,params=None):
        raw=np.asarray(model_input,dtype=np.float64)
        if raw.ndim!=2 or raw.shape[1]!=264 or not np.isfinite(raw).all():raise ValueError('finite raw (n,264) features required')
        score=self.teacher.predict(raw,num_threads=1)/self.document['scale']
        if self.document['mix'] and self.mc is not None:score+=self.document['mix']*self.mc.predict(raw)
        return score


def fit_ranker(data,params):
    groups=data['groups'];mask=data['validation_groups'];row_mask=np.repeat(mask,groups)
    kwargs=dict(objective='lambdarank',metric='ndcg',max_depth=10,n_estimators=350,learning_rate=.06,
                n_jobs=4,verbosity=-1,random_state=2026,deterministic=True,force_col_wise=True,max_bin=63,
                lambdarank_truncation_level=10,**params)
    fit=lgb.LGBMRanker(**kwargs)
    fit.fit(data['x'][~row_mask],data['y'][~row_mask],group=groups[~mask],
            eval_set=[(data['x'][row_mask],data['y'][row_mask])],eval_group=[groups[mask]],eval_at=[1],
            callbacks=[lgb.early_stopping(25,verbose=False)])
    iterations=fit.best_iteration_
    final=lgb.LGBMRanker(**{**kwargs,'n_estimators':iterations})
    final.fit(data['x'],data['y'],group=groups)
    predictions=final.booster_.predict(data['x'],num_threads=1)
    spreads=[];cursor=0;agreements=0
    for n in groups:
        q=predictions[cursor:cursor+n];y=data['y'][cursor:cursor+n]
        spreads.append(float(np.std(q)));agreements+=bool(y[np.argmax(q)]==y.max());cursor+=n
    scale=max(.1,float(np.median(spreads)))
    return final.booster_,scale,{'best_iterations':iterations,'teacher_best_set_agreement_train':agreements/len(groups),'scale':scale}


def monte_carlo_data(root,phases):
    xs=[];targets=[];weights=[];ids=[]
    for phase in phases:
        data=dict(np.load(root/phase/'experience.npz'))
        metadata=[json.loads(s) for s in (root/phase/'transitions.jsonl').read_text().splitlines()]
        results={r['game_index']:r for r in map(json.loads,(root/phase/'results.jsonl').read_text().splitlines())}
        counts={}
        for m in metadata:counts[m['game_index']]=counts.get(m['game_index'],0)+1
        for i,m in enumerate(metadata):
            result=results[m['game_index']]
            if result['termination']!='victory':continue
            # Undiscounted shaped return from this real decision to termination.
            target=(1. if result['learner_won'] else -1.)-data['phi'][i]
            xs.append(data['x'][i]);targets.append(target);weights.append(1/counts[m['game_index']])
            ids.append({'phase':phase,'game_index':m['game_index'],'action_index':m['action_index'],
                        'target':target,'phi':float(data['phi'][i]),'won':result['learner_won']})
    return np.asarray(xs),np.asarray(targets),np.asarray(weights),ids


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=REPO/'runs/tree-policy-native-f-max-v1')
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--test-games',type=int,default=500)
    args=parser.parse_args();root=args.output.resolve()
    data=dict(np.load(root/'teacher_data.npz'));teacher_report=json.loads((root/'teacher_report.json').read_text())
    source=REPO/'runs/online-sarsa-features-100-v4';excluded,manifest=exclusions(source)
    normalizer=json.loads((source/'normalizer.json').read_text());write(root/'normalizer.json',normalizer)
    config={'algorithm':'native-F policy distillation then Monte Carlo tree policy improvement selected by live wins',
            'max_depth':10,'neural_network':False,'learner_forward_simulation':False,'opponent':'native F','players':2,
            'teacher_sources':teacher_report['training_sources'],'teacher_data_rows':len(data['x']),'teacher_data_groups':len(data['groups']),
            'features':'raw 264, no clipping; trees handle numeric scales','reward':'win +1/loss -1, gamma1, potential .05 own VP+.02 public gap',
            'ranker_configurations':[{'num_leaves':31,'min_child_samples':16,'reg_lambda':.1},
                                    {'num_leaves':63,'min_child_samples':24,'reg_lambda':1.},
                                    {'num_leaves':127,'min_child_samples':16,'reg_lambda':1.}],
            'RL_rounds':2,'games_per_RL_round':100,'exploration':.08,'MC_trees':64,'MC_max_depth':10,'MC_min_leaf':16,
            'MC_mix_candidates':[.2,.6],'validation_games':50,'validation_seed':48500000,
            'test_seed':49000000,'test_games':args.test_games,'workers':args.workers,'excluded_board_hashes':excluded,
            'selection':'maximize frozen validation wins vs native F, retain previous best on ties; test used once after selection',
            'teacher_labels':'warm start only; improvement targets use actual own games vs F, not F scores',
            'sampling':'whole-game splits; source labels never include prior game validation/test states'}
    write(root/'config.json',config)
    mlflow.set_tracking_uri(manifest['tracking_uri']);client=mlflow.MlflowClient();name='CatanZero_TreePolicy_NativeF'
    exp=client.get_experiment_by_name(name);eid=exp.experiment_id if exp else client.create_experiment(name,artifact_location=(REPO/'runs/mlflow/artifacts').as_uri())
    mlflow.set_experiment(experiment_id=eid)
    started=perf_counter();history=[];policies={};best=None;best_wins=-1;validation_boards=[];training_rows=[];phases=[]
    with mlflow.start_run(run_name='depth10_ranker_then_MC_maximize_F_wins') as run:
        write(root/'tracking.json',{'run_id':run.info.run_id,'tracking_uri':manifest['tracking_uri']})
        mlflow.log_params({'max_depth':10,'model':'LightGBM LambdaRank plus Monte Carlo ExtraTrees','opponent':'F',
                           'players':2,'learner_simulation':False,'teacher_games':300,'RL_games':200,'test_games':args.test_games,'loss_reward':-1})
        mlflow.log_artifact(str(root/'config.json'),'protocol_before_optimization')
        for i,params in enumerate(config['ranker_configurations']):
            print(f'STAGE: depth10 ranker candidate {i+1}/3',flush=True)
            fit_start=perf_counter();booster,scale,fit_info=fit_ranker(data,params)
            model=TreePolicy(booster,normalizer,scale,version=100+i)
            directory=root/f'candidate_{model.version}';model.save(directory);policies[model.version]=model
            rows,validation,_=play_phase(root,f'validation_candidate_{model.version}',model,50,args.workers,training=False,
                                         base_seed=48500000,excluded=excluded)
            if not validation_boards:validation_boards=[r['initial_map_hash'] for r in rows]
            entry={'version':model.version,'type':'teacher_ranker','params':params,'fit':fit_info,'validation':validation,
                   'fit_seconds':perf_counter()-fit_start-validation['elapsed_seconds']}
            history.append(entry)
            if not validation['unfinished'] and validation['wins']>best_wins:best=model;best_wins=validation['wins']
            write(root/'selection_history.json',history)
            print(f'BEST: {best_wins}/50 validation wins, version={best.version}',flush=True)
        for round_index in range(1,3):
            phase=f'RL_round_{round_index}';phases.append(phase)
            print(f'STAGE: actual RL games {round_index}/2, current best={best_wins}/50',flush=True)
            rows,summary,_=play_phase(root,phase,best,100,args.workers,training=True,base_seed=50000000+1000*round_index,
                                      excluded=excluded+validation_boards+[r['initial_map_hash'] for r in training_rows],epsilon=.08)
            training_rows.extend(rows)
            x,y,w,ids=monte_carlo_data(root,phases)
            fit_start=perf_counter();mc=ExtraTreesRegressor(n_estimators=64,max_depth=10,min_samples_leaf=16,max_features=.75,
                                                          n_jobs=4,random_state=2026+round_index)
            mc.fit(x,y,sample_weight=w);mc.n_jobs=1
            np.savez_compressed(root/f'MC_round_{round_index}.npz',x=x,target=y,weight=w)
            (root/f'MC_round_{round_index}.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in ids),encoding='utf-8')
            joblib.dump(mc,root/f'MC_round_{round_index}.joblib',compress=3)
            print(f'MC FIT: {len(x)} real return targets, {perf_counter()-fit_start:.1f}s',flush=True)
            # Preserve the best ranking teacher; only outcome-based correction varies.
            reference=best
            for i,mix in enumerate((.2,.6)):
                version=200+10*round_index+i
                model=TreePolicy(reference.teacher,normalizer,reference.scale,mc,mix,version)
                model.save(root/f'candidate_{version}');policies[version]=model
                _,validation,_=play_phase(root,f'validation_candidate_{version}',model,50,args.workers,training=False,
                                          base_seed=48500000,excluded=excluded+[r['initial_map_hash'] for r in training_rows])
                history.append({'version':version,'type':'MonteCarlo_RL','round':round_index,'mix':mix,'validation':validation})
                if not validation['unfinished'] and validation['wins']>best_wins:best=model;best_wins=validation['wins']
                write(root/'selection_history.json',history)
                print(f'BEST: {best_wins}/50 validation wins, version={best.version}',flush=True)
        best.save(root/'best_model')
        write(root/'selection.json',{'model_version':best.version,'validation_wins':best_wins,'validation_games':50,
                                     'test_inspected_for_selection':False,'selection':'highest wins, earlier tie'})
        print(f'STAGE: final frozen best tree vs native F, {args.test_games} fresh games',flush=True)
        _,test,_=play_phase(root,'final_test',best,args.test_games,args.workers,training=False,base_seed=49000000,
                            excluded=excluded+validation_boards+[r['initial_map_hash'] for r in training_rows])
        sample=data['x'][::max(1,len(data['x'])//512)].astype(np.float64);expected=best.predict(sample)
        np.savez_compressed(root/'isolated_predictions.npz',raw=sample,expected=expected)
        model_info=mlflow.pyfunc.log_model(name='model',python_model=TreePolicyModel(),
             artifacts={'ranker':str(root/'best_model/ranker.txt'),'mc':str(root/'best_model/monte_carlo.joblib'),
                        'policy':str(root/'best_model/policy.json')},input_example=sample[:3],signature=infer_signature(sample,expected),
             pip_requirements=['mlflow=='+mlflow.__version__,'numpy=='+np.__version__,'lightgbm=='+lgb.__version__,
                               'scikit-learn==1.7.1','joblib=='+joblib.__version__])
        restored=mlflow.pyfunc.load_model(model_info.model_uri)
        reload_error=float(np.max(np.abs(restored.predict(sample)-expected)));assert reload_error<1e-12
        registered=mlflow.register_model(model_info.model_uri,'CatanZero_TreePolicy_NativeF')
        client.set_registered_model_alias('CatanZero_TreePolicy_NativeF','best_pilot',str(registered.version))
        write(root/'registration.json',{'family':'CatanZero_TreePolicy_NativeF','version':str(registered.version),'alias':'best_pilot',
             'model_uri':model_info.model_uri,'reload_max_abs_error':reload_error,'input_columns':264})
        report={'best_version':best.version,'best_validation_wins':best_wins,'validation_games':50,'final_test':test,
                'final_win_rate':test['wins']/args.test_games if not test['unfinished'] else None,
                'wilson_95':wilson_interval(test['wins'],args.test_games) if not test['unfinished'] else None,
                'neural_network':False,'learner_forward_simulation':False,'teacher_games':300,'fresh_RL_games':200,
                'best_contains_MC_RL_correction':bool(best.mix),'model_selection_uses_test':False,'elapsed_seconds':perf_counter()-started,
                'reload_max_abs_error':reload_error,'limitations':['selection set only 50 games, seven candidates may overfit it',
                    'two-player base simulator, no claim for four players','distillation inherits native F tendencies',
                    'best among tested candidates, not a globally optimal policy','if warm-start-only model wins selection, do not claim RL improved it']}
        write(root/'report.json',report)
        for file in ('tools/optimize_tree_policy_vs_f.py','tools/collect_native_f_labels.py','src/catanzero/label_f_policy.py','src/catanzero/tree_duel_reward.py'):
            dest=root/'implementation_snapshot'/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/file,dest)
        mlflow.log_metrics({'final_wins':test['wins'],'final_games':args.test_games,'validation_wins':best_wins,
                            'final_win_rate':report['final_win_rate'] or 0.,'elapsed_seconds':report['elapsed_seconds'],'reload_error':reload_error})
        for name in ('config.json','report.json','selection.json','selection_history.json','registration.json','teacher_report.json'):
            mlflow.log_artifact(str(root/name),'experiment')
        mlflow.log_artifacts(str(root/'implementation_snapshot'),'implementation_snapshot')
        print('RESULT: '+json.dumps(report),flush=True)


if __name__=='__main__':main()
