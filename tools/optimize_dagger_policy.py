"""Depth-10 tree candidates on aggregated learner states; fresh final test."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
import json
from pathlib import Path
import shutil
from time import perf_counter
import joblib
import lightgbm as lgb
import catboost as cb
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
import numpy as np
from optimize_tree_policy_vs_f import fit_ranker
from train_residual_tree_fqi import REPO,play_phase,write
from train_three_game_sarsa import exclusions
from catanzero.duel_protocol import wilson_interval


class Policy:
    policy_name='Tree_Ranking_MonteCarlo'
    def __init__(self,teacher,kind,normalizer,scale,version,mc=None,mix=0.):
        self.teacher,self.kind,self.normalizer,self.scale,self.version=teacher,kind,normalizer,scale,version
        self.mc,self.mix=mc,mix

    def predict(self,raw):
        raw=np.asarray(raw,dtype=np.float64)
        if not len(raw):return np.empty(0)
        values=self.teacher.predict(raw,num_threads=1) if self.kind=='lightgbm' else self.teacher.predict(raw,thread_count=1)
        values=values/self.scale
        if self.mix and self.mc is not None:values=values+self.mix*self.mc.predict(raw)
        return values

    def choose(self,response,rng,epsilon):
        raw=np.asarray(response['features'],dtype=np.float64)
        scores=np.asarray(response['_scores']) if '_scores' in response else self.predict(raw)
        explored=rng.random()<epsilon
        selected=rng.randrange(len(raw)) if explored else int(np.argmax(scores))
        return {'index':selected,'raw':raw[selected],'scores':scores.tolist(),'version':self.version,'explored':explored}

    def choose_many(self,responses,rngs,epsilon):
        raw=np.concatenate([np.asarray(r['features'],dtype=np.float64) for r in responses])
        scores=self.predict(raw);cursor=0;plans=[]
        for response,rng in zip(responses,rngs):
            n=len(response['features'])
            plans.append(self.choose({**response,'_scores':scores[cursor:cursor+n]},rng,epsilon));cursor+=n
        return plans

    def save(self,directory):
        directory.mkdir(exist_ok=True)
        self.teacher.save_model(str(directory/'ranker.model'))
        joblib.dump(self.mc,directory/'monte_carlo.joblib',compress=3)
        write(directory/'policy.json',{'kind':self.kind,'scale':self.scale,'version':self.version,'mix':self.mix,
                                      'max_depth':10,'input_columns':264,'scores':'action preferences, not calibrated probabilities'})


class PortablePolicy(mlflow.pyfunc.PythonModel):
    def load_context(self,context):
        self.document=json.loads(Path(context.artifacts['policy']).read_text());self.mc=joblib.load(context.artifacts['mc'])
        if self.document['kind']=='lightgbm':self.teacher=lgb.Booster(model_file=context.artifacts['ranker'])
        else:self.teacher=cb.CatBoostRanker();self.teacher.load_model(context.artifacts['ranker'])

    def predict(self,context,model_input,params=None):
        raw=np.asarray(model_input,dtype=np.float64)
        if raw.ndim!=2 or raw.shape[1]!=264 or not np.isfinite(raw).all():raise ValueError('expected finite raw (n,264) features')
        score=self.teacher.predict(raw,num_threads=1) if self.document['kind']=='lightgbm' else self.teacher.predict(raw,thread_count=1)
        score=score/self.document['scale']
        if self.document['mix'] and self.mc is not None:score+=self.document['mix']*self.mc.predict(raw)
        return score


def fit_catboost(data):
    groups=data['groups'];mask=np.repeat(data['validation_groups'],groups);gid=np.repeat(np.arange(len(groups)),groups)
    train=cb.Pool(data['x'][~mask],data['y'][~mask],group_id=gid[~mask])
    val=cb.Pool(data['x'][mask],data['y'][mask],group_id=gid[mask])
    params={'iterations':120,'depth':10,'learning_rate':.1,'loss_function':'QueryRMSE','l2_leaf_reg':3,
            'thread_count':4,'random_seed':2026,'allow_writing_files':False,'verbose':False}
    preliminary=cb.CatBoostRanker(**params)
    preliminary.fit(train,eval_set=val,early_stopping_rounds=20,use_best_model=True)
    iterations=preliminary.tree_count_
    final=cb.CatBoostRanker(**{**params,'iterations':iterations})
    final.fit(cb.Pool(data['x'],data['y'],group_id=gid))
    prediction=final.predict(data['x'],thread_count=1);spreads=[];cursor=0;agreements=0
    for n in groups:
        p=prediction[cursor:cursor+n];y=data['y'][cursor:cursor+n]
        spreads.append(float(np.std(p)));agreements+=y[np.argmax(p)]==y.max();cursor+=n
    return final,max(.1,float(np.median(spreads))),{'trees':iterations,'teacher_best_set_agreement_train':float(agreements/len(groups))}


def main():
    root=REPO/'runs/tree-policy-dagger-max-v2';parent=REPO/'runs/tree-policy-native-f-max-v1'
    data=dict(np.load(root/'teacher_data.npz'));teacher_report=json.loads((root/'teacher_report.json').read_text())
    normalizer=json.loads((parent/'normalizer.json').read_text());write(root/'normalizer.json',normalizer)
    excluded,manifest=exclusions(REPO/'runs/online-sarsa-features-100-v4')
    config={'algorithm':'native-F dataset aggregation on learner-visited states, tree ranking plus optional real-return MC correction',
            'parent':str(parent),'teacher_games':500,'teacher_candidate_rows':len(data['x']),'max_depth':10,'input_columns':264,
            'players':2,'opponent':'native F','neural_network':False,'learner_forward_simulation':False,
            'reward':'MC correction from +1 win/-1 loss, gamma1, .05 own VP+.02 public gap potential',
            'candidate_protocol':['prior best LightGBM','aggregated-data LightGBM','aggregated-data CatBoost','best ranker + .2 MC','best ranker + .6 MC'],
            'validation_seed':52000000,'validation_games':100,'final_test_seed':53000000,'final_test_games':500,'workers':8,
            'excluded_board_hashes':excluded,'MC_training_source':str(parent/'MC_round_2.npz'),
            'selection':'maximum wins on NEW 100-board validation, keep earliest ties; NEW 500-board test used once after selection',
            'prior_test':'prior 49000000 test is retired and excluded; it is not reused as new heldout evidence',
            'raw_features':True,'tree_normalization':'no scaling/clipping; inherited normalizer supplies feature schema only'}
    write(root/'config.json',config)
    mlflow.set_tracking_uri(manifest['tracking_uri']);client=mlflow.MlflowClient();name='CatanZero_TreePolicy_NativeF'
    exp=client.get_experiment_by_name(name);mlflow.set_experiment(experiment_id=exp.experiment_id)
    history=[];best=None;best_wins=-1;val_boards=[];started=perf_counter()

    def evaluate(model,label):
        nonlocal best,best_wins,val_boards
        model.save(root/f'candidate_{model.version}')
        rows,summary,_=play_phase(root,f'validation_candidate_{model.version}',model,100,8,training=False,
                                 base_seed=52000000,excluded=excluded)
        if not val_boards:val_boards=[r['initial_map_hash'] for r in rows]
        history.append({'version':model.version,'type':label,'kind':model.kind,'mix':model.mix,'validation':summary})
        if not summary['unfinished'] and summary['wins']>best_wins:best=model;best_wins=summary['wins']
        write(root/'selection_history.json',history)
        print(f'BEST: {best_wins}/100 native-F validation wins, version={best.version}',flush=True)

    with mlflow.start_run(run_name='depth10_dagger_new_validation_and_500_test') as run:
        write(root/'tracking.json',{'run_id':run.info.run_id,'tracking_uri':manifest['tracking_uri']})
        mlflow.log_params({'max_depth':10,'teacher_games':500,'opponent':'F','players':2,'validation_games':100,'final_test_games':500,'candidate_search':False})
        mlflow.log_artifact(str(root/'config.json'),'protocol_before_optimization')
        prior=json.loads((parent/'best_model/policy.json').read_text())
        model=Policy(lgb.Booster(model_file=str(parent/'best_model/ranker.txt')),'lightgbm',normalizer,prior['scale'],300)
        print('STAGE: prior strongest trees on NEW validation boards',flush=True);evaluate(model,'prior best ranker')
        print('STAGE: aggregate-data LightGBM depth10',flush=True)
        fit_start=perf_counter();booster,scale,info=fit_ranker(data,{'num_leaves':127,'min_child_samples':16,'reg_lambda':1.})
        write(root/'lightgbm_fit.json',{**info,'seconds':perf_counter()-fit_start})
        evaluate(Policy(booster,'lightgbm',normalizer,scale,301),'aggregated LightGBM')
        print('STAGE: aggregate-data CatBoost depth10',flush=True)
        fit_start=perf_counter();cat,scale,info=fit_catboost(data)
        write(root/'catboost_fit.json',{**info,'seconds':perf_counter()-fit_start})
        evaluate(Policy(cat,'catboost',normalizer,scale,302),'aggregated CatBoost')
        reference=best;mc=joblib.load(parent/'MC_round_2.joblib')
        for version,mix in ((303,.2),(304,.6)):
            evaluate(Policy(reference.teacher,reference.kind,normalizer,reference.scale,version,mc,mix),'real-return MC correction')
        best.save(root/'best_model')
        write(root/'selection.json',{'model_version':best.version,'validation_wins':best_wins,'validation_games':100,
                                     'test_inspected_for_selection':False,'retained_prior_on_ties':True})
        print(f'STAGE: frozen best tree vs native F, NEW 500 games, selected={best.version}',flush=True)
        _,test,_=play_phase(root,'final_test',best,500,8,training=False,base_seed=53000000,excluded=excluded+val_boards)
        sample=data['x'][::max(1,len(data['x'])//512)].astype(np.float64);expected=best.predict(sample)
        np.savez_compressed(root/'isolated_predictions.npz',raw=sample,expected=expected)
        info=mlflow.pyfunc.log_model(name='model',python_model=PortablePolicy(),
             artifacts={'ranker':str(root/'best_model/ranker.model'),'mc':str(root/'best_model/monte_carlo.joblib'),'policy':str(root/'best_model/policy.json')},
             input_example=sample[:3],signature=infer_signature(sample,expected),pip_requirements=['mlflow=='+mlflow.__version__,
             'numpy=='+np.__version__,'lightgbm=='+lgb.__version__,'catboost=='+cb.__version__,'scikit-learn==1.7.1','joblib=='+joblib.__version__])
        restored=mlflow.pyfunc.load_model(info.model_uri);error=float(np.max(np.abs(restored.predict(sample)-expected)));assert error<1e-12
        version=mlflow.register_model(info.model_uri,'CatanZero_TreePolicy_NativeF')
        client.set_registered_model_alias('CatanZero_TreePolicy_NativeF','best_pilot',str(version.version))
        write(root/'registration.json',{'family':'CatanZero_TreePolicy_NativeF','version':str(version.version),'alias':'best_pilot','model_uri':info.model_uri,'reload_max_abs_error':error,'input_columns':264})
        report={'best_version':best.version,'best_kind':best.kind,'best_contains_MC_RL_correction':bool(best.mix),
                'validation_wins':best_wins,'validation_games':100,'final_test':test,'final_win_rate':test['wins']/500,
                'wilson_95':wilson_interval(test['wins'],500) if not test['unfinished'] else None,
                'reached_50_percent_observed':test['wins']>=250 and not test['unfinished'],'max_depth':10,
                'neural_network':False,'learner_forward_simulation':False,'teacher_games':500,'MC_real_training_games':200,
                'final_test_used_for_selection':False,'reload_max_abs_error':error,'elapsed_seconds':perf_counter()-started,
                'limitations':['five candidates selected on 100 games; maximum among tested candidates only',
                    'two-player base simulator, not full four-player strength','if MC correction loses selection, winner is imitation learning, not an RL improvement',
                    'a win rate above 50% alone does not establish superiority without sufficient statistical certainty']}
        write(root/'report.json',report)
        for file in ('tools/optimize_dagger_policy.py','tools/collect_dagger_labels.py','src/catanzero/label_f_policy.py'):
            dest=root/'implementation_snapshot'/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/file,dest)
        mlflow.log_metrics({'final_wins':test['wins'],'final_games':500,'final_win_rate':report['final_win_rate'],'validation_wins':best_wins,
                            'elapsed_seconds':report['elapsed_seconds'],'reload_error':error})
        for name in ('config.json','report.json','selection.json','selection_history.json','registration.json','teacher_report.json','lightgbm_fit.json','catboost_fit.json'):
            mlflow.log_artifact(str(root/name),'experiment')
        mlflow.log_artifacts(str(root/'implementation_snapshot'),'implementation_snapshot')
        print('RESULT: '+json.dumps(report),flush=True)


if __name__=='__main__':main()
