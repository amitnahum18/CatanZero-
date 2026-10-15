"""Classical tree actor updates from real advantages, with PPO clipping and KL."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
import argparse
import json
from pathlib import Path
from time import perf_counter
import joblib
import lightgbm as lgb
import catboost as cb
import mlflow
from mlflow.models import infer_signature
import numpy as np
from continue_champion_tree_rl import SOURCE,load_policy,sha,paired_stats
from optimize_dagger_policy import Policy,PortablePolicy
from train_residual_tree_fqi import REPO,play_phase,write
from train_three_game_sarsa import exclusions
from catanzero.duel_protocol import wilson_interval


def grouped_softmax(values,starts,lengths):
    values=np.asarray(values,dtype=np.float64)
    maximum=np.maximum.reduceat(values,starts)
    exponent=np.exp(values-np.repeat(maximum,lengths))
    return exponent/np.repeat(np.add.reduceat(exponent,starts),lengths)


class SamplingPolicy(Policy):
    def __init__(self,reference,temperature=.25):
        super().__init__(reference.teacher,reference.kind,reference.normalizer,reference.scale,
                         reference.version,reference.mc,reference.mix)
        self.temperature=temperature
        self.behavior_config={'behavior_policy':'softmax','temperature':temperature,'epsilon':0.}

    def choose(self,response,rng,epsilon):
        raw=np.asarray(response['features'],dtype=np.float64)
        scores=np.asarray(response['_scores']) if '_scores' in response else self.predict(raw)
        probabilities=np.exp((scores-scores.max())/self.temperature);probabilities/=probabilities.sum()
        u=rng.random();selected=int(np.searchsorted(np.cumsum(probabilities),u,side='right'))
        selected=min(selected,len(probabilities)-1)
        return {'index':selected,'raw':raw[selected],'scores':scores.tolist(),'version':self.version,
                'explored':selected!=int(np.argmax(scores)),'probability':float(probabilities[selected])}


class ClippedActorObjective:
    """Exact first derivative of clipped surrogate + KL(old||new).

    Hessian is a positive diagonal Fisher majorizer, not the exact Hessian of
    the nonconvex PPO objective. It is used only as a tree-fitting step metric.
    """
    def __init__(self,data,scale,temperature,clip=.2,kl_weight=.1):
        self.starts=np.asarray(data['starts'],dtype=np.int64);self.lengths=np.asarray(data['lengths'],dtype=np.int64)
        self.selected=np.asarray(data['selected_rows'],dtype=np.int64)
        self.advantage=np.asarray(data['advantage']);self.weights=np.asarray(data['weight'])
        self.old_p=np.asarray(data['old_probabilities']);self.fixed=np.asarray(data['fixed_critic'])
        self.old_chosen=self.old_p[self.selected];self.factor=1/(scale*temperature)
        self.temperature=temperature;self.clip=clip;self.kl=kl_weight

    def probabilities(self,pred):
        return grouped_softmax(pred*self.factor+self.fixed/self.temperature,self.starts,self.lengths)

    def loss(self,pred):
        p=self.probabilities(pred);ratio=p[self.selected]/self.old_chosen
        clipped=np.clip(ratio,1-self.clip,1+self.clip)
        objective=-np.minimum(ratio*self.advantage,clipped*self.advantage)
        relative=self.old_p*(np.log(np.maximum(self.old_p,1e-300))-np.log(np.maximum(p,1e-300)))
        kl=np.add.reduceat(relative,self.starts)
        return float(np.sum(self.weights*(objective+self.kl*kl)))

    def __call__(self,pred,dataset):
        p=self.probabilities(pred);ratio=p[self.selected]/self.old_chosen
        active=~(((self.advantage>=0)&(ratio>1+self.clip))|((self.advantage<0)&(ratio<1-self.clip)))
        coefficient=self.advantage*ratio*active
        grad=np.repeat(coefficient,self.lengths)*p
        grad[self.selected]-=coefficient
        grad+=self.kl*(p-self.old_p)
        row_weight=np.repeat(self.weights,self.lengths)
        grad*=self.factor*row_weight
        hessian=(np.repeat(np.abs(coefficient),self.lengths)+self.kl)*p*(1-p)*self.factor**2
        hessian=np.maximum(hessian,1e-4)*row_weight
        if not np.isfinite(grad).all() or not np.isfinite(hessian).all():raise ValueError('nonfinite actor derivative')
        return grad,hessian


def actor_data(root,reference,temperature):
    experience=dict(np.load(root/'RL_round_1/experience.npz'))
    metadata=[json.loads(s) for s in (root/'RL_round_1/transitions.jsonl').read_text().splitlines()]
    results={r['game_index']:r for r in map(json.loads,(root/'RL_round_1/results.jsonl').read_text().splitlines())}
    if any(r['termination']!='victory' for r in results.values()):raise ValueError('incomplete actor game')
    x=experience['current_x'];starts=experience['current_ranges'][:,0];lengths=np.diff(experience['current_ranges'],axis=1)[:,0]
    selected=starts+np.asarray([m['selected_index'] for m in metadata])
    np.testing.assert_array_equal(x[selected],experience['x'])
    base_logits=reference.predict(x)/temperature
    old_probabilities=grouped_softmax(base_logits,starts,lengths)
    np.testing.assert_allclose(old_probabilities[selected],experience['behavior_probability'],rtol=0,atol=1e-12)
    critic=reference.mc.predict(x)
    baseline=np.add.reduceat(old_probabilities*critic,starts)
    outcome=np.asarray([1. if results[m['game_index']]['learner_won'] else -1. for m in metadata])
    returns=outcome-experience['phi'];advantage=returns-baseline
    counts={}
    for m in metadata:counts[m['game_index']]=counts.get(m['game_index'],0)+1
    weights=np.asarray([1/counts[m['game_index']] for m in metadata]);weights/=weights.mean()
    data={'x':x.astype(np.float32),'starts':starts,'lengths':lengths,'selected_rows':selected,
          'old_probabilities':old_probabilities,'fixed_critic':reference.mix*critic,'baseline':baseline,
          'return':returns,'advantage':advantage,'weight':weights}
    np.savez_compressed(root/'actor_data.npz',**data)
    return data


def fit_actor(root,reference,data,temperature):
    objective=ClippedActorObjective(data,reference.scale,temperature)
    before=reference.teacher.predict(data['x'],num_threads=1);initial_loss=objective.loss(before)
    started=perf_counter()
    params={'objective':objective,'metric':'None','max_depth':10,'num_leaves':31,'min_data_in_leaf':32,
            'lambda_l2':3.,'learning_rate':.05,'max_delta_step':.5,'max_bin':63,
            'num_threads':4,'verbosity':-1,'deterministic':True,'force_col_wise':True,'seed':20261015}
    train=lgb.Dataset(data['x'],label=np.zeros(len(data['x'])),free_raw_data=False)
    model=lgb.train(params,train,num_boost_round=24,init_model=reference.teacher)
    versions=[];base_trees=reference.teacher.num_trees()
    for added,version in ((8,701),(24,702)):
        teacher=lgb.Booster(model_str=model.model_to_string(num_iteration=base_trees+added))
        candidate=Policy(teacher,'lightgbm',reference.normalizer,reference.scale,version,reference.mc,reference.mix)
        candidate.save(root/f'candidate_{version}');versions.append(candidate)
    final=versions[-1].teacher.predict(data['x'],num_threads=1)
    grad,hessian=objective(before,None)
    fit={'initial_loss':initial_loss,'final_loss':objective.loss(final),'rows':len(data['x']),
          'decision_groups':len(data['lengths']),'base_trees':base_trees,'added_trees':[8,24],
          'seconds':perf_counter()-started,'gradient_max_abs':float(np.max(np.abs(grad))),
          'positive_diagonal_majorizer':bool((hessian>0).all()),'advantage_mean':float(np.mean(data['advantage'])),
          'clipping':.2,'KL_reference_weight':.1,'temperature':temperature,
          'parameters':{k:v for k,v in params.items() if k!='objective'}}
    np.testing.assert_array_equal(reference.teacher.predict(data['x'],num_threads=1),before)
    assert fit['final_loss']<fit['initial_loss']
    write(root/'fit.json',fit);print('ACTOR FIT: '+json.dumps(fit),flush=True)
    return versions


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();root=REPO/'runs/tree-policy-actor-rl-v5';root.mkdir(exist_ok=args.resume)
    normalizer=json.loads((SOURCE/'normalizer.json').read_text());write(root/'normalizer.json',normalizer)
    baseline=load_policy(SOURCE/'best_model',normalizer,700);baseline.save(root/'candidate_700')
    if args.resume:
        existing=json.loads((root/'config.json').read_text());excluded=existing['excluded_board_hashes']
        manifest=json.loads((REPO/'runs/mlflow-label-study-v2/manifest.json').read_text())
    else:excluded,manifest=exclusions(REPO/'runs/online-sarsa-features-100-v4')
    config={'algorithm':'tree actor PPO-clipped advantage update; frozen ExtraTrees state baseline',
            'source':str(SOURCE),'source_hashes':{n:sha(SOURCE/'best_model'/n) for n in ('ranker.model','monte_carlo.joblib','policy.json')},
            'fresh_training_games':100,'training_seed':59000000,'validation_seed':60000000,'test_seed':61000000,
            'phase_plan':['RL_round_1','validation_candidate_700','validation_candidate_701','validation_candidate_702',
                          'baseline_test','candidate_test'],'validation_phase':'validation_candidate_700','test_phase':'baseline_test',
            'validation_games':100,'test_games_per_policy':300,'temperature':.25,'max_depth':10,
            'learned_actor':'LightGBM114 plus8/24 new trees; existing64 ExtraTrees frozen',
            'selection':'highest validation wins, baseline retained on ties; one continued candidate selected before final test',
            'promotion':'positive paired delta and exact p<.05; all audits pass; never use test to retune',
            'excluded_board_hashes':excluded,'players':2,'opponent':'native F','learner_forward_simulation':False,
            'reward':'actual terminal +1/-1, gamma1, state-only potential; advantage=actual return-fixed state baseline',
            'reference':'https://arxiv.org/abs/1707.06347','implementation':'custom LightGBM gradients with positive Fisher diagonal majorizer'}
    if not args.resume:write(root/'config.json',config)
    started=perf_counter()
    def phase(name,model,count,**kwargs):
        directory=root/name
        if (directory/'summary.json').exists():
            summary=json.loads((directory/'summary.json').read_text());assert summary['completed']==count
            rows=sorted([json.loads(s) for s in (directory/'results.jsonl').read_text().splitlines()],key=lambda r:r['game_index'])
            print(f'RESUME {name}: {count} completed games retained',flush=True)
            return rows,summary,None
        if directory.exists():raise ValueError('archive interrupted phase before resume: '+str(directory))
        return play_phase(root,name,model,count,args.workers,**kwargs)
    mlflow.set_tracking_uri(manifest['tracking_uri']);client=mlflow.MlflowClient()
    mlflow.set_experiment(experiment_id=client.get_experiment_by_name('CatanZero_TreePolicy_NativeF').experiment_id)
    run_args={'run_id':json.loads((root/'tracking.json').read_text())['run_id']} if args.resume else {'run_name':'tree_actor_real_advantage_100'}
    with mlflow.start_run(**run_args) as run:
        write(root/'tracking.json',{'run_id':run.info.run_id,'tracking_uri':manifest['tracking_uri']})
        mlflow.log_params({'actor':'LightGBM PPO-clipped custom objective','training_games':100,'max_depth':10,
                          'temperature':.25,'KL_reference_weight':.1,'critic_frozen':True})
        mlflow.log_artifact(str(root/'config.json'),'protocol_before_training')
        print('STAGE: 100 actual stochastic actor games, temperature .25',flush=True)
        train_rows,training,_=phase('RL_round_1',SamplingPolicy(baseline),100,
             training=True,base_seed=config['training_seed'],excluded=excluded,epsilon=0,record_candidates=True)
        assert training['completed']==100
        if args.resume and (root/'fit.json').exists():
            candidates=[load_policy(root/f'candidate_{v}',normalizer) for v in (701,702)]
            print('RESUME: fitted actors retained; no retraining',flush=True)
        else:
            data=actor_data(root,baseline,config['temperature']);candidates=fit_actor(root,baseline,data,config['temperature'])
            del data
        history=[];val_boards=[]
        for candidate in [baseline]+candidates:
            rows,summary,_=phase(f'validation_candidate_{candidate.version}',candidate,100,
                 training=False,base_seed=config['validation_seed'],excluded=excluded+[r['initial_map_hash'] for r in train_rows])
            assert summary['completed']==100
            if not val_boards:val_boards=[r['initial_map_hash'] for r in rows]
            history.append({'version':candidate.version,'validation':summary});write(root/'selection_history.json',history)
        best=max([baseline]+candidates,key=lambda c:(history[c.version-700]['validation']['wins'],-c.version))
        continued=max(candidates,key=lambda c:(history[c.version-700]['validation']['wins'],-c.version))
        continued.save(root/'continued_model');best.save(root/'best_validation_model')
        write(root/'selection.json',{'model_version':best.version,'continued_version':continued.version,
              'validation_wins':history[best.version-700]['validation']['wins'],'test_inspected_for_selection':False})
        test_excluded=excluded+val_boards+[r['initial_map_hash'] for r in train_rows]
        print(f'STAGE: selected actor {continued.version}; fresh paired300-game tests vs F',flush=True)
        before_rows,before,_=phase('baseline_test',baseline,300,training=False,
                                      base_seed=config['test_seed'],excluded=test_excluded)
        after_rows,after,_=phase('candidate_test',continued,300,training=False,
                                    base_seed=config['test_seed'],excluded=test_excluded)
        paired=paired_stats(before_rows,after_rows)
        with np.load(root/'actor_data.npz') as stored:
            raw=stored['x'];sample=raw[::max(1,len(raw)//512)].astype(np.float64)
        del raw
        expected=continued.predict(sample)
        np.savez_compressed(root/'isolated_predictions.npz',raw=sample,expected=expected)
        info=mlflow.pyfunc.log_model(name='model',python_model=PortablePolicy(),
             artifacts={'ranker':str(root/'continued_model/ranker.model'),'mc':str(root/'continued_model/monte_carlo.joblib'),
                        'policy':str(root/'continued_model/policy.json')},input_example=sample[:3],
             signature=infer_signature(sample,expected),pip_requirements=['mlflow=='+mlflow.__version__,'numpy=='+np.__version__,
             'lightgbm=='+lgb.__version__,'catboost=='+cb.__version__,'scikit-learn==1.7.1','joblib=='+joblib.__version__],
             code_paths=[str(REPO/'tools/optimize_dagger_policy.py'),str(REPO/'tools/optimize_tree_policy_vs_f.py'),
                         str(REPO/'tools/train_residual_tree_fqi.py'),str(REPO/'tools/train_online_sarsa.py'),
                         str(REPO/'tools/train_three_game_sarsa.py'),str(REPO/'src/catanzero')])
        error=float(np.max(np.abs(mlflow.pyfunc.load_model(info.model_uri).predict(sample)-expected)));assert error<1e-12
        version=mlflow.register_model(info.model_uri,'CatanZero_TreePolicy_NativeF')
        client.set_registered_model_alias('CatanZero_TreePolicy_NativeF','actor_rl_pilot',str(version.version))
        write(root/'registration.json',{'family':'CatanZero_TreePolicy_NativeF','version':str(version.version),
               'alias':'actor_rl_pilot','model_uri':info.model_uri,'reload_max_abs_error':error})
        report={'training':training,'baseline_test':before,'continued_test':after,'paired':paired,
             'baseline_CI95':wilson_interval(before['wins'],300),'continued_CI95':wilson_interval(after['wins'],300),
             'qualifies_for_promotion_after_audits':bool(after['wins']>before['wins'] and paired['two_sided_p']<.05 and best.version==continued.version),
             'promoted':False,'test_used_for_training_or_selection':False,'elapsed_seconds':perf_counter()-started,
             'resumed_after_speed_batching':bool(args.resume),
             'critic_frozen':True,'source_unchanged':all(sha(SOURCE/'best_model'/n)==h for n,h in config['source_hashes'].items())}
        assert report['source_unchanged'];write(root/'report.json',report)
        for name in ('config.json','report.json','fit.json','selection.json','selection_history.json','registration.json'):
            mlflow.log_artifact(str(root/name),'experiment')
        mlflow.log_metrics({'baseline_wins':before['wins'],'actor_wins':after['wins'],'paired_p':paired['two_sided_p']})
        print('RESULT: '+json.dumps(report),flush=True)


if __name__=='__main__':main()
