"""Short v4 continuation with a smaller step and public relative-point shaping."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
from scipy.stats import binomtest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from train_online_sarsa import CentralSARSA, play_phase, write, REPO
from train_three_game_sarsa import exclusions
from refit_linear_returns import FrozenLinear
from catanzero.online_linear import LinearActionValues


class CheckpointSARSA(CentralSARSA):
    def __init__(self,root,normalizer,source,eta0):
        super().__init__(root,normalizer,resume_from=source,eta0=eta0)
        self.finished_games=0

    def finish_game(self,index,censored=False):
        self.finished_games+=1
        if self.finished_games==50:
            self.save_checkpoint('checkpoint_50')

    def save_checkpoint(self,name):
        folder=self.root/name
        folder.mkdir(exist_ok=True)
        joblib.dump(self.model,folder/'model.joblib',compress=3)
        write(folder/'linear_model.json',self.document())
        write(folder/'checkpoint.json',{'finished_games':self.finished_games,'model_version':self.version,
            'training_transitions_used':self.transitions,'optimizer_t':float(self.model.t_),
            'note':'Parallel active games can contribute transitions; no checkpoint-specific extra flush.'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--games',type=int,default=100)
    parser.add_argument('--eval-games',type=int,default=100)
    parser.add_argument('--validation-games',type=int,default=32)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--eta0',type=float,default=.005)
    parser.add_argument('--gap-weight',type=float,default=.02)
    args=parser.parse_args()
    if args.games!=100 or any(n<=0 or n%4 for n in (args.eval_games,args.validation_games)) or args.workers<1:
        parser.error('100-game pilot, positive balanced evaluation and validation counts, positive workers')
    if not np.isfinite(args.eta0) or args.eta0<=0 or not np.isfinite(args.gap_weight) or args.gap_weight<0:
        parser.error('Positive finite eta0 and nonnegative finite gap weight required')
    root=args.output.resolve()
    source=REPO/'runs/online-sarsa-features-100-v4'
    excluded,manifest=exclusions(source)
    root.mkdir(parents=True,exist_ok=False)
    norm=json.loads((source/'normalizer.json').read_text())
    write(root/'normalizer.json',norm)
    shutil.copyfile(source/'feature_schema.json',root/'feature_schema.json')
    model=CheckpointSARSA(root,norm,source,args.eta0)
    old=joblib.load(source/'model.joblib')
    assert np.array_equal(old.coef_,model.model.coef_) and old.t_==model.model.t_
    config={'algorithm':'live semi-gradient SARSA with public relative VP potential','resume_from':str(source),
        'source_model_sha256':hashlib.sha256((source/'model.joblib').read_bytes()).hexdigest(),
        'input_columns':len(norm['offset']),'training_games':100,'evaluation_games':args.eval_games,
        'validation_games':args.validation_games,'workers':args.workers,'batch_transitions':32,
        'eta0_before':float(old.eta0),'eta0':args.eta0,'estimator_parameters':model.model.get_params(),
        'initial_optimizer_t':float(model.model.t_),'normalization_refitted':False,
        'normalization_source':str(source/'normalizer.json'),'gamma':.999,'vp_weight':.05,'gap_weight':args.gap_weight,
        'terminal_win_reward':1,'terminal_loss_reward':0,
        'potential':'0.05*known own actual VP + gap_weight*(known own actual VP - maximum opponent public VP)',
        'reward':'gamma^gap*Phi(next)-Phi(current); terminal Phi=0 and discounted win/loss reward',
        'epsilon_start':.1,'epsilon_end':.05,'training_seed':32000000,'evaluation_seed':33000000,
        'validation_seed':34000000,'excluded_board_hashes':excluded,'forward_simulation':False,
        'opponent':'three frozen F_static','features_changed':False,
        'selection':'Fixed 100-game final weights; checkpoint validation diagnostic only; test boards not used for model selection',
        'alias':'small_lr_gap_pilot; v4 feature_pilot remains unchanged'}
    write(root/'config.json',config)
    mlflow.set_tracking_uri(manifest['tracking_uri'])
    mlflow.set_experiment('CatanZero_Online_Linear_SARSA')
    opponents={'opponent_policy':'F_static','max_actions':2000,'max_seconds':120,'trace_compresslevel':1}
    started=perf_counter()
    with mlflow.start_run(run_name='v4_100_games_eta005_relative_gap002') as run:
        write(root/'tracking.json',{'run_id':run.info.run_id,'tracking_uri':manifest['tracking_uri']})
        mlflow.log_params({'model':'SGDRegressor','algorithm':'SARSA','train_games':100,'eta0':args.eta0,
            'gap_weight':args.gap_weight,'vp_weight':.05,'terminal_win_reward':1,'input_columns':len(norm['offset']),
            'batch_transitions':32,'workers':args.workers,'forward_simulation':False,'normalization_refitted':False})
        mlflow.log_artifact(str(root/'config.json'),'protocol_before_play')
        print(f'STAGE: frozen v4, {args.eval_games} fresh paired test boards',flush=True)
        before_rows,before=play_phase(root,'initial_evaluation',model,args.eval_games,args.workers,
            training=False,base_seed=config['evaluation_seed'],excluded=excluded,opponent_config=opponents)
        assert model.version==0
        print('STAGE: 100 training games, eta0=.005, public-gap potential=.02, batch threshold=32',flush=True)
        train_started=perf_counter()
        train_rows,train=play_phase(root,'live_training',model,100,args.workers,training=True,
            base_seed=config['training_seed'],excluded=excluded+[r['initial_map_hash'] for r in before_rows],
            exploration_start=.1,exploration_end=.05,opponent_config=opponents,gap_weight=args.gap_weight)
        training_seconds=perf_counter()-train_started
        model.trace.close()
        assert model.finished_games==100
        write(root/'training_history.json',model.history)
        write(root/'linear_model.json',model.document())
        joblib.dump(model.model,root/'model.joblib',compress=3)
        model.save_checkpoint('checkpoint_100')
        saved_hash=hashlib.sha256((root/'model.joblib').read_bytes()).hexdigest()
        validation={}
        validation_excluded=excluded+[r['initial_map_hash'] for r in before_rows+train_rows]
        snapshots={'before':(old,0),'50':(joblib.load(root/'checkpoint_50/model.joblib'),json.loads((root/'checkpoint_50/linear_model.json').read_text())['model_version']),
            '100':(model.model,model.version)}
        validation_rows={}
        for label,(estimator,version) in snapshots.items():
            print(f'STAGE: checkpoint {label}, {args.validation_games} separate validation boards',flush=True)
            validation_rows[label],validation[label]=play_phase(root,f'validation_{label}',FrozenLinear(estimator,norm,version),
                args.validation_games,args.workers,training=False,base_seed=config['validation_seed'],
                excluded=validation_excluded,opponent_config=opponents)
        for a,b,c in zip(validation_rows['before'],validation_rows['50'],validation_rows['100']):
            assert (a['seed'],a['initial_map_hash'],a['learner_seat'])==(b['seed'],b['initial_map_hash'],b['learner_seat'])==(c['seed'],c['initial_map_hash'],c['learner_seat'])
        write(root/'validation_report.json',validation)
        print('STAGE: frozen final 100-game model on paired test boards',flush=True)
        after_rows,after=play_phase(root,'final_evaluation',model,args.eval_games,args.workers,
            training=False,base_seed=config['evaluation_seed'],excluded=excluded+[r['initial_map_hash'] for r in train_rows]+[r['initial_map_hash'] for r in validation_rows['before']],
            opponent_config=opponents)
        assert hashlib.sha256((root/'model.joblib').read_bytes()).hexdigest()==saved_hash
        plus=minus=complete=0
        for a,b in zip(before_rows,after_rows):
            assert (a['seed'],a['initial_map_hash'],a['learner_seat'])==(b['seed'],b['initial_map_hash'],b['learner_seat'])
            if a['termination']==b['termination']=='victory':
                complete+=1;plus+=bool(b['learner_won'] and not a['learner_won']);minus+=bool(a['learner_won'] and not b['learner_won'])
        p=float(binomtest(plus,plus+minus,.5).pvalue) if plus+minus else 1.
        report={'initial_evaluation':before,'live_training':train,'final_evaluation':after,'validation':validation,
            'central_updates':model.version,'fresh_training_transitions':model.transitions,
            'updates_with_nonzero_parameter_change':sum(h['weight_change_l2']>0 for h in model.history),
            'training_seconds':training_seconds,'total_seconds_before_registration':perf_counter()-started,
            'paired_improvements':plus,'paired_regressions':minus,'paired_two_sided_p':p,'complete_pairs':complete,
            'confirmed_improvement':bool(plus>minus and p<.05 and complete==args.eval_games),
            'model_sha256':saved_hash,'best_alias_changed':False,
            'limitations':['Learning rate, reward shaping and training duration differ from previous 600-game pilot together',
                '100 paired test games is a pilot; checkpoint validation has only 32 boards','Checkpoints evaluated after training; diagnostic only, not early stopping',
                'Opponent private VP cards excluded from relative reward; only public VP used']}
        write(root/'report.json',report)
        scaler=StandardScaler();scaler.mean_,scaler.scale_=np.asarray(norm['offset']),np.asarray(norm['scale'])
        scaler.var_,scaler.n_features_in_=scaler.scale_**2,len(norm['offset'])
        counts=np.full(len(norm['offset']),norm['fit_rows'],dtype=np.int64)
        for block in norm.get('normalization_source_blocks',[]):counts[block['start']:block['stop']]=block['fit_rows']
        scaler.n_samples_seen_=counts
        pipeline=Pipeline([('float64',FunctionTransformer(np.asarray,kw_args={'dtype':np.float64})),('normalization',scaler),
            ('clip',FunctionTransformer(np.clip,kw_args={'a_min':-4,'a_max':4})),('sarsa',model.model)])
        raw=scaler.mean_+np.random.default_rng(42).normal(size=(32,len(norm['offset'])))*scaler.scale_
        expected=model.model.predict(model.normalize(raw))
        portable=LinearActionValues(model.document(),norm)
        np.testing.assert_allclose([portable.predict(x.tolist()) for x in raw],expected,atol=1e-12,rtol=0)
        requirements=[s for s in (REPO/'requirements-mlflow.txt').read_text().splitlines() if s and not s.startswith('#')]
        info=mlflow.sklearn.log_model(pipeline,name='model',serialization_format='cloudpickle',input_example=raw[:2],
            signature=infer_signature(raw,expected),pip_requirements=requirements)
        error=float(np.max(np.abs(mlflow.sklearn.load_model(info.model_uri).predict(raw)-expected)))
        assert error<1e-12
        registered=mlflow.register_model(info.model_uri,'CatanZero_Online_Linear_SARSA')
        mlflow.MlflowClient().set_registered_model_alias('CatanZero_Online_Linear_SARSA','small_lr_gap_pilot',str(registered.version))
        write(root/'registration.json',{'registered_model':'CatanZero_Online_Linear_SARSA','version':str(registered.version),
            'alias':'small_lr_gap_pilot','model_uri':info.model_uri,'run_id':run.info.run_id,
            'input_columns':len(norm['offset']),'reload_max_abs_error':error,'normalization_and_clipping_included':True,'best_alias_changed':False})
        for filename in ('tools/train_short_relative_sarsa.py','tools/train_online_sarsa.py','src/catanzero/online_linear.py','src/catanzero/online_env.py'):
            dest=root/'implementation_snapshot'/filename;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/filename,dest)
        mlflow.log_metrics({'before_wins':before['learner_wins'],'after_wins':after['learner_wins'],'paired_p':p,
            'training_seconds':training_seconds,'updates':model.version})
        for path in root.iterdir():
            if path.is_file() and path.suffix not in ('.gz','.joblib'):
                mlflow.log_artifact(str(path),'short_relative_sarsa')
        print('RESULT: '+json.dumps(report),flush=True)


if __name__=='__main__':main()
