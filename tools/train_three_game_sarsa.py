"""Continue frozen-normalization SARSA; one update per three finished games."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
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
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from train_online_sarsa import CentralSARSA, play_phase, write, REPO
from catanzero.board_exclusions import normalization_source_board_hashes
from catanzero.online_linear import LinearActionValues


class ThreeGameSARSA(CentralSARSA):
    def __init__(self, root, normalizer, resume_from=None):
        super().__init__(root, normalizer, resume_from=resume_from)
        self.completed_games = 0
        self.finished_ids = set()
        self.pending_finished = []
        self.batches = []
        (root/'checkpoints').mkdir()

    def flush(self, force=False):
        # play_phase calls this after each decision round. Episode completion,
        # rather than the number of transitions, gates the SGD update.
        if not force:
            return
        if not self.queue:
            return
        if len(self.pending_finished) != 3:
            raise ValueError('Updates require exactly three newly finished games')
        super().flush(force=True)
        batch = {'update_version':self.version,'completed_games':self.completed_games,
                 'newly_finished_games':self.pending_finished.copy(),
                 'transitions':self.history[-1]['transitions'],
                 'cumulative_transitions':self.transitions}
        self.batches.append(batch)
        self.pending_finished=[]
        write(self.root/'update_game_batches.json',self.batches)
        joblib.dump(self.model,self.root/'checkpoints'/f'version_{self.version:06d}.joblib',compress=3)
        write(self.root/'checkpoint_status.json',{'completed_games':self.completed_games,
              'updates':self.version,'transitions':self.transitions,
              'checkpoint':f'checkpoints/version_{self.version:06d}.joblib'})

    def finish_game(self, game_index, censored=False):
        if game_index in self.finished_ids:
            raise ValueError('Game completion counted twice')
        self.finished_ids.add(game_index)
        self.completed_games+=1
        self.pending_finished.append({'game_index':game_index,'censored':bool(censored)})
        if len(self.pending_finished)==3:
            self.flush(force=True)


def exclusions(source):
    manifest=json.loads((REPO/'runs/mlflow-label-study-v2/manifest.json').read_text())
    hashes=set(manifest['source_board_hashes'])
    hashes.update(json.loads((REPO/'runs/fqi-pilot-500-v1/experience_audit.json').read_text())['collection']['board_hashes'])
    visited=set()
    ancestor=source
    while ancestor:
        if ancestor in visited:
            raise ValueError('Cyclic model ancestry')
        visited.add(ancestor)
        config=json.loads((ancestor/'config.json').read_text())
        if config.get('new_normalization_source'):
            hashes.update(normalization_source_board_hashes(Path(config['new_normalization_source'])))
        for phase in ('initial_evaluation','live_training','final_evaluation'):
            hashes.update(json.loads(s)['initial_map_hash'] for s in (ancestor/phase/'results.jsonl').read_text().splitlines())
        parent=config.get('resume_from')
        ancestor=Path(parent) if parent else None
    # All previously observed gameplay boards are excluded, including comparisons.
    for path in (REPO/'runs').rglob('results.jsonl'):
        if any(s in path.parts for s in ('mlflow','artifacts','implementation_snapshot')):
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row.get('initial_map_hash'):
                hashes.add(row['initial_map_hash'])
    return sorted(hashes),manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=REPO/'runs/online-sarsa-features-100-v4')
    parser.add_argument('--games',type=int,default=600)
    parser.add_argument('--eval-games',type=int,default=200)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.games<=0 or args.games%12 or args.eval_games<=0 or args.eval_games%4 or args.workers<1:
        parser.error('Training games divisible by 12, evaluation games divisible by 4, positive workers')
    root,source=args.output.resolve(),args.source.resolve()
    if root.exists():
        parser.error('Output must be a new directory')
    excluded,manifest=exclusions(source)
    root.mkdir(parents=True)
    norm=json.loads((source/'normalizer.json').read_text())
    write(root/'normalizer.json',norm)
    shutil.copyfile(source/'feature_schema.json',root/'feature_schema.json')
    model=ThreeGameSARSA(root,norm,resume_from=source)
    config={'algorithm':'live three-finished-game semi-gradient SARSA','model':'SGDRegressor',
        'resume_from':str(source),'source_model_sha256':hashlib.sha256((source/'model.joblib').read_bytes()).hexdigest(),
        'input_columns':len(norm['offset']),'training_games':args.games,'evaluation_games':args.eval_games,
        'planned_updates':args.games//3,'finished_games_per_update':3,'workers':args.workers,
        'update_semantics':'One central partial_fit on all fresh transitions accumulated since the last update; parallel games may span updates.',
        'normalization_refitted':False,'normalization_source':str(source/'normalizer.json'),
        'initial_optimizer_t':float(model.model.t_),'estimator_parameters':model.model.get_params(),
        'gamma':.999,'vp_weight':.05,'epsilon_start':.1,'epsilon_end':.05,
        'training_seed':26000000,'evaluation_seed':27000000,'excluded_board_hashes':excluded,
        'opponent':'three frozen F_static','forward_simulation':False,
        'selection':'fixed final weights; paired fresh evaluation; no early stopping or tuning on test wins',
        'alias':'every3_pilot; existing best alias remains unchanged'}
    write(root/'config.json',config)
    mlflow.set_tracking_uri(manifest['tracking_uri'])
    mlflow.set_experiment('CatanZero_Online_Linear_SARSA')
    opponents={'opponent_policy':'F_static','max_actions':2000,'max_seconds':120,'trace_compresslevel':1}
    started=perf_counter()
    with mlflow.start_run(run_name=f'v4_continued_{args.games}_games_update_every3') as run:
        write(root/'tracking.json',{'run_id':run.info.run_id,'tracking_uri':manifest['tracking_uri']})
        mlflow.log_params({'model':'SGDRegressor','algorithm':'SARSA','input_columns':len(norm['offset']),
            'training_games':args.games,'finished_games_per_update':3,'workers':args.workers,
            'forward_simulation':False,'normalization_refitted':False,'epsilon_start':.1,'epsilon_end':.05})
        mlflow.log_artifact(str(root/'config.json'),'protocol_before_play')
        print('STAGE: frozen v4, 200 fresh paired boards',flush=True)
        before_rows,before=play_phase(root,'initial_evaluation',model,args.eval_games,args.workers,
            training=False,base_seed=config['evaluation_seed'],excluded=excluded,opponent_config=opponents)
        assert model.version==0
        print('STAGE: 600 new training games, one update per three finished games',flush=True)
        train_started=perf_counter()
        training_rows,training=play_phase(root,'live_training',model,args.games,args.workers,
            training=True,base_seed=config['training_seed'],excluded=excluded+[r['initial_map_hash'] for r in before_rows],
            exploration_start=.1,exploration_end=.05,opponent_config=opponents)
        training_seconds=perf_counter()-train_started
        assert model.version==args.games//3 and model.completed_games==args.games and not model.queue and not model.pending_finished
        model.trace.close()
        write(root/'training_history.json',model.history)
        write(root/'linear_model.json',model.document())
        joblib.dump(model.model,root/'model.joblib',compress=3)
        saved_hash=hashlib.sha256((root/'model.joblib').read_bytes()).hexdigest()
        print('STAGE: final saved candidate, frozen paired evaluation',flush=True)
        frozen=model.model.coef_.copy()
        after_rows,after=play_phase(root,'final_evaluation',model,args.eval_games,args.workers,
            training=False,base_seed=config['evaluation_seed'],excluded=excluded+[r['initial_map_hash'] for r in training_rows],opponent_config=opponents)
        assert model.version==args.games//3 and np.array_equal(model.model.coef_,frozen)
        assert hashlib.sha256((root/'model.joblib').read_bytes()).hexdigest()==saved_hash
        plus=minus=complete=0
        for a,b in zip(before_rows,after_rows):
            assert (a['seed'],a['initial_map_hash'],a['learner_seat'])==(b['seed'],b['initial_map_hash'],b['learner_seat'])
            if a['termination']==b['termination']=='victory':
                complete+=1
                plus+=bool(b['learner_won'] and not a['learner_won'])
                minus+=bool(a['learner_won'] and not b['learner_won'])
        p=float(binomtest(plus,plus+minus,.5).pvalue) if plus+minus else 1.
        report={'initial_evaluation':before,'live_training':training,'final_evaluation':after,
            'central_updates':model.version,'fresh_training_transitions':model.transitions,
            'updates_with_nonzero_parameter_change':sum(h['weight_change_l2']>0 for h in model.history),
            'paired_improvements':plus,'paired_regressions':minus,'complete_pairs':complete,'paired_two_sided_p':p,
            'confirmed_improvement':bool(plus>minus and p<.05 and complete==args.eval_games),
            'training_seconds':training_seconds,'total_seconds_before_registration':perf_counter()-started,
            'normalization_changed':False,'features_changed':False,'model_sha256':saved_hash,
            'limitations':['Additional training and update cadence change together','One final checkpoint evaluated; intermediate weights not selected by test wins','Feature influence describes this linear policy, not causal feature quality']}
        write(root/'report.json',report)
        scaler=StandardScaler()
        scaler.mean_,scaler.scale_=np.asarray(norm['offset']),np.asarray(norm['scale'])
        scaler.var_,scaler.n_features_in_=scaler.scale_**2,len(norm['offset'])
        scaler.n_samples_seen_=np.asarray([41924]*164+[11260]*100)
        pipeline=Pipeline([('float64',FunctionTransformer(np.asarray,kw_args={'dtype':np.float64})),('normalization',scaler),
            ('clip',FunctionTransformer(np.clip,kw_args={'a_min':-4,'a_max':4})),('sarsa',model.model)])
        raw=scaler.mean_+np.random.default_rng(42).normal(size=(32,len(norm['offset'])))*scaler.scale_
        expected=model.model.predict(model.normalize(raw))
        portable=LinearActionValues(model.document(),norm)
        np.testing.assert_allclose([portable.predict(x.tolist()) for x in raw],expected,atol=1e-12,rtol=0)
        requirements=[s for s in (REPO/'requirements-mlflow.txt').read_text().splitlines() if s and not s.startswith('#')]
        info=mlflow.sklearn.log_model(pipeline,name='model',serialization_format='cloudpickle',input_example=raw[:2],
            signature=infer_signature(raw,expected),pip_requirements=requirements)
        restored=mlflow.sklearn.load_model(info.model_uri)
        error=float(np.max(np.abs(restored.predict(raw)-expected)))
        assert error<1e-12
        registered=mlflow.register_model(info.model_uri,'CatanZero_Online_Linear_SARSA')
        mlflow.MlflowClient().set_registered_model_alias('CatanZero_Online_Linear_SARSA','every3_pilot',str(registered.version))
        write(root/'registration.json',{'registered_model':'CatanZero_Online_Linear_SARSA','version':str(registered.version),
            'alias':'every3_pilot','model_uri':info.model_uri,'run_id':run.info.run_id,'input_columns':len(norm['offset']),
            'reload_max_abs_error':error,'normalization_and_clipping_included':True,'best_alias_changed':False})
        for filename in ('tools/train_three_game_sarsa.py','tools/train_online_sarsa.py','src/catanzero/online_env.py','src/catanzero/features.py','src/catanzero/board_exclusions.py'):
            dest=root/'implementation_snapshot'/filename
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(REPO/filename,dest)
        mlflow.log_metrics({'before_wins':before['learner_wins'],'after_wins':after['learner_wins'],
            'paired_p':p,'updates':model.version,'training_seconds':training_seconds})
        for path in root.iterdir():
            if path.is_file() and path.suffix not in ('.gz','.joblib'):
                mlflow.log_artifact(str(path),'three_game_sarsa')
        print('RESULT: '+json.dumps(report),flush=True)


if __name__=='__main__':
    main()
