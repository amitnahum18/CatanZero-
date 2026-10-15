"""One bounded policy-improvement round using paired real terminal rollout rewards."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from multiprocessing import get_context
from pathlib import Path
import random
import shutil
from time import perf_counter

import joblib
import lightgbm as lgb
import numpy as np
from catanatron import Color
from catanatron.serialization import action_to_json, state_from_json, state_to_json, map_to_json
from catanzero.counterfactual_402 import evaluate_pair
from catanzero.dataset import observation, ordered_actions
from catanzero.features import FeatureEncoder
from catanzero.simulation import create_game, SimulationConfig, SimulationRandomPlayer
from catanzero.static_policies import forbid_forward_simulation
from continue_champion_tree_rl import load_policy, paired_stats
from optimize_tree_policy_vs_f import TreePolicy
from train_residual_tree_fqi import REPO, play_phase, write
from train_three_game_sarsa import exclusions
from catanzero.duel_protocol import wilson_interval
from evaluation_confidence import win_rate_acceptance

SOURCE = REPO/'runs/tree-policy-trade-aware-v3'
_POLICY = None


def save_policy(model, directory, normalizer, sample):
    """Save the same portable layout used by champion 402 and verify reload."""
    model.save(directory)
    model.teacher.save_model(str(directory/'ranker.model'))
    info=json.loads((directory/'policy.json').read_text())
    info['kind']='lightgbm'
    write(directory/'policy.json',info)
    write(directory/'normalizer.json',normalizer)
    restored=load_policy(directory,normalizer)
    np.testing.assert_allclose(restored.predict(sample),model.predict(sample),rtol=0,atol=1e-12)


def init_worker():
    global _POLICY
    normalizer = json.loads((SOURCE/'normalizer.json').read_text())
    _POLICY = load_policy(SOURCE/'best_model', normalizer)


def label_position(task):
    path, trials, seed, seconds = task
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        item = json.load(stream)
    doc = item['state']
    game = state_from_json(doc, [SimulationRandomPlayer(Color[c]) for c in doc['colors']])
    legal = ordered_actions(game)
    actions = [next(a for a in legal if action_to_json(a) == payload) for payload in item['actions']]
    seeds = random.Random(seed).getrandbits
    trial_seeds = [seeds(128) for _ in range(trials)]
    started = perf_counter()
    result = evaluate_pair(game, actions, trial_seeds, _POLICY, max_seconds=seconds)
    return {**{k: item[k] for k in ('position_id','game_index','action_index','stratum','actions')},
            **result, 'elapsed_seconds': perf_counter()-started}


def select_positions(root, rows, model, count, seed):
    rng = random.Random(seed)
    pools = {'opening': [], 'middle': [], 'late': []}
    for result in rows:
        index = result['game_index']
        with gzip.open(root/'collection'/f'game_{index:04d}'/'decisions.jsonl.gz', 'rt', encoding='utf-8') as f:
            for row in map(json.loads, f):
                if row['policy'] == 'F' or len(row['inputs']['legal_actions']) < 2:
                    continue
                obs = row['inputs']['observation']
                own = next(p for p in obs['players'] if p['color'] == obs['actor'])
                stratum = 'opening' if obs['initial_phase'] else 'late' if own['actual_vp'] >= 7 else 'middle'
                pools[stratum].append((index, row['action_index'], stratum))
    for pool in pools.values():
        rng.shuffle(pool)
    picked, game_counts = {}, Counter()
    while sum(map(len, picked.values())) < count and any(pools.values()):
        for stratum, pool in pools.items():
            if stratum == 'opening' and sum(s == 'opening' for v in picked.values() for _,s in v) >= count//4:
                pool.clear()
            while pool:
                index, action_index, name = pool.pop()
                if game_counts[index] < 3:
                    picked.setdefault(index, []).append((action_index, name))
                    game_counts[index] += 1
                    break
            if sum(map(len, picked.values())) >= count:
                break
    positions = []
    (root/'positions').mkdir()
    for index, selected in sorted(picked.items()):
        game = create_game(SimulationConfig(players=2), rows[index]['seed'])
        encoder = FeatureEncoder(map_to_json(game.state.board.map),
                                 version=model.normalizer['feature_version'], allow_fewer_players=True)
        wanted = dict(selected)
        with gzip.open(root/'collection'/f'game_{index:04d}'/'decisions.jsonl.gz','rt',encoding='utf-8') as f:
            for row in map(json.loads, f):
                actions = ordered_actions(game)
                if row['action_index'] in wanted:
                    inputs = {'map_ref':0, 'observation':observation(game,game.state.current_color()),
                              'legal_actions':[action_to_json(a) for a in actions]}
                    assert all(inputs[k] == row['inputs'][k] for k in inputs)
                    with forbid_forward_simulation():
                        features = np.asarray([encoder.encode({**inputs,'action':a}) for a in inputs['legal_actions']])
                        scores = model.predict(features)
                    order = sorted(range(len(actions)), key=lambda j:(-scores[j],j))
                    first = order[0]
                    second = next((j for j in order[1:] if not np.array_equal(features[j],features[first])),None)
                    if second is not None:
                        ident = len(positions)
                        item = {'position_id':ident, 'game_index':index,'action_index':row['action_index'],
                                'stratum':wanted[row['action_index']], 'state':state_to_json(game),
                                'actions':[action_to_json(actions[j]) for j in (first,second)],
                                'features':features[[first,second]].tolist(),
                                'baseline_scores':[float(scores[j]) for j in (first,second)]}
                        path = root/'positions'/f'position_{ident:04d}.json.gz'
                        with gzip.open(path,'wt',encoding='utf-8',compresslevel=1) as stream:
                            json.dump(item,stream)
                        positions.append(path)
                game.execute(next(a for a in actions if action_to_json(a) == row['inputs']['action']))
    return positions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=REPO/'runs/402-counterfactual-rl-v1')
    parser.add_argument('--collection-games',type=int,default=32)
    parser.add_argument('--positions',type=int,default=64)
    parser.add_argument('--trials',type=int,default=8)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--seconds-per-position',type=float,default=120)
    parser.add_argument('--validation-games',type=int,default=100)
    parser.add_argument('--test-games',type=int,default=200)
    parser.add_argument('--seed',type=int,default=82_000_000)
    parser.add_argument('--reuse-collection',type=Path,
                        help='reuse only training positions from an earlier counterfactual collection')
    args=parser.parse_args()
    if any(n<2 or n%2 for n in (args.collection_games,args.validation_games,args.test_games)):
        parser.error('even game counts >=2 required')
    if min(args.positions,args.trials,args.workers,args.seconds_per_position)<=0:
        parser.error('positive budgets required')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    norm=json.loads((SOURCE/'normalizer.json').read_text())
    write(root/'normalizer.json',norm)
    baseline=load_policy(SOURCE/'best_model',norm,402)
    excluded,_=exclusions(REPO/'runs/online-sarsa-features-100-v4')
    config={**vars(args),'output':str(root),
            'reuse_collection':str(args.reuse_collection.resolve()) if args.reuse_collection else None,
            'algorithm':'paired counterfactual Monte Carlo policy improvement',
            'continuation':'frozen 402 learner vs unchanged public-belief F',
            'candidate_selection':'402 top two with distinct encoded features',
            'sampling':'opening/middle/late; cap three states per collection game; not selected by outcome',
            'reward':'terminal +1 win / -1 loss; no heuristic target',
            'preference_weight':'absolute paired win-count difference divided by trials+2; uncertain empirical labels',
            'censored_states':'excluded from fitting if any allocated pair is incomplete',
            'excluded_evaluation_used_for_training':False,
            'new_ranker_trees':[8,24], 'new_tree_max_depth':3,
            'validation_seed':args.seed+100_000,'test_seed':args.seed+200_000,
            'selection':'baseline wins ties; final test only for candidate beating validation baseline'}
    config['acceptance']='exact 95% interval lower bound >55% on a predeclared independent final test'
    write(root/'config.json',config)
    started=perf_counter()
    if args.reuse_collection:
        parent=args.reuse_collection.resolve()
        parent_config=json.loads((parent/'config.json').read_text())
        if parent_config['algorithm']!='paired counterfactual Monte Carlo policy improvement':
            raise ValueError('only counterfactual TRAINING collection may be reused')
        collection=json.loads((parent/'collection/summary.json').read_text())
        if not collection['training']:
            raise ValueError('refusing to use an evaluation collection for training')
        rows=sorted([json.loads(line) for line in (parent/'collection/results.jsonl').read_text().splitlines()],
                    key=lambda r:r['game_index'])
        (root/'positions').mkdir()
        positions=[]
        for path in sorted((parent/'positions').glob('position_*.json.gz')):
            target=root/'positions'/path.name
            shutil.copyfile(path,target);positions.append(target)
        print(f'STAGE: reuse {len(positions)} training positions; independent new rollout seeds',flush=True)
    else:
        print('STAGE: new collection games, frozen 402 vs F, epsilon=.03',flush=True)
        rows,collection,_=play_phase(root,'collection',baseline,args.collection_games,args.workers,
            training=True,base_seed=args.seed,excluded=excluded,epsilon=.03,opponent_policy='F',max_seconds=600)
        positions=select_positions(root,rows,baseline,args.positions,args.seed+7)
    if collection['unfinished']:
        raise RuntimeError('collection contains incomplete games')
    write(root/'sampling.json',{'positions':len(positions),'source_games':args.collection_games})
    tasks=[(path,args.trials,args.seed+1_000_000+i*7919,args.seconds_per_position)
           for i,path in enumerate(positions)]
    print(f'STAGE: {len(tasks)} positions, two actions, {args.trials} paired trials each',flush=True)
    labels=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn'),initializer=init_worker) as pool:
        futures=[pool.submit(label_position,task) for task in tasks]
        with (root/'preferences.jsonl').open('w',encoding='utf-8') as stream:
            for future in as_completed(futures):
                result=future.result();labels.append(result)
                stream.write(json.dumps(result)+'\n');stream.flush()
                if len(labels)%8==0 or len(labels)==len(tasks):
                    print(f'LABELS: {len(labels)}/{len(tasks)}, eligible={sum(r["training_eligible"] for r in labels)}, '
                          f'resolved={sum(r["statistically_resolved_05"] for r in labels)}',flush=True)
    labels.sort(key=lambda r:r['position_id'])
    xs,ys,weights=[],[],[]
    for label in labels:
        if not label['training_eligible']:continue
        with gzip.open(positions[label['position_id']],'rt',encoding='utf-8') as stream:item=json.load(stream)
        delta=label['alternative_only_wins']-label['baseline_only_wins']
        xs.extend(item['features']);ys.extend([int(delta<0),int(delta>0)])
        weights.extend([abs(delta)/(args.trials+2)]*2)
    report={'config':config,'collection':collection,'positions':len(labels),
            'preference_groups':len(ys)//2,'resolved_groups':sum(r['statistically_resolved_05'] for r in labels),
            'censored_groups':sum(r['censored_pairs']>0 for r in labels),
            'simulated_actions':sum(r['steps'] for r in labels),
            'rollout_learned_decisions':sum(r['learned_decisions'] for r in labels),
            'rollout_F_decisions':sum(r['F_decisions'] for r in labels)}
    if len(ys)<16:
        report.update(status='insufficient_preference_data',selected_version=402,elapsed_seconds=perf_counter()-started)
        write(root/'report.json',report);print('FINAL: '+json.dumps(report),flush=True);return
    x=np.asarray(xs,dtype=np.float32);y=np.asarray(ys);w=np.asarray(weights)
    np.savez_compressed(root/'preference_training.npz',x=x,y=y,weight=w,groups=np.full(len(y)//2,2))
    models=[baseline]
    params={'objective':'lambdarank','metric':'ndcg','label_gain':[0,1],'learning_rate':.05,
            'num_leaves':7,'max_depth':3,'min_data_in_leaf':8,'min_data_in_bin':1,
            'lambda_l2':10.,'max_delta_step':.5,'feature_fraction':1.,'num_threads':4,
            'verbosity':-1,'seed':20261015,'deterministic':True,'force_col_wise':True,
            'lambdarank_truncation_level':2}
    for trees in (8,24):
        print(f'FIT: {trees} new actor trees from terminal rollout preferences',flush=True)
        dataset=lgb.Dataset(x,label=y,weight=w,group=np.full(len(y)//2,2),free_raw_data=False)
        teacher=lgb.train(params,dataset,num_boost_round=trees,
                          init_model=lgb.Booster(model_file=str(SOURCE/'best_model/ranker.model')),
                          keep_training_booster=True)
        candidate=TreePolicy(teacher,norm,baseline.scale,baseline.mc,baseline.mix,900+trees)
        candidate.policy_name=f'402_counterfactual_actor_{trees}'
        save_policy(candidate,root/f'candidate_{candidate.version}',norm,x)
        models.append(candidate)
    train_maps=[r['initial_map_hash'] for r in rows]
    validations=[];best=baseline;best_wins=-1;validation_maps=[]
    for model in models:
        print(f'STAGE: validation policy {model.version}',flush=True)
        vr,summary,_=play_phase(root,f'validation_{model.version}',model,args.validation_games,args.workers,
            training=False,base_seed=args.seed+100_000,excluded=excluded+train_maps,opponent_policy='F',max_seconds=600)
        validations.append(summary)
        if not validation_maps:validation_maps=[r['initial_map_hash'] for r in vr]
        if not summary['unfinished'] and summary['wins']>best_wins:best=model;best_wins=summary['wins']
        write(root/'validation_history.json',validations)
    report.update(validation=validations,selected_version=best.version)
    if best.version!=402:
        save_policy(best,root/'selected_candidate',norm,x)
        test_excluded=excluded+train_maps+validation_maps
        a,before,_=play_phase(root,'final_test_402',baseline,args.test_games,args.workers,training=False,
            base_seed=args.seed+200_000,excluded=test_excluded,opponent_policy='F',max_seconds=600)
        b,after,_=play_phase(root,'final_test_candidate',best,args.test_games,args.workers,training=False,
            base_seed=args.seed+200_000,excluded=test_excluded,opponent_policy='F',max_seconds=600)
        report.update(final_test_402=before,final_test_candidate=after,paired=paired_stats(a,b),
                      candidate_wilson_95=wilson_interval(after['wins'],after['completed']),
                      acceptance=win_rate_acceptance(after['wins'],after['completed']))
    report.update(status='completed',elapsed_seconds=perf_counter()-started,
                  limitations=['small pilot with noisy rollout preferences',
                               'only two screened actions per sampled state; not an optimal policy',
                               'original 402 remains champion unless independent evidence supports replacement'])
    write(root/'report.json',report);print('FINAL: '+json.dumps(report),flush=True)


if __name__=='__main__':main()
