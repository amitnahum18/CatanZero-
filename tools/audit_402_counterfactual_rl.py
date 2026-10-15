"""Verify counterfactual return labels, predecision inputs, model lineage and splits."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random

import joblib
import lightgbm as lgb
import numpy as np
from catanatron import Color
from catanatron.serialization import state_from_json, map_to_json
from catanzero.counterfactual_402 import pair_statistics, terminal_rollout
from catanzero.dataset import observation, ordered_actions
from catanzero.features import FeatureEncoder
from catanzero.simulation import SimulationRandomPlayer
from catanatron.serialization import action_to_json
from continue_champion_tree_rl import load_policy
from optimize_tree_policy_vs_f import TreePolicy
from train_residual_tree_fqi import REPO, write

SOURCE=REPO/'runs/tree-policy-trade-aware-v3'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    root=parser.parse_args().root.resolve()
    report=json.loads((root/'report.json').read_text())
    config=report['config']
    norm=json.loads((SOURCE/'normalizer.json').read_text())
    baseline=load_policy(SOURCE/'best_model',norm,402)
    source_report=json.loads((SOURCE/'report.json').read_text())
    source_hashes={name:hashlib.sha256((SOURCE/'best_model'/name).read_bytes()).hexdigest()
                   for name in ('ranker.model','monte_carlo.joblib','policy.json')}
    assert source_hashes==source_report['model_hashes']
    labels=sorted([json.loads(line) for line in (root/'preferences.jsonl').read_text().splitlines()],
                  key=lambda row:row['position_id'])
    xs,ys,weights=[],[],[]
    for label in labels:
        stats=pair_statistics(label['trial_outcomes'])
        assert all(stats[k]==label[k] for k in stats)
        path=root/'positions'/f'position_{label["position_id"]:04d}.json.gz'
        with gzip.open(path,'rt',encoding='utf-8') as stream:item=json.load(stream)
        game=state_from_json(item['state'],[SimulationRandomPlayer(Color[c]) for c in item['state']['colors']])
        actor=game.state.current_color()
        inputs={'map_ref':0,'observation':observation(game,actor),
                'legal_actions':[action_to_json(a) for a in ordered_actions(game)]}
        encoder=FeatureEncoder(map_to_json(game.state.board.map),norm['feature_version'],allow_fewer_players=True)
        for payload,features in zip(item['actions'],item['features']):
            assert payload in inputs['legal_actions']
            np.testing.assert_array_equal(encoder.encode({**inputs,'action':payload}),features)
        if label['training_eligible']:
            delta=label['alternative_only_wins']-label['baseline_only_wins']
            xs.extend(item['features']);ys.extend([int(delta<0),int(delta>0)])
            weights.extend([abs(delta)/(config['trials']+2)]*2)
    matrix=np.load(root/'preference_training.npz')
    np.testing.assert_array_equal(matrix['x'],np.asarray(xs,dtype=np.float32))
    np.testing.assert_array_equal(matrix['y'],ys)
    np.testing.assert_array_equal(matrix['weight'],weights)
    assert len(matrix['groups'])==len(ys)//2 and (matrix['groups']==2).all()

    original_trees=baseline.teacher.dump_model()['tree_info']
    model_checks={}
    for directory in sorted(root.glob('candidate_*')):
        if not directory.is_dir():continue
        info=json.loads((directory/'policy.json').read_text())
        ranker=directory/'ranker.model'
        if not ranker.exists():ranker=directory/'ranker.txt'
        teacher=lgb.Booster(model_file=str(ranker))
        trees=teacher.dump_model()['tree_info']
        assert trees[:len(original_trees)]==original_trees
        def depth(node):
            return 0 if 'leaf_value' in node else 1+max(depth(node['left_child']),depth(node['right_child']))
        assert all(depth(t['tree_structure'])<=3 for t in trees[len(original_trees):])
        mc=joblib.load(directory/'monte_carlo.joblib')
        np.testing.assert_array_equal(mc.predict(matrix['x']),baseline.mc.predict(matrix['x']))
        model=TreePolicy(teacher,norm,info['scale'],mc,info['mix'],info['version'])
        np.testing.assert_allclose(model.predict(matrix['x']),
            teacher.predict(matrix['x'],num_threads=1)/baseline.scale+baseline.mix*baseline.mc.predict(matrix['x']),
            rtol=0,atol=1e-12)
        model_checks[str(info['version'])]={'original_trees_preserved':len(original_trees),
                                          'new_trees':len(trees)-len(original_trees),
                                          'critic_predictions_unchanged':True}
    source=root if (root/'collection').exists() else Path(config['reuse_collection'])
    train_rows=[json.loads(line) for line in (source/'collection/results.jsonl').read_text().splitlines()]
    train_maps={r['initial_map_hash'] for r in train_rows}
    validation_maps=set()
    for directory in root.glob('validation_*'):
        if directory.is_dir():
            validation_maps.update(json.loads(line)['initial_map_hash']
                                   for line in (directory/'results.jsonl').read_text().splitlines())
    assert not train_maps & validation_maps
    final_maps=set()
    for directory in root.glob('final_test_*'):
        if directory.is_dir():
            final_maps.update(json.loads(line)['initial_map_hash']
                              for line in (directory/'results.jsonl').read_text().splitlines())
    assert not final_maps & (train_maps|validation_maps)
    replayed=0
    eligible=[r for r in labels if r['training_eligible']]
    for label in (eligible[0],eligible[-1]):
        with gzip.open(root/'positions'/f'position_{label["position_id"]:04d}.json.gz','rt',encoding='utf-8') as stream:
            item=json.load(stream)
        doc=item['state']
        game=state_from_json(doc,[SimulationRandomPlayer(Color[c]) for c in doc['colors']])
        seed=random.Random(config['seed']+1_000_000+label['position_id']*7919).getrandbits(128)
        for i,payload in enumerate(item['actions']):
            action=next(a for a in ordered_actions(game) if action_to_json(a)==payload)
            actual=terminal_rollout(game,action,seed,baseline)
            assert actual['outcome']==label['trial_outcomes'][i][0]
            replayed+=1
    audit={'status':'passed','predecision_states_reencoded':len(labels),
           'paired_return_labels_verified':len(labels),'training_groups':len(ys)//2,
           'sampled_terminal_rollouts_replayed':replayed,'models':model_checks,
           'train_validation_test_maps_disjoint':True,'original_402_hashes_unchanged':True,
           'scope':'all preference accounting and features; all original actor trees; four sampled full rollout replays',
           'limitations':'not a proof of optimality or a calibrated preference-confidence guarantee'}
    write(root/'audit.json',audit)
    print(json.dumps(audit))


if __name__=='__main__':main()
