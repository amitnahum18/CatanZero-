import json
import random
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from continue_champion_tree_rl import SOURCE,load_policy
from optimize_tree_actor_rl import SamplingPolicy


def test_batch_scoring_preserves_actual_model_scores_choices_and_sampling_probabilities():
    norm=json.loads((SOURCE/'normalizer.json').read_text());model=load_policy(SOURCE/'best_model',norm)
    root=SOURCE.parent/'tree-policy-actor-rl-v5'
    with np.load(root/'actor_data.npz') as d:
        groups=[d['x'][a:a+n] for a,n in zip(d['starts'][:8],d['lengths'][:8])]
    responses=[{'features':x} for x in groups]
    for policy in (model,SamplingPolicy(model,.25)):
        expected=[policy.choose(r,random.Random(i),0) for i,r in enumerate(responses)]
        actual=policy.choose_many(responses,[random.Random(i) for i in range(len(responses))],0)
        for a,b in zip(actual,expected):
            assert a['index']==b['index']
            np.testing.assert_array_equal(a['scores'],b['scores'])
            if 'probability' in a:assert a['probability']==b['probability']
