"""Export only the original frozen 402 numeric trees, without retraining."""
from hashlib import sha256
import json
from pathlib import Path

from catanzero.play_core import ROOT, load_models
from catanzero.features import feature_schema


def lightgbm_tree(root):
    nodes = []

    def visit(node):
        index = len(nodes)
        nodes.append(None)
        if 'leaf_value' in node:
            nodes[index] = [-1, 0, -1, -1, node['leaf_value']]
        else:
            if node['decision_type'] != '<=':
                raise ValueError('Only numeric <= splits are supported.')
            left, right = visit(node['left_child']), visit(node['right_child'])
            nodes[index] = [node['split_feature'], float(node['threshold']), left, right, 0]
        return index

    visit(root)
    return nodes


def main():
    ranker, critic, policy, norm = load_models()
    names = feature_schema(norm['feature_version'])['feature_names']
    assert names == norm['feature_names']
    document = {
        'format': 'catanzero-402-numeric-v1',
        'feature_version': norm['feature_version'], 'feature_names': names,
        'scale': policy['scale'], 'mix': policy['mix'],
        'ranker': [lightgbm_tree(t['tree_structure']) for t in ranker.dump_model()['tree_info']],
        'critic': [[[int(f), float(th), int(l), int(r), float(v[0][0])]
                    for f, th, l, r, v in zip(est.tree_.feature, est.tree_.threshold,
                                              est.tree_.children_left, est.tree_.children_right,
                                              est.tree_.value)] for est in critic.estimators_],
        'source_sha256': {p.name: sha256(p.read_bytes()).hexdigest()
                          for p in (ROOT/'models/champion402').rglob('*')
                          if p.is_file() and p.suffix in ('.model', '.joblib')},
        'note': 'Original frozen 402; unnormalized raw features; preferences, not probabilities.'
    }
    output = ROOT/'models/champion402/portable.json'
    output.write_text(json.dumps(document, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(output), 'bytes': output.stat().st_size,
                      'ranker_trees': len(document['ranker']), 'critic_trees': len(document['critic'])}))


if __name__ == '__main__':
    main()
