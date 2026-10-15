"""Read-only diagnostics of 402's representation and correction on training states.

These are in-sample diagnostics, never performance estimates or new labels.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import json
import gzip
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def stochastic_label_example():
    from catanatron.players.value import get_value_fn
    from catanatron.serialization import action_to_json
    from catanzero.dataset import ordered_actions
    from catanzero.simulation import create_game, SimulationConfig
    from catanzero.search_labels import live_belief
    from catanzero.label_f_policy import detached_label_copy
    seed, target = 46_000_000, 39
    game = create_game(SimulationConfig(players=2), seed)
    trace = REPO/'runs/residual-extratrees-depth10-f-duel-v1/training_round_1/game_0000/decisions.jsonl.gz'
    with gzip.open(trace, 'rt', encoding='utf-8') as stream:
        for row in map(json.loads, stream):
            if row['action_index'] == target:
                break
            game.execute(next(a for a in ordered_actions(game) if action_to_json(a) == row['inputs']['action']))
    actor = game.state.current_color()
    actions = ordered_actions(game)
    value = get_value_fn('base_fn', None)
    result = {'seed': seed, 'action_index': target, 'candidate_indices': [8, 11],
              'candidate_actions': [action_to_json(actions[i]) for i in (8, 11)]}
    for mode in ('legacy_forward', 'legacy_reverse', 'fixed_forward', 'fixed_reverse'):
        view = live_belief(game, actor, seed*1_000_003+target)
        scores = {}
        order = reversed(range(len(actions))) if mode.endswith('reverse') else range(len(actions))
        for i in order:
            child = detached_label_copy(view) if mode.startswith('fixed') else view.copy()
            child.execute(actions[i])
            scores[i] = value(child, actor)
        result[mode] = {'pair_scores': [scores[i] for i in (8, 11)],
                        'best_index': max(range(len(actions)), key=scores.get)}
    return result


def main():
    source = REPO / 'runs/tree-policy-trade-aware-v3'
    output = REPO / 'runs/402-project-review-20261015'
    output.mkdir(exist_ok=True)
    names = json.loads((source/'normalizer.json').read_text())['feature_names']
    policy = json.loads((source/'best_model/policy.json').read_text())
    data = np.load(source/'teacher_data.npz')
    x, y, groups = data['x'], data['y'], data['groups']
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    variation = np.maximum.reduceat(x, starts, axis=0) != np.minimum.reduceat(x, starts, axis=0)
    action_sensitive = variation.any(axis=0)
    forest = joblib.load(source/'best_model/monte_carlo.joblib')
    ranker = lgb.Booster(model_file=str(source/'best_model/ranker.model'))
    importance = forest.feature_importances_
    gain = ranker.feature_importance(importance_type='gain')

    indexes = np.linspace(0, len(groups)-1, min(5000, len(groups)), dtype=int)
    chunks = [x[starts[i]:starts[i]+groups[i]] for i in indexes]
    sampled = np.concatenate(chunks)
    base = ranker.predict(sampled, num_threads=1)/policy['scale']
    correction = policy['mix']*forest.predict(sampled)
    changed = aliases = conflicting_aliases = top_conflicts = 0
    aliases_by_action = {}
    correction_spread, ranking_spread = [], []
    pos = 0
    action_names = ('ROLL','MOVE_ROBBER','DISCARD_RESOURCE','BUILD_ROAD','BUILD_SETTLEMENT',
                    'BUILD_CITY','BUY_DEVELOPMENT_CARD','PLAY_KNIGHT_CARD','PLAY_YEAR_OF_PLENTY',
                    'PLAY_MONOPOLY','PLAY_ROAD_BUILDING','MARITIME_TRADE','END_TURN')
    action_columns = [names.index('action_'+name) for name in action_names]
    for i, chunk in zip(indexes, chunks):
        n = len(chunk)
        a, b = base[pos:pos+n], correction[pos:pos+n]
        changed += int(np.argmax(a) != np.argmax(a+b))
        correction_spread.append(float(np.ptp(b)))
        ranking_spread.append(float(np.ptp(a)))
        labels = y[starts[i]:starts[i]+n]
        _, inverse, counts = np.unique(chunk, axis=0, return_inverse=True, return_counts=True)
        if (counts > 1).any():
            aliases += 1
            conflict = best_conflict = False
            for j in np.flatnonzero(counts > 1):
                rows = np.flatnonzero(inverse == j)
                if np.ptp(labels[rows]) > 0:
                    conflict = True
                    best_conflict |= bool(labels[rows].max() == labels.max())
                    kind = action_names[int(np.argmax(chunk[rows[0], action_columns]))]
                    aliases_by_action[kind] = aliases_by_action.get(kind, 0)+1
            conflicting_aliases += int(conflict)
            top_conflicts += int(best_conflict)
        pos += n

    report = {
        'scope': 'training-state diagnostics only; no evaluation states used for fitting or labels',
        'teacher_rows': int(len(x)), 'teacher_decision_groups': int(len(groups)),
        'diagnostic_sample_groups': len(indexes), 'diagnostic_sample_rows': len(sampled),
        'features_varying_across_actions_in_some_teacher_group': int(action_sensitive.sum()),
        'features_constant_within_every_teacher_group': int((~action_sensitive).sum()),
        'critic_importance_on_state_only_features': float(importance[~action_sensitive].sum()),
        'ranker_gain_share_on_state_only_features': float(gain[~action_sensitive].sum()/gain.sum()),
        'groups_where_original_critic_changes_ranker_argmax': changed,
        'critic_action_spread_median': float(np.median(correction_spread)),
        'ranker_action_spread_median': float(np.median(ranking_spread)),
        'groups_with_identical_feature_rows_for_different_candidates': aliases,
        'groups_with_identical_feature_rows_but_different_teacher_relevance': conflicting_aliases,
        'groups_where_feature_alias_conflict_includes_top_relevance': top_conflicts,
        'conflicting_alias_sets_by_action': aliases_by_action,
        'critic_top_features': [{'name': names[i], 'importance': float(importance[i]),
                                'varies_across_actions': bool(action_sensitive[i])}
                               for i in np.argsort(importance)[-15:][::-1]],
        'interpretation_limits': [
            'state-only features can still support useful conditional action interactions',
            'teacher label agreement is not true action optimality',
            'candidate argmax change on training states is not a win-rate estimate',
            'aliases are counted at the serialized float32 teacher representation',
        ],
        'stochastic_teacher_rng_reproduction': stochastic_label_example(),
    }
    (output/'diagnostics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
