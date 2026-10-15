"""Train an RL-only LightGBM continuation with a depth-12 value objective."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')

import json
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow.models import infer_signature
from scipy.stats import binomtest

from optimize_tree_policy_vs_f import TreePolicy, TreePolicyModel, monte_carlo_data
from train_residual_tree_fqi import REPO, play_phase, write
from train_three_game_sarsa import exclusions
from catanzero.duel_protocol import wilson_interval


ROOT = REPO / 'runs/lightgbm-rl-depth12-v1'
PARENT = REPO / 'runs/tree-policy-trade-aware-v3'
TRAIN_GAMES, VALIDATION_GAMES, TEST_GAMES = 500, 200, 500
WORKERS = 8
TRAIN_SEED, VALIDATION_SEED, TEST_SEED = 66_000_000, 67_000_000, 68_000_000
EPSILON = .10
ADDED_TREE_COUNTS = (200, 350, 500)


class RLOnlyLightGBMPolicy:
    policy_name = 'LightGBM_MC_RL_Depth12'

    def __init__(self, booster, normalizer, version):
        self.booster, self.normalizer, self.version = booster, normalizer, version

    def predict(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        return self.booster.predict(raw, num_threads=1) if len(raw) else np.empty(0)

    def choose(self, response, rng, epsilon):
        raw = np.asarray(response['features'], dtype=np.float64)
        scores = self.predict(raw)
        explored = rng.random() < epsilon
        selected = rng.randrange(len(raw)) if explored else int(np.argmax(scores))
        return {'index': selected, 'raw': raw[selected], 'scores': scores.tolist(),
                'version': self.version, 'explored': explored}

    def save(self, directory):
        directory.mkdir(exist_ok=True)
        self.booster.save_model(str(directory / 'model.txt'))
        write(directory / 'policy.json', {
            'kind': 'single_lightgbm_rl_value', 'version': self.version,
            'input_columns': 264, 'base_trees': 114,
            'added_rl_trees': self.booster.num_trees() - 114,
            'total_trees': self.booster.num_trees(), 'configured_max_depth': 12,
            'actual_max_depth': max(tree_depth(info['tree_structure'])
                                    for info in self.booster.dump_model()['tree_info']),
            'training_objective': 'regression on real-game Monte Carlo returns only',
            'scores': 'action preferences, not calibrated win probabilities',
        })


def tree_depth(node):
    if 'split_index' not in node:
        return 0
    return 1 + max(tree_depth(node['left_child']), tree_depth(node['right_child']))


class LightGBMRLModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.booster = lgb.Booster(model_file=context.artifacts['model'])

    def predict(self, context, model_input, params=None):
        raw = np.asarray(model_input, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 264 or not np.isfinite(raw).all():
            raise ValueError('expected finite raw predecision features, shape (n,264)')
        return self.booster.predict(raw, num_threads=1)


def fit_candidate(data_x, target, weight, base_booster, normalizer, added_trees, version):
    estimator = lgb.LGBMRegressor(
        objective='regression', n_estimators=added_trees, learning_rate=.03,
        max_depth=12, num_leaves=255, min_child_samples=32,
        reg_lambda=2.0, max_bin=127, colsample_bytree=.9,
        subsample=.85, subsample_freq=1, n_jobs=4, random_state=20261015 + version,
        verbosity=-1, deterministic=True, force_col_wise=True,
    )
    estimator.fit(data_x, target, sample_weight=weight, init_model=base_booster)
    return RLOnlyLightGBMPolicy(estimator.booster_, normalizer, version)


def main():
    if ROOT.exists():
        raise SystemExit(f'Output already exists: {ROOT}')
    started = perf_counter()
    ROOT.mkdir(parents=True)
    parent_doc = json.loads((PARENT / 'best_model/policy.json').read_text())
    normalizer = json.loads((PARENT / 'normalizer.json').read_text())
    base_ranker = lgb.Booster(model_file=str(PARENT / 'best_model/ranker.model'))
    assert base_ranker.num_trees() == 114 and parent_doc['version'] == 402
    # 402 is used only to generate behavior trajectories; the new policy has a
    # single LightGBM booster and does not include the old ExtraTrees critic.
    behavior = TreePolicy(base_ranker, normalizer, parent_doc['scale'],
                          __import__('joblib').load(PARENT / 'best_model/monte_carlo.joblib'),
                          parent_doc['mix'], 402)
    excluded, manifest = exclusions(REPO / 'runs/online-sarsa-features-100-v4')
    write(ROOT / 'normalizer.json', normalizer)
    config = {
        'algorithm': 'LightGBM fitted Monte Carlo control/value improvement initialized from teacher ranker',
        'opponent': 'native F', 'players': 2, 'training_games': TRAIN_GAMES,
        'epsilon': EPSILON, 'validation_games_per_policy': VALIDATION_GAMES,
        'paired_test_games_per_policy': TEST_GAMES, 'target_win_rate': .60,
        'training_seed': TRAIN_SEED, 'validation_seed': VALIDATION_SEED, 'test_seed': TEST_SEED,
        'initial_model': str(PARENT / 'best_model/ranker.model'),
        'initial_weights_are_permitted_warm_start': True,
        'new_tree_training_labels': 'only actual-game MC returns from the fresh RL collection',
        'new_tree_objective': 'L2 regression on outcome - current potential',
        'extra_trees_or_second_model_in_candidate': False,
        'added_lightgbm_tree_counts': list(ADDED_TREE_COUNTS),
        'configured_max_depth': 12,
        'reward': 'terminal +1/-1; gamma 1; Phi=.05 own actual VP + .02 own-public-opponent VP gap',
        'selection': 'highest wins on fresh 200-game validation; incumbent retained on ties; separate paired 500-board test',
        'promotion_rule': 'promote only if pure-LightGBM challenger wins >=60% and exceeds incumbent wins on paired heldout boards',
        'excluded_board_hashes': excluded,
    }
    write(ROOT / 'config.json', config)
    mlflow.set_tracking_uri(manifest['tracking_uri'])
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name('CatanZero_TreePolicy_NativeF')
    mlflow.set_experiment(experiment_id=experiment.experiment_id)

    with mlflow.start_run(run_name='LightGBM_depth12_RL_only_to_60pct') as run:
        write(ROOT / 'tracking.json', {'run_id': run.info.run_id, 'tracking_uri': manifest['tracking_uri']})
        mlflow.log_params({'parent_version': 402, 'training_games': TRAIN_GAMES,
                           'epsilon': EPSILON, 'opponent': 'native F',
                           'objective': 'RL Monte Carlo return regression',
                           'max_depth': 12, 'added_tree_candidates': str(ADDED_TREE_COUNTS)})
        mlflow.log_artifact(str(ROOT / 'config.json'), 'protocol_before_training')

        print(f'STAGE: collect {TRAIN_GAMES} fresh games using frozen 402, epsilon={EPSILON}', flush=True)
        training_results, training_summary, _ = play_phase(
            ROOT, 'RL_training', behavior, TRAIN_GAMES, WORKERS,
            training=True, base_seed=TRAIN_SEED, excluded=excluded, epsilon=EPSILON)
        training_hashes = [row['initial_map_hash'] for row in training_results]
        assert len(training_hashes) == TRAIN_GAMES and len(set(training_hashes)) == TRAIN_GAMES
        x, target, weight, ids = monte_carlo_data(ROOT, ['RL_training'])
        assert len(x) > 10000 and len(x) == len(target) == len(weight) == len(ids)
        np.savez_compressed(ROOT / 'rl_returns.npz', x=x, target=target, weight=weight)
        write(ROOT / 'rl_return_rows.jsonl', ''.join(json.dumps(row) + '\n' for row in ids))
        write(ROOT / 'rl_training_summary.json', {
            'games': TRAIN_GAMES, 'wins_during_exploration': training_summary['wins'],
            'completed': training_summary['completed'], 'return_rows': len(x),
            'unique_games_with_targets': len({row['game_index'] for row in ids}),
            'input_rows_are_only_executed_actions': True,
            'targets_use_only_actual_terminal_outcomes_and_predecision_potential': True,
        })
        print(f"RL DATA: {len(x)} actual-action returns from {training_summary['completed']} complete games", flush=True)

        candidates = [behavior]
        for i, added_trees in enumerate(ADDED_TREE_COUNTS, start=1):
            version = 1200 + i
            print(f'FIT: LightGBM RL candidate {version}, {added_trees} new depth-12 trees', flush=True)
            model = fit_candidate(x, target, weight, base_ranker, normalizer, added_trees, version)
            model.save(ROOT / f'candidate_{version}')
            assert model.booster.num_trees() == 114 + added_trees
            candidates.append(model)

        validation_results = []
        selected = None
        selected_wins = -1
        validation_hashes = []
        for model in candidates:
            rows, summary, _ = play_phase(
                ROOT, f'validation_{model.version}', model, VALIDATION_GAMES, WORKERS,
                training=False, base_seed=VALIDATION_SEED,
                excluded=excluded + training_hashes)
            if not validation_hashes:
                validation_hashes = [r['initial_map_hash'] for r in rows]
            entry = {'version': model.version, 'policy': 'hybrid_incumbent' if model.version == 402 else 'single_LightGBM_RL',
                     'wins': summary['wins'], 'games': summary['completed'],
                     'unfinished': summary['unfinished'],
                     'tree_count': None if model.version == 402 else model.booster.num_trees(),
                     'actual_max_depth': None if model.version == 402 else max(
                         tree_depth(t['tree_structure']) for t in model.booster.dump_model()['tree_info'])}
            validation_results.append(entry)
            if summary['unfinished'] == 0 and summary['wins'] > selected_wins:
                selected, selected_wins = model, summary['wins']
            print(f"VALIDATION {model.version}: {summary['wins']}/{summary['completed']} wins", flush=True)
        write(ROOT / 'validation.json', validation_results)
        assert selected is not None
        print(f'STAGE: paired 500-board test, incumbent 402 vs selected {selected.version}', flush=True)
        incumbent_rows, incumbent, _ = play_phase(
            ROOT, 'final_test_402', behavior, TEST_GAMES, WORKERS, training=False,
            base_seed=TEST_SEED, excluded=excluded + training_hashes + validation_hashes)
        selected_rows, challenger, _ = play_phase(
            ROOT, 'final_test_challenger', selected, TEST_GAMES, WORKERS, training=False,
            base_seed=TEST_SEED, excluded=excluded + training_hashes + validation_hashes)
        assert [r['initial_map_hash'] for r in incumbent_rows] == [r['initial_map_hash'] for r in selected_rows]
        assert [r['learner_seat'] for r in incumbent_rows] == [r['learner_seat'] for r in selected_rows]
        paired = {'challenger_only_wins': 0, 'incumbent_only_wins': 0, 'same_outcome': 0}
        for a, b in zip(incumbent_rows, selected_rows):
            if a['learner_won'] == b['learner_won']:
                paired['same_outcome'] += 1
            elif b['learner_won']:
                paired['challenger_only_wins'] += 1
            else:
                paired['incumbent_only_wins'] += 1
        challenger_rate, incumbent_rate = challenger['wins'] / TEST_GAMES, incumbent['wins'] / TEST_GAMES
        promotion = (selected.version != 402 and challenger_rate >= .60 and
                     challenger['wins'] > incumbent['wins'] and challenger['unfinished'] == 0 and
                     incumbent['unfinished'] == 0)
        report = {
            'initial_version': 402, 'selected_version': selected.version,
            'selected_model_is_single_lightgbm': selected.version != 402,
            'selected_tree_count': None if selected.version == 402 else selected.booster.num_trees(),
            'selected_actual_max_depth': None if selected.version == 402 else max(
                tree_depth(t['tree_structure']) for t in selected.booster.dump_model()['tree_info']),
            'fresh_RL_training': training_summary,
            'RL_return_rows': len(x), 'validation_candidates': validation_results,
            'paired_final_test': {
                'games_per_policy': TEST_GAMES, 'same_boards_and_seats': True,
                'incumbent_402': {'wins': incumbent['wins'], 'win_rate': incumbent_rate,
                                  'CI95': wilson_interval(incumbent['wins'], TEST_GAMES),
                                  'unfinished': incumbent['unfinished']},
                'challenger': {'wins': challenger['wins'], 'win_rate': challenger_rate,
                               'CI95': wilson_interval(challenger['wins'], TEST_GAMES),
                               'unfinished': challenger['unfinished']},
                'paired_outcomes': paired,
                'discordant_exact_binomial_p': (float(binomtest(paired['challenger_only_wins'],
                    paired['challenger_only_wins'] + paired['incumbent_only_wins'], .5).pvalue)
                    if paired['challenger_only_wins'] + paired['incumbent_only_wins'] else 1.),
            },
            'target_60pct_reached': selected.version != 402 and challenger_rate >= .60 and challenger['unfinished'] == 0,
            'promote_to_best_pilot': promotion,
            'leakage_controls': {'fresh_training_after_prior_board_exclusions': True,
                                 'validation_and_test_disjoint_from_training': True,
                                 'validation_and_test_disjoint_from_each_other': True,
                                 'test_used_for_selection': False},
            'elapsed_seconds': perf_counter() - started,
        }
        write(ROOT / 'report.json', report)
        mlflow.log_metrics({'rl_training_wins': training_summary['wins'],
                            'validation_wins': selected_wins,
                            'incumbent_paired_test_win_rate': incumbent_rate,
                            'challenger_paired_test_win_rate': challenger_rate,
                            'target_60pct_reached': float(report['target_60pct_reached']),
                            'promoted': float(promotion)})
        for filename in ('report.json', 'validation.json', 'rl_training_summary.json'):
            mlflow.log_artifact(str(ROOT / filename), 'results')

        if promotion:
            selected.save(ROOT / 'promoted_model')
            expected = selected.predict(x[:512].astype(np.float64))
            info = mlflow.pyfunc.log_model(
                name='model', python_model=LightGBMRLModel(),
                artifacts={'model': str(ROOT / 'promoted_model/model.txt')},
                input_example=x[:3].astype(np.float64), signature=infer_signature(x[:512], expected),
                pip_requirements=['mlflow==' + mlflow.__version__, 'numpy==' + np.__version__,
                                  'lightgbm==' + lgb.__version__],
                code_paths=[str(Path(__file__).resolve())])
            restored = mlflow.pyfunc.load_model(info.model_uri)
            reload_error = float(np.max(np.abs(restored.predict(x[:512].astype(np.float64)) - expected)))
            assert reload_error < 1e-12
            registered = mlflow.register_model(info.model_uri, 'CatanZero_TreePolicy_NativeF')
            client.set_registered_model_alias('CatanZero_TreePolicy_NativeF', 'best_pilot', str(registered.version))
            report['registration'] = {'version': str(registered.version), 'alias': 'best_pilot',
                                      'model_uri': info.model_uri, 'reload_max_abs_error': reload_error}
            write(ROOT / 'report.json', report)
            mlflow.log_artifact(str(ROOT / 'report.json'), 'results')
        print('RESULT: ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
