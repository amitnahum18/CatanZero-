"""Continue real-game Monte Carlo learning from the current champion (402)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')

import json
from pathlib import Path
from time import perf_counter

import joblib
import lightgbm as lgb
import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow.models import infer_signature
from scipy.stats import binomtest
from sklearn.ensemble import ExtraTreesRegressor

from optimize_tree_policy_vs_f import TreePolicy, TreePolicyModel, monte_carlo_data
from train_residual_tree_fqi import REPO, play_phase, write
from train_three_game_sarsa import exclusions
from catanzero.duel_protocol import wilson_interval


PARENT = REPO / 'runs/tree-policy-trade-aware-v3'
ROOT = REPO / 'runs/tree-rl-continue-v2'
TRAIN_GAMES = 200
VALIDATION_GAMES = 100
TEST_GAMES = 500
WORKERS = 8
TRAIN_SEED = 60_000_100
VALIDATION_SEED = 61_000_300
TEST_SEED = 62_000_000
EPSILON = .05
MIXES = (.10, .20, .30, .40, .60)


def outcomes(root, phase):
    return [json.loads(s) for s in (root / phase / 'results.jsonl').read_text().splitlines()]


def main():
    if ROOT.exists():
        raise SystemExit(f'Output already exists: {ROOT}')
    start = perf_counter()
    ROOT.mkdir(parents=True)
    parent_policy = json.loads((PARENT / 'best_model/policy.json').read_text())
    normalizer = json.loads((PARENT / 'normalizer.json').read_text())
    champion = TreePolicy(lgb.Booster(model_file=str(PARENT / 'best_model/ranker.model')),
                          normalizer, parent_policy['scale'],
                          joblib.load(PARENT / 'best_model/monte_carlo.joblib'),
                          parent_policy['mix'], 402)
    excluded, manifest = exclusions(REPO / 'runs/online-sarsa-features-100-v4')
    write(ROOT / 'normalizer.json', normalizer)
    config = {
        'algorithm': 'fresh real-game MC RL continuation from frozen champion 402',
        'parent_model': str(PARENT / 'best_model'), 'opponent': 'native F', 'players': 2,
        'training_games': TRAIN_GAMES, 'epsilon': EPSILON,
        'validation_games': VALIDATION_GAMES, 'paired_test_games_per_policy': TEST_GAMES,
        'target_win_rate': .60, 'seeds': {'training': TRAIN_SEED, 'validation': VALIDATION_SEED, 'test': TEST_SEED},
        'mix_candidates': list(MIXES), 'reward': 'terminal +1/-1, gamma 1, potential .05 own VP + .02 public VP gap',
        'selection': 'maximum wins on one fresh validation set; final paired comparison on a separate 500 boards',
        'promotion_rule': 'promote only if challenger reaches 60% in paired heldout games and wins more paired games than incumbent',
        'excluded_board_hashes': excluded,
    }
    write(ROOT / 'config.json', config)
    mlflow.set_tracking_uri(manifest['tracking_uri'])
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name('CatanZero_TreePolicy_NativeF')
    mlflow.set_experiment(experiment_id=experiment.experiment_id)

    with mlflow.start_run(run_name='continue_MC_RL_from_champion_402_to_60pct') as run:
        write(ROOT / 'tracking.json', {'run_id': run.info.run_id, 'tracking_uri': manifest['tracking_uri']})
        mlflow.log_params({'parent_version': 402, 'training_games': TRAIN_GAMES, 'epsilon': EPSILON,
                           'opponent': 'native F', 'target_win_rate': .60, 'test_games_per_policy': TEST_GAMES})
        mlflow.log_artifact(str(ROOT / 'config.json'), 'protocol_before_training')

        print(f'STAGE: collect {TRAIN_GAMES} fresh games with frozen 402, epsilon={EPSILON}', flush=True)
        train_results, train_summary, _ = play_phase(
            ROOT, 'RL_round_1', champion, TRAIN_GAMES, WORKERS, training=True,
            base_seed=TRAIN_SEED, excluded=excluded, epsilon=EPSILON)
        train_hashes = [r['initial_map_hash'] for r in train_results]
        assert len(train_hashes) == TRAIN_GAMES and len(set(train_hashes)) == TRAIN_GAMES

        new_x, new_target, new_weight, new_ids = monte_carlo_data(ROOT, ['RL_round_1'])
        old = np.load(REPO / 'runs/tree-policy-native-f-max-v1/MC_round_2.npz')
        x = np.concatenate([old['x'], new_x]).astype(np.float32)
        target = np.concatenate([old['target'], new_target]).astype(np.float32)
        weight = np.concatenate([old['weight'], new_weight]).astype(np.float64)
        mc = ExtraTreesRegressor(n_estimators=64, max_depth=10, min_samples_leaf=16,
                                 max_features=.75, n_jobs=4, random_state=2060)
        mc.fit(x, target, sample_weight=weight)
        mc.n_jobs = 1
        joblib.dump(mc, ROOT / 'continued_mc.joblib', compress=3)
        np.savez_compressed(ROOT / 'mc_training_data.npz', x=x, target=target, weight=weight)
        write(ROOT / 'mc_training_sources.json', {
            'old_source': 'tree-policy-native-f-max-v1/MC_round_2.npz',
            'old_rows': int(len(old['x'])), 'fresh_games': TRAIN_GAMES,
            'fresh_transitions': int(len(new_x)), 'fresh_wins': train_summary['wins'],
            'fresh_target_rows': int(len(new_x)), 'combined_rows': int(len(x)),
            'fresh_target_games': len(set(i['game_index'] for i in new_ids)),
        })
        print(f'MC UPDATE: {len(new_x)} fresh real-game returns; {len(x)} total rows', flush=True)

        candidates = [champion]
        for index, mix in enumerate(MIXES, start=403):
            model = TreePolicy(champion.teacher, normalizer, champion.scale, mc, mix, index)
            candidates.append(model)

        history = []
        for model in candidates:
            rows, summary, _ = play_phase(
                ROOT, f'validation_candidate_{model.version}', model, VALIDATION_GAMES, WORKERS,
                training=False, base_seed=VALIDATION_SEED,
                excluded=excluded + train_hashes)
            history.append({'version': model.version, 'mix': model.mix,
                            'wins': summary['wins'], 'games': summary['completed'],
                            'unfinished': summary['unfinished']})
            print(f"VALIDATION {model.version}: {summary['wins']}/{summary['completed']} wins", flush=True)
        write(ROOT / 'validation.json', history)
        chosen_record = max(enumerate(history), key=lambda pair: (pair[1]['wins'], -pair[0]))[1]
        chosen = next(model for model in candidates if model.version == chosen_record['version'])
        validation_hashes = [r['initial_map_hash'] for r in
                             outcomes(ROOT, f'validation_candidate_{chosen.version}')]

        print(f'STAGE: paired fresh test of incumbent 402 and challenger {chosen.version}', flush=True)
        incumbent_results, incumbent_summary, _ = play_phase(
            ROOT, 'final_test_incumbent', champion, TEST_GAMES, WORKERS,
            training=False, base_seed=TEST_SEED,
            excluded=excluded + train_hashes + validation_hashes)
        challenger_results, challenger_summary, _ = play_phase(
            ROOT, 'final_test_challenger', chosen, TEST_GAMES, WORKERS,
            training=False, base_seed=TEST_SEED,
            excluded=excluded + train_hashes + validation_hashes)
        assert [r['initial_map_hash'] for r in incumbent_results] == [r['initial_map_hash'] for r in challenger_results]
        assert all(a['learner_seat'] == b['learner_seat'] for a, b in zip(incumbent_results, challenger_results))
        paired = {'challenger_only_wins': 0, 'incumbent_only_wins': 0, 'same_outcome': 0}
        for a, b in zip(incumbent_results, challenger_results):
            aw, bw = a['learner_won'], b['learner_won']
            if aw == bw:
                paired['same_outcome'] += 1
            elif bw:
                paired['challenger_only_wins'] += 1
            else:
                paired['incumbent_only_wins'] += 1
        challenger_rate = challenger_summary['wins'] / TEST_GAMES
        incumbent_rate = incumbent_summary['wins'] / TEST_GAMES
        promotion = (challenger_rate >= .60 and
                     challenger_summary['wins'] > incumbent_summary['wins'] and
                     challenger_summary['unfinished'] == 0 and incumbent_summary['unfinished'] == 0)
        report = {
            'parent_version': 402, 'selected_version': chosen.version, 'selected_mix': chosen.mix,
            'training': train_summary, 'validation': chosen_record,
            'paired_final': {
                'games_per_policy': TEST_GAMES, 'same_boards_and_seats': True,
                'incumbent_402': {'wins': incumbent_summary['wins'], 'win_rate': incumbent_rate,
                                  'CI95': wilson_interval(incumbent_summary['wins'], TEST_GAMES),
                                  'unfinished': incumbent_summary['unfinished']},
                'challenger': {'wins': challenger_summary['wins'], 'win_rate': challenger_rate,
                               'CI95': wilson_interval(challenger_summary['wins'], TEST_GAMES),
                               'unfinished': challenger_summary['unfinished']},
                'paired_outcomes': paired,
                'discordant_exact_binomial_p': (float(binomtest(paired['challenger_only_wins'],
                    paired['challenger_only_wins'] + paired['incumbent_only_wins'], .5).pvalue)
                    if paired['challenger_only_wins'] + paired['incumbent_only_wins'] else 1.),
            },
            'target_60pct_reached': challenger_rate >= .60 and challenger_summary['unfinished'] == 0,
            'promote_to_best_pilot': promotion,
            'leakage_controls': {'fresh_games_after_all_prior_maps': True,
                                 'validation_and_test_boards_disjoint': True,
                                 'test_used_for_selection': False},
            'elapsed_seconds': perf_counter() - start,
        }
        write(ROOT / 'report.json', report)
        mlflow.log_metrics({'fresh_training_wins': train_summary['wins'],
                            'validation_wins': chosen_record['wins'],
                            'incumbent_paired_test_win_rate': incumbent_rate,
                            'challenger_paired_test_win_rate': challenger_rate,
                            'target_60pct_reached': float(report['target_60pct_reached']),
                            'promoted': float(promotion)})
        for filename in ('report.json', 'validation.json', 'mc_training_sources.json'):
            mlflow.log_artifact(str(ROOT / filename), 'results')

        if promotion:
            chosen.save(ROOT / 'promoted_model')
            sample = new_x[:512].astype(np.float64)
            expected = chosen.predict(sample)
            info = mlflow.pyfunc.log_model(
                name='model', python_model=TreePolicyModel(),
                artifacts={'ranker': str(ROOT / 'promoted_model/ranker.txt'),
                           'mc': str(ROOT / 'promoted_model/monte_carlo.joblib'),
                           'policy': str(ROOT / 'promoted_model/policy.json')},
                input_example=sample[:3], signature=infer_signature(sample, expected),
                pip_requirements=['mlflow==' + mlflow.__version__, 'numpy==' + np.__version__,
                                  'lightgbm==' + lgb.__version__, 'scikit-learn==1.7.1',
                                  'joblib==' + joblib.__version__],
                code_paths=[str(REPO / 'tools/optimize_tree_policy_vs_f.py'),
                            str(REPO / 'tools/train_residual_tree_fqi.py'),
                            str(REPO / 'tools/train_online_sarsa.py'),
                            str(REPO / 'tools/train_three_game_sarsa.py'), str(REPO / 'src/catanzero')])
            restored = mlflow.pyfunc.load_model(info.model_uri)
            error = float(np.max(np.abs(restored.predict(sample) - expected)))
            assert error < 1e-12
            registered = mlflow.register_model(info.model_uri, 'CatanZero_TreePolicy_NativeF')
            client.set_registered_model_alias('CatanZero_TreePolicy_NativeF', 'best_pilot', str(registered.version))
            report['registration'] = {'version': str(registered.version), 'model_uri': info.model_uri,
                                      'alias': 'best_pilot', 'reload_max_abs_error': error}
            write(ROOT / 'report.json', report)
            mlflow.log_artifact(str(ROOT / 'report.json'), 'results')
        print('RESULT: ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
