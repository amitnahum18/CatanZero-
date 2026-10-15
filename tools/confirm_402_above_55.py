"""One predeclared, fixed-size confirmation test of frozen 402 versus one F."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
from continue_champion_tree_rl import SOURCE, REPO, load_policy
from evaluation_confidence import win_rate_acceptance
from train_residual_tree_fqi import play_phase, write
from train_three_game_sarsa import exclusions


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=84_000_000)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('positive workers required')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    games = 1000
    excluded, _ = exclusions(REPO / 'runs/online-sarsa-features-100-v4')
    source_files = {n: SOURCE / 'best_model' / n
                    for n in ('ranker.model', 'monte_carlo.joblib', 'policy.json')}
    source_files['normalizer.json'] = SOURCE / 'normalizer.json'
    hashes = {n: sha(p) for n, p in source_files.items()}
    snapshot = root / 'frozen_model'
    snapshot.mkdir()
    for name, path in source_files.items():
        shutil.copyfile(path, snapshot / name)
    norm = json.loads((snapshot / 'normalizer.json').read_text(encoding='utf-8'))
    model = load_policy(snapshot, norm, 402)
    model.policy_name = 'Frozen_402_Confirmation'
    feature_probe = np.zeros((1, len(norm['feature_names'])))
    before = model.predict(feature_probe)
    minimum = next(w for w in range(games + 1)
                   if win_rate_acceptance(w, games)['passed'])
    config = {
        'declared_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'confirm frozen 402 win rate exceeds 55%, not train or select candidates',
        'games': games, 'players': 2, 'opponent': 'F', 'base_seed': args.seed,
        'workers': args.workers, 'weights_updated': False, 'exploration': 0.,
        'learner_forward_search': False, 'max_seconds_per_game': 600,
        'seats': 'alternating; 500 per seat', 'target': .55, 'confidence': .95,
        'acceptance_method': 'exact two-sided Clopper-Pearson lower bound > .55',
        'minimum_wins': minimum, 'sample_size_fixed_before_play': True,
        'optional_stopping': False, 'rerun_until_pass': False,
        'unfinished_games': 'no replacement or removal; report scheduled denominator, no certification if unfinished',
        'excluded_prior_board_count': len(excluded), 'source': str(SOURCE),
        'source_sha256': hashes,
        'scope': 'this fixed simulator, opponent and balanced-seat test protocol only',
    }
    write(root / 'protocol_before_play.json', config)
    write(root / 'excluded_boards.json', excluded)
    files = ('tools/confirm_402_above_55.py', 'tools/evaluation_confidence.py',
             'tools/train_residual_tree_fqi.py', 'tools/train_online_sarsa.py',
             'tools/continue_champion_tree_rl.py', 'tools/optimize_dagger_policy.py',
             'src/catanzero/online_env.py', 'src/catanzero/live_belief.py')
    code_hashes = {}
    for name in files:
        src = REPO / name
        if not src.exists():
            continue
        dst = root / 'implementation_snapshot' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        code_hashes[name] = sha(src)
    write(root / 'implementation_sha256.json', code_hashes)
    print(f'PROTOCOL LOCKED: {games} games, minimum {minimum} wins, no early stopping', flush=True)
    started = perf_counter()
    rows, summary, _ = play_phase(root, 'evaluation', model, games, args.workers,
        training=False, base_seed=args.seed, excluded=excluded,
        opponent_policy='F', max_seconds=600)
    assert len(rows) == games
    assert [r['game_index'] for r in rows] == list(range(games))
    assert all(r['seed'] == args.seed + r['game_index'] and r['players'] == 2
               and r['opponent_policy'] == 'F' for r in rows)
    seat_games = {str(s): sum(r['learner_seat'] == s for r in rows) for s in (0, 1)}
    assert seat_games == {'0': 500, '1': 500}
    assert not set(r['initial_map_hash'] for r in rows).intersection(excluded)
    assert len({r['initial_map_hash'] for r in rows}) == games
    assert all(sha(path) == hashes[name] for name, path in source_files.items())
    assert all(sha(snapshot / name) == value for name, value in hashes.items())
    assert all(sha(REPO / name) == value for name, value in code_hashes.items())
    np.testing.assert_array_equal(before, model.predict(feature_probe))
    assert model.version == 402 and not summary['training'] and summary['transitions'] == 0
    acceptance = win_rate_acceptance(summary['wins'], games)
    certified = acceptance['passed'] and summary['unfinished'] == 0
    report = {
        'summary': summary, 'acceptance_all_scheduled': acceptance,
        'requirement_met': certified, 'certification_requires_all_games_complete': True,
        'seat_games': seat_games,
        'seat_wins': {str(s): sum(r['learner_seat'] == s and r['learner_won'] is True
                                  for r in rows) for s in (0, 1)},
        'termination_counts': {t: sum(r['termination'] == t for r in rows)
                               for t in sorted({r['termination'] for r in rows})},
        'audits': {'model_and_code_hashes_unchanged': True, 'training_transitions': 0,
                   'unique_fresh_maps': games, 'seats_balanced': True},
        'elapsed_seconds': perf_counter() - started,
        'previous_539_of_1000_combined_into_this_test': False,
        'scope': config['scope'],
    }
    write(root / 'report.json', report)
    print('FINAL: ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
