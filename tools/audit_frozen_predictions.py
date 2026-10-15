"""Sample stored frozen-model decisions, recompute features and direct Q scores."""
import argparse
import gzip
import json
from pathlib import Path

from catanatron.serialization import map_to_json
from catanzero.simulation import create_game, SimulationConfig
from catanzero.features import FeatureEncoder, VERSION
from catanzero.online_linear import LinearActionValues
from catanzero.regression_data import privacy_invariance_check
from catanzero.static_policies import forbid_forward_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    counts = {}
    for label in ("baseline", "expanded"):
        source = args.root / label
        scorer = LinearActionValues.load(source / "linear_model.json", source / "normalizer.json")
        norm = json.loads((source / "normalizer.json").read_text())
        decisions = candidates = privacy = 0
        phase = args.root / f"{label}_evaluation"
        results = sorted((json.loads(s) for s in (phase / "results.jsonl").read_text().splitlines()), key=lambda r: r['game_index'])
        for result in results[:4]:
            game = create_game(SimulationConfig(), result['seed'])
            encoder = FeatureEncoder(map_to_json(game.state.board.map), version=norm.get('feature_version', VERSION))
            taken = 0
            with gzip.open(phase / f"game_{result['game_index']:04d}" / "decisions.jsonl.gz", "rt", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row['policy'] != f'Linear_SARSA_{label}' or not row.get('raw_scores'):
                        continue
                    inputs = row['inputs']
                    with forbid_forward_simulation():
                        scores = [scorer.predict(encoder.encode({**inputs, 'action': a})) for a in inputs['legal_actions']]
                        assert len(scores) == len(row['raw_scores'])
                        assert max(abs(a-b) for a,b in zip(scores,row['raw_scores'])) < 1e-12
                        selected = inputs['legal_actions'].index(inputs['action'])
                        assert abs(scores[selected]-max(scores)) < 1e-12 and not row['explored']
                        if taken == 0:
                            assert privacy_invariance_check(encoder, row)
                            privacy += 1
                    decisions += 1
                    candidates += len(scores)
                    taken += 1
                    if taken >= 20:
                        break
        counts[label] = {'sampled_decisions': decisions, 'candidate_predictions_checked': candidates, 'privacy_checks': privacy}
    report = {'passed': True, 'models': counts, 'checks': ['direct frozen scores', 'greedy legal selection',
              'no exploration', 'feature privacy invariance', 'no engine simulation in feature/prediction computation'],
              'scope': 'sampled prediction audit; not a complete engine replay'}
    (args.root / 'prediction_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
