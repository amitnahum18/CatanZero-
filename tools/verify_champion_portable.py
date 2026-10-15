"""Compare exported inference with both original native libraries on real states."""
import json
from pathlib import Path
import numpy as np
from catanatron.serialization import action_to_json
from catanzero.play_core import Table, ROOT, load_models
from catanzero.observation import ordered_actions, observation
from catanzero.features import FeatureEncoder
from catanzero.champion_portable import Champion402


def main():
    portable = Champion402.load(ROOT/'models/champion402/portable.json')
    ranker, critic, policy, norm = load_models()
    groups, candidates, worst = [], 0, 0.
    for count in (1, 2, 3):
        table = Table(['402'] * count, seed=9200 + count)
        encoder = FeatureEncoder(table.map_doc, version=norm['feature_version'], allow_fewer_players=True)
        for _ in range(80):
            if table.game.winning_color(): break
            actor = table.game.state.current_color()
            actions = ordered_actions(table.game)
            inputs = {'map_ref': 0, 'observation': observation(table.game, actor),
                      'legal_actions': [action_to_json(a) for a in actions]}
            rows = [encoder.encode({**inputs, 'action': a}) for a in inputs['legal_actions']]
            raw = np.asarray(rows, dtype=np.float64)
            expected = ranker.predict(raw, num_threads=1) / policy['scale'] + policy['mix'] * critic.predict(raw)
            actual = [portable.predict(row) for row in rows]
            error = max(abs(a-b) for a,b in zip(actual, expected))
            worst = max(worst, error)
            assert error < 1e-10, error
            choice = int(np.argmax(expected))
            assert portable.choose(rows) == choice
            candidates += len(rows)
            if len(actions) > 1 and len(groups) < 40:
                groups.append({'rows': rows, 'scores': expected.tolist(), 'choice': choice})
            table.game.execute(actions[choice])
    target = ROOT/'runs/pages-parity.json'
    target.write_text(json.dumps(groups, separators=(',', ':')), encoding='utf-8')
    print(json.dumps({'status': 'passed', 'candidates': candidates, 'max_absolute_error': worst,
                      'fixture_groups': len(groups), 'trees': [len(portable.ranker), len(portable.critic)]}))


if __name__ == '__main__':
    main()
