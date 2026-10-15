/* Real WebAssembly Python execution under Node. This is not browser pixel QA. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { loadPyodide } from '../node_modules/pyodide/pyodide.mjs';

async function main() {
  const started = performance.now();
  const py = await loadPyodide({ indexURL: path.resolve('site/runtime') + path.sep });
  py.unpackArchive(new Uint8Array(fs.readFileSync('site/game.zip')), 'zip', { extractDir: '/game' });
  await py.runPythonAsync(`
import sys, json, time
sys.path.insert(0, '/game')
from catanzero.browser_game import BrowserGame, BrowserTable
from catanzero.observation import ordered_actions
from catanzero.play_core import Conflict
web_game = BrowserGame('/game/champion402.json')
`);
  console.log('WebAssembly game engine and 402 loaded.');
  const report = JSON.parse(py.runPython(`
report = {'players_checked': [], 'bot_decisions': 0, 'actions': 0}
for count in (1, 2, 3):
    view, _ = web_game.request('/api/new', {'bots': ['402'] * count, 'name': 'WASM test'})
    assert len(view['players']) == count + 1
    assert all(p['model'] == '402' for p in view['players'] if not p['human'])
    opponent = next(p for p in view['players'] if not p['human'])
    assert 'resources' not in opponent and 'actual_vp' not in opponent
    for step in range(75):
        table = web_game.table
        if table.game.winning_color(): break
        before = table.revision
        before_rng = table.game.state.random.getstate()
        if table.game.state.current_color() == table.human:
            # Use 402 as a test driver on the human seat, without changing bots.
            action = table.choose_bot()
            assert table.game.state.random.getstate() == before_rng
            index = ordered_actions(table.game).index(action)
            view, saved = web_game.request('/api/action', {'id': table.id, 'revision': before, 'index': index})
        else:
            view, saved = web_game.request('/api/advance', {'id': table.id, 'revision': before})
            report['bot_decisions'] += 1
        assert view['revision'] == before + 1
        report['actions'] += 1
        if step == 15:
            restored = BrowserTable.from_dump(json.loads(json.dumps(saved)), web_game.policy)
            assert restored.public_view() == table.public_view()
            assert restored.choose_bot() == table.choose_bot()
        try:
            web_game.request('/api/advance', {'id': table.id, 'revision': before})
            raise AssertionError('duplicate revision accepted')
        except Conflict:
            pass
    report['players_checked'].append(count + 1)
json.dumps(report)
`));
  assert.deepEqual(report.players_checked, [2, 3, 4]);
  assert(report.actions >= 200);
  // Native-generated candidate scores validate actual WASM float semantics.
  if (fs.existsSync('tests/fixtures/champion402-parity.json')) {
    py.FS.writeFile('/parity.json', fs.readFileSync('tests/fixtures/champion402-parity.json'));
    report.native_parity = JSON.parse(py.runPython(`
fixtures = json.load(open('/parity.json'))
worst = 0.0
for group in fixtures:
    scores = [web_game.policy.predict(row) for row in group['rows']]
    worst = max(worst, max(abs(a-b) for a,b in zip(scores, group['scores'])))
    assert max(range(len(scores)), key=scores.__getitem__) == group['choice']
assert worst < 1e-10, worst
json.dumps({'candidate_groups': len(fixtures), 'max_absolute_error': worst})
`));
  }
  console.log(JSON.stringify({ status: 'passed', runtime: 'Pyodide 314.0.7 / real WebAssembly', ...report,
    seconds: (performance.now() - started) / 1000, visual_browser_qa: false }));
}
main().catch(error => { console.error(String(error)); process.exitCode = 1; });
