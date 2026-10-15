/* Actual module worker + real WASM + IndexedDB implementation under Node.
 * Verifies persistence across worker reloads; no rendered browser is used. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { Worker } from 'node:worker_threads';
import { indexedDB } from 'fake-indexeddb';

const workers = [];
class GameWorker {
  constructor(url, options) {
    assert.equal(url.pathname, '/CatanZero-/game-worker.mjs');
    assert.equal(options.type, 'module');
    this.worker = new Worker(new URL('./pages_worker_node.mjs', import.meta.url));
    workers.push(this.worker);
    this.worker.on('message', data => this.onmessage?.({ data }));
    this.worker.on('error', error => { console.error(String(error)); this.onerror?.(error); });
  }
  postMessage(data) { this.worker.postMessage(data); }
}
function client() {
  const status = { textContent: '' }, start = { disabled: false };
  const context = vm.createContext({ document: { baseURI: 'https://example.test/CatanZero-/',
    getElementById: id => id === 'runtime-status' ? status : start },
    URL, indexedDB, Worker: GameWorker, console });
  vm.runInContext(fs.readFileSync('site/browser-client.js', 'utf8'), context);
  return { api: context.catanOfflineApi, context, status, start };
}
async function main() {
  const first = client();
  assert.equal(first.start.disabled, true);
  const preview = await first.api('/api/bootstrap');
  assert(preview.preview.tiles.length > 19);
  assert.equal(first.start.disabled, false);
  assert.equal(first.context.catanStorageKey, 'catanzero-pages:/CatanZero-/');
  let view = await first.api('/api/new', { bots: ['402'], name: 'Save test' });
  for (let i = 0; i < 14; i++) view = await first.api(view.human_turn ? '/api/action' : '/api/advance',
    { id: view.id, revision: view.revision, index: view.legal_actions[0]?.index });
  assert.equal(view.revision, 14);
  await assert.rejects(first.api('/api/advance', { id: view.id, revision: 0 }), error => error.status === 409);
  const second = client();
  const restored = await second.api(`/api/game/${view.id}`);
  assert.deepEqual(restored, view, 'reload restores complete board, dice, actions and private hand');
  const next = await second.api(restored.human_turn ? '/api/action' : '/api/advance',
    { id: restored.id, revision: restored.revision, index: restored.legal_actions[0]?.index });
  assert.equal(next.revision, 15);
  const newMatch = await second.api('/api/new', { bots: ['402', '402', '402'], name: 'Another' });
  assert.equal(newMatch.players.length, 4);
  assert.notEqual(newMatch.id, view.id);
  const previous = await second.api(`/api/game/${view.id}`);
  assert.equal(previous.revision, 15, 'new game preserves earlier saved match');
  await assert.rejects(second.api('/api/game/missing'), error => error.status === 404);
  // A browser that denies IndexedDB can still play and receives a visible warning.
  const blockedContext = vm.createContext({ document: { baseURI: 'https://example.test/CatanZero-/',
    getElementById: id => id === 'runtime-status' ? blockedStatus : { disabled: false } },
    URL, indexedDB: { open() { throw new Error('Storage denied'); } }, Worker: GameWorker, console });
  const blockedStatus = { textContent: '' };
  vm.runInContext(fs.readFileSync('site/browser-client.js', 'utf8'), blockedContext);
  const unsaved = await blockedContext.catanOfflineApi('/api/new', { bots: ['402'] });
  assert.equal(unsaved.players.length, 2);
  assert(blockedStatus.textContent.includes('Autosave is unavailable'));
  console.log(JSON.stringify({ status: 'passed', worker: 'actual game-worker.mjs',
    checks: ['initialization', '14 real actions', 'stale revision rejected', 'IndexedDB reload',
      'continue restored game', 'old save retained', 'missing save 404', 'storage denial fallback'],
    visual_browser_qa: false }));
}
try { await main(); } catch (error) { console.error(String(error)); process.exitCode = 1; }
finally { await Promise.all(workers.map(worker => worker.terminate())); }
