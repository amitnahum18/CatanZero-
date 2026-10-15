/* Static-host adapter: all game requests execute in a local WebAssembly worker. */
'use strict';
(() => {
  const base = new URL('.', document.baseURI);
  globalThis.catanStorageKey = `catanzero-pages:${base.pathname}`;
  const status = document.getElementById('runtime-status');
  const start = document.getElementById('start-game');
  start.disabled = true;
  const pending = new Map();
  let worker, serial = 0, failed = false, storageWarning = false;
  const database = new Promise((resolve, reject) => {
    const request = indexedDB.open(`catanzero-games:${base.pathname}`, 1);
    request.onupgradeneeded = () => request.result.createObjectStore('games', { keyPath: 'id' });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  // A blocked storage API must not prevent playing; report the save limitation.
  database.catch(() => {});
  async function stored(id, value) {
    const db = await database;
    return new Promise((resolve, reject) => {
      const tx = db.transaction('games', value ? 'readwrite' : 'readonly');
      const request = value ? tx.objectStore('games').put(value) : tx.objectStore('games').get(id);
      let result;
      request.onsuccess = () => { result = request.result; };
      tx.oncomplete = () => resolve(result);
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error || new Error('Saving interrupted.'));
    });
  }
  function fail(error) {
    failed = true;
    status.textContent = 'The game could not load. Reload the page to retry.';
    start.disabled = true;
    for (const request of pending.values()) request.reject(error);
    pending.clear();
  }
  function connect() {
    if (worker) return;
    status.textContent = 'Loading game… The first visit downloads the game engine.';
    worker = new Worker(new URL('./game-worker.mjs', base), { type: 'module' });
    worker.onerror = () => fail(new Error('Could not start the game engine. Reload to retry.'));
    worker.onmessage = async ({ data }) => {
      if (data.progress) { status.textContent = data.progress; return; }
      if (data.ready) { status.textContent = 'Ready. Games run and save in this browser.'; start.disabled = false; return; }
      if (data.fatal) { fail(new Error('Could not load the game engine. Reload to retry.')); return; }
      const request = pending.get(data.id);
      if (!request) return;
      pending.delete(data.id);
      if (data.error) {
        const error = new Error(data.error); error.status = data.status || 500; request.reject(error); return;
      }
      if (data.saved) {
        try { await stored(data.saved.id, data.saved); }
        catch {
          storageWarning = true;
          status.textContent = 'Autosave is unavailable. Keep this page open to finish your game.';
        }
      }
      if (!storageWarning && data.saved) status.textContent = 'Saved in this browser. No account needed.';
      request.resolve(data.result);
    };
  }
  globalThis.catanOfflineApi = async (path, body) => {
    if (failed) throw new Error('Reload the page to restart the game engine.');
    connect();
    let saved = null;
    if (path.startsWith('/api/game/')) {
      try { saved = await stored(path.slice('/api/game/'.length)); } catch {}
    }
    return new Promise((resolve, reject) => {
      const id = ++serial;
      pending.set(id, { resolve, reject });
      worker.postMessage({ id, path, payload: body || {}, saved: saved || null });
    });
  };
})();
