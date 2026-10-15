import { loadPyodide } from './runtime/pyodide.mjs';

const sendProgress = progress => self.postMessage({ progress });
const initialized = (async () => {
  sendProgress('Loading game engine…');
  const pyodide = await loadPyodide();
  sendProgress('Loading board rules and Model 402…');
  const response = await fetch(new URL('./game.zip', import.meta.url));
  if (!response.ok) throw new Error('Game download failed.');
  pyodide.unpackArchive(new Uint8Array(await response.arrayBuffer()), 'zip', { extractDir: '/game' });
  await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, '/game')
from catanzero.browser_game import BrowserGame
web_game = BrowserGame('/game/champion402.json')
`);
  self.postMessage({ ready: true });
  return pyodide;
})();
initialized.catch(error => { console.error(error); self.postMessage({ fatal: true }); });
let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try {
      const pyodide = await initialized;
      pyodide.globals.set('request_path', data.path);
      pyodide.globals.set('request_payload', JSON.stringify(data.payload));
      pyodide.globals.set('request_saved', JSON.stringify(data.saved));
      const response = JSON.parse(pyodide.runPython('web_game.request_json(request_path, request_payload, request_saved)'));
      self.postMessage({ id: data.id, ...response });
    } catch (error) {
      console.error(error);
      self.postMessage({ id: data.id, error: 'A game error occurred. Reload to resume your saved match.', status: 500 });
    }
  });
};
