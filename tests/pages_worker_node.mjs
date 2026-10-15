/* Node harness for the actual static worker module, not a browser driver. */
import { parentPort } from 'node:worker_threads';
import fs from 'node:fs/promises';

globalThis.self = { postMessage: data => parentPort.postMessage(data) };
const nativeFetch = globalThis.fetch;
globalThis.fetch = async (url, options) => {
  if (String(url).startsWith('file:')) return new Response(await fs.readFile(new URL(url)));
  return nativeFetch(url, options);
};
await import('../site/game-worker.mjs');
parentPort.on('message', data => self.onmessage({ data }));
