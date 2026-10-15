# GitHub Pages deployment

The public game runs entirely in the visitor's browser. GitHub Pages serves
static assets; no Python server, cloud API, account or paid inference service
is required. A module Web Worker runs the original Catanatron rules under
Pyodide and evaluates the numeric export of frozen Model 402.

## Build

Use Node 22 and Python 3.11 or later:

```sh
uv sync --frozen
npm ci --ignore-scripts
uv run python tools/build_pages.py
npm run test:wasm
```

`site/` is generated, ignored by Git and uploaded by the Pages workflow.
To preview it, run `python -m http.server 8080 --directory site` and open
`http://localhost:8080/`. The normal local Python server remains available via
`play.cmd`. Both versions share the game session, feature encoder and UI code.
For the native server on a fresh checkout, use Python 3.12, `uv sync`, then
`uv run --with-requirements requirements-play.txt python -m catanzero.play_server --open`.

In GitHub Settings → Pages, the publishing source is **GitHub Actions**.
Pushes to `main` build, test and deploy automatically. The project URL is
`https://amitnahum18.github.io/CatanZero-/`.

## Browser behavior

- Choose 1–3 bots; all use the original frozen Model 402.
- Initial download is about 16 MB before HTTP compression and includes the
  Python runtime. Later visits can use normal browser HTTP caching.
- A loading message appears while the engine starts; the UI remains responsive
  because game execution runs in a worker. A modern browser with WebAssembly,
  module workers and IndexedDB is required.
- Saves use IndexedDB on that device/browser. An opaque current-match ID uses a
  path-specific localStorage key. Starting a new game keeps old browser saves.
- Private browsing, clearing site data or storage restrictions may remove or
  prevent saves. A storage failure is shown without blocking gameplay.
- Local-server saves are separate and are not uploaded or migrated.
- All full state remains on the visitor's device. Opponent private cards are
  omitted from the UI view, as before, but a local user can inspect their own
  browser memory. This is a single-human game, not a secure multiplayer server.

## What is published

Source, tests, research documentation, licenses and the selected model are in
the repository. The website contains only the UI, required Python modules,
numeric model export and runtime files. No virtual environments, training
datasets, MLflow database, human saves, replay archives, logs, credentials or
third-party reference screenshots are published. Historical experiments remain
locally under `runs/`; some research docs reference these local-only artifacts.

## Verification

- All twelve existing engine/HTTP checks passed after extracting the shared core.
- Native export comparison: 1,573 candidates from 2/3/4-player states; same
  selected actions, maximum score error `2.67e-15`.
- Real WebAssembly execution under Node: 225 legal actions, all three player
  counts, bot actions, rejected stale actions and save/restore consistency.
- Native reference scores are checked again by the WebAssembly test.
- Rendered browser and physical-phone QA are not available in this environment.

References: [GitHub Pages static hosting](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages),
[Pyodide worker API](https://pyodide.org/en/stable/usage/webworker.html).
