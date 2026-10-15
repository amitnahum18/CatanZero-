# Repository layout

| Path | Purpose |
|---|---|
| `src/catanzero/play_web/` | Shared English, responsive game interface and static worker adapter |
| `src/catanzero/play_core.py` | Shared engine-backed session and public UI view |
| `src/catanzero/browser_game.py` | Browser-local request and save adapter |
| `src/catanzero/play_server.py` | Optional localhost HTTP server |
| `src/catanzero/observation.py` | Observation whitelist and canonical legal-action ordering |
| `models/champion402/` | Only the selected frozen model and its portable export |
| `tools/build_pages.py` | Static website build |
| `tests/` | Rule, data, policy and application checks |
| `docs/` | Project history, experiments and operating instructions |
| `tools/` | Training, evaluation, audits and build utilities |
| `licenses/` | Third-party license notices |
| `.github/workflows/pages.yml` | Automatic verified deployment |

Local generated directories are kept separate and ignored: `runs/` for
experiments, replays, saves and publication verification; `.venv/` and
`.ml-venv/` for Python environments; `.uv-cache/`, `.pytest_cache/` and
`__pycache__/` for caches; `node_modules/` for JS dependencies; `site/` for the
built website. These are not repository content.

No historical dataset, trained model or human save was deleted during the
publication cleanup. Old policy implementations remain useful for reproducible
benchmarks and are not offered as difficulty options in the game.
