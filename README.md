# CatanZero

A playable Catan browser game and a classical machine-learning bot built on
[Catanatron](https://github.com/bcollazo/catanatron).

**Model 402 achieved a 57% win rate against one native F bot: 570 wins in
1,000 new two-player games.** It selects legal moves without forward search
or neural networks.

**[Play CatanZero online](https://amitnahum18.github.io/CatanZero-/)**

Choose **1, 2 or 3 bots**. Every bot uses the same frozen Model 402; there is
no difficulty selector. The game runs entirely in your browser, including the
rules engine and model inference. No account or running backend is required.

## Browser game

- Responsive board with roads, settlements, cities, ports and the robber.
- Clear pip dice, a separate total and a special indicator for 7.
- Visible resources, building prices, development cards and an action log.
- Legal bank and port trades at 4:1, 3:1 and 2:1 rates.
- Select a legal building location, preview it, then confirm.
- Automatic browser saves and saved-game restoration.
- Engine-enforced placement distance, costs, card timing and victory rules.

The first visit downloads about 16 MB of game/runtime files before HTTP
compression. A Web Worker keeps game execution off the UI thread. Saves stay
on the current device; clearing site data removes them. Direct player-to-player
trading is not implemented. Two-player matches use the simulator's variant on
the standard board.

The UI includes responsive phone layouts; physical-phone testing has not yet
been completed. A modern browser supporting WebAssembly, module workers and
IndexedDB is required.

## Model 402 and results

| Component | Trees | Maximum depth | Role |
|---|---:|---:|---|
| LightGBM ranking model | 114 | 10 | Rank legal candidate actions |
| ExtraTrees regressor | 64 | 10 | Add a correction learned from game returns |

Each candidate receives **264 numeric predecision features**: the acting
player's hand, public opponents, board position, resources, construction
capacity and the candidate action. The browser shares the original feature
encoder and uses a numeric export of the frozen trees.

```text
action_score = LightGBM(raw_features) / 1.4757017551196938
             + 0.2 * ExtraTrees(raw_features)
```

The bot selects the highest-scoring legal action. Scores are preferences,
not calibrated win probabilities. No forward simulation or weight update occurs
during inference. The historical normalizer records the feature schema;
402 uses raw features, matching the original selected policy.

| Evaluation | Wins | Win rate |
|---|---:|---:|
| Latest independent confirmation against **one native F bot**, two players | **570 / 1,000** | **57.0%** |
| Earlier test against one native F bot, two players | 282 / 500 | 56.4% |
| Separate test against **three native F bots**, four players | 158 / 500 | 31.6% |

The latest test used new boards, balanced seats and frozen weights. Its exact
95% confidence interval is **53.8647%-60.0940%**. The measured result is 57%;
the test does not establish a true win rate above 55% at that confidence level.
Player counts and evaluation samples are reported separately. Native F and
F_static are different opponents.

See the [confirmation report](docs/402-confirmation-f-1000-20261015.md),
[model files](models/champion402/README.md) and
[project history](docs/PROJECT_STATUS.md) for the experiments and limitations.

## Run locally

Install Git, Python 3.12 and [uv](https://docs.astral.sh/uv/), then run:

```sh
git clone https://github.com/amitnahum18/CatanZero-.git
cd CatanZero-
uv sync --frozen
uv run --with-requirements requirements-play.txt python -m catanzero.play_server --open
```

Open **http://127.0.0.1:8787/**. On Windows, `play.cmd` is also available after
setup. Leave the server running while playing; Ctrl+C stops it. The server
binds to localhost and stores saves under `runs/human-games/`. These saves are
separate from the public website's browser saves.

## Build and deploy GitHub Pages

The public site needs no separate server. Pyodide runs the shared Python core
and portable model in the browser. With Node.js 22 and the Python environment:

```sh
npm ci --ignore-scripts
uv run python tools/build_pages.py
npm run test:wasm
node tests/check_pages_client.mjs
python -m http.server 8080 --directory site
```

Open **http://localhost:8080/** to preview. The generated `site/` directory is
ignored by Git. Pushes to `main` trigger the [Pages workflow](.github/workflows/pages.yml),
which builds, tests and deploys automatically. Pages uses GitHub Actions.
See [deployment details](docs/github-pages.md) for browser storage and checks.

## Simulations and replays

Generate reproducible games, save engine states and export HTML replays:

```sh
uv run python -m catanzero --seed 42 --save-games --replay
uv run python -m catanzero --games 100 --seed 42
uv run python -m catanzero --players 2 --seed 42 --trace
uv run python -m catanzero.viewer runs/my-run/game_0000.json
```

On Windows, `run.cmd` starts simulations and `watch.cmd` opens saved replays.
The simulation CLI defaults to four random players, separately from the human
game whose opponents all use Model 402.

Outputs go into a new `runs/` directory: configuration, per-game results and
a summary, plus full states and HTML replays when requested. Full states include
hidden cards and random-generator state for research; they are not uploaded.
Games stopped at a safety limit have no winner and are not counted as wins.

The engine is pinned to Catanatron commit
`ecf931181b9a65bb4116a2153fb78c16f1438e00`. Reproducibility requires matching
engine versions, settings and Python environments.

## Verification

- **12 engine/HTTP checks** passed: legality, hidden-information filtering,
  full-game completion, save restoration and bank/port trades.
- **29 dataset/feature checks** passed after extracting the shared core.
- Portable inference matched the native libraries on **1,573 candidates**,
  with identical selected actions and maximum score error below `2.67e-15`.
- Real WebAssembly execution checked **225 actions** across 2/3/4-player games,
  stale-action rejection and save/restore consistency.
- The actual worker and an IndexedDB test implementation checked reload,
  continued play, retained saves and storage-denial handling.
- Application-code integration exercised **120 live HTTP actions**.

These checks do not constitute rendered browser or physical-device visual QA.
Run the native game tests with:

```sh
uv run --with-requirements requirements-play.txt pytest tests/test_play_server.py -q
```

## Project layout

| Path | Contents |
|---|---|
| `src/catanzero/play_web/` | Shared UI and worker adapter |
| `src/catanzero/play_core.py` | Shared game session and public player view |
| `src/catanzero/browser_game.py` | Browser-local requests and persistence |
| `src/catanzero/play_server.py` | Optional localhost server |
| `models/champion402/` | Selected model and portable export |
| `tools/` | Build, training, evaluation and audit utilities |
| `tests/` | Engine, feature, policy and application checks |
| `docs/` | Experiment history and operating instructions |
| `licenses/` | Third-party license texts |

Generated experiments, datasets, MLflow data, personal saves, virtual
environments, dependencies, logs and caches are excluded from Git. Historical
local data remains under `runs/`; some research documents reference local-only
artifacts.

Research documentation covers [datasets](docs/dataset-pilot.md),
[weighted labels](docs/weighted-scoring.md), [search labels](docs/search-labels.md),
[linear regression](docs/linear-regression.md),
[model comparisons](docs/model-comparison.md),
[online SARSA](docs/online-sarsa-2026-10-14.md),
[no-lookahead evaluation](docs/no-lookahead-experiment-2026-10-14.md) and
[402 training and evaluation](docs/tree-policy-depth10-native-f.md).
Some historical documents are in Hebrew.

## License and attribution

This is an unofficial fan project, not affiliated with CATAN GmbH,
Catan Universe or Colonist. Board symbols are drawn in SVG; third-party game
screenshots are not published as game assets.

Original CatanZero code retains the MIT license in [LICENSE](LICENSE).
The combined game distribution includes Catanatron under GPL-3.0-or-later;
see [COPYING](COPYING) and [third-party notices](THIRD_PARTY_NOTICES.md).
The deployed website provides a source archive and license notices.
