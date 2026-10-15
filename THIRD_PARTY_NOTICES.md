# Third-party components

CatanZero is an unofficial fan project, not affiliated with CATAN GmbH,
Catan Universe or Colonist. No artwork from those products is used in the game.
The board symbols in this repository are drawn in SVG by the application.

- **Catanatron** — rules engine, revision
  `ecf931181b9a65bb4116a2153fb78c16f1438e00`, GPL-3.0-or-later.
  Source: https://github.com/bcollazo/catanatron
  License: [licenses/catanatron.txt](licenses/catanatron.txt).
- **NetworkX** — graph algorithms, version 3.7, BSD 3-Clause license.
  Source: https://github.com/networkx/networkx
  License: [licenses/networkx.txt](licenses/networkx.txt).
- **Pyodide** — Python WebAssembly runtime, version 314.0.7, MPL 2.0.
  Unmodified runtime files are copied from the pinned npm package at build time.
  Corresponding source: https://github.com/pyodide/pyodide/tree/314.0.7
  License: [licenses/pyodide.txt](licenses/pyodide.txt).
- **CPython** — Python language runtime distributed with Pyodide, PSF license.
  Source: https://github.com/python/cpython/tree/3.14
  License: [licenses/python.txt](licenses/python.txt).

The repository's original CatanZero code retains the MIT notice in `LICENSE`.
The combined game distribution that includes Catanatron is distributed under
GPL-3.0-or-later; see `COPYING`. The website provides the application source and
build scripts in `source.zip`, and the Python modules in `game.zip` are source,
not bytecode. Separately licensed runtime components retain their notices.

Native training/inference tools also use LightGBM, scikit-learn, NumPy and
joblib. These libraries are installed through their package distributions;
their native code is not copied into the static website. The browser uses the
exported numeric tree structure of the project's own trained Model 402.
