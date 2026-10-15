"""Build a static, self-contained GitHub Pages game (no API server required)."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/'site'
WEB = ROOT/'src/catanzero/play_web'


def package_files(name):
    spec = importlib.util.find_spec(name)
    package = Path(spec.origin).parent
    for path in sorted(package.rglob('*.py')):
        relative = path.relative_to(package)
        if any(part in ('tests', 'cli', 'gym', '__pycache__') for part in relative.parts):
            continue
        yield path, f'{name}/{relative.as_posix()}'


def main():
    SITE.mkdir(exist_ok=True)
    digest = hashlib.sha256()
    for path in sorted((ROOT/'src/catanzero').rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.js', '.mjs', '.html', '.css'):
            digest.update(path.read_bytes())
    digest.update((ROOT/'models/champion402/portable.json').read_bytes())
    version = digest.hexdigest()[:12]
    for name in ('app.js', 'style.css'):
        shutil.copy2(WEB/name, SITE/name)
    html = (WEB/'index.html').read_text(encoding='utf-8')
    html = html.replace('href="/style.css?', 'href="./style.css?').replace('src="/app.js?', 'src="./app.js?')
    html = re.sub(r'(\./(?:app\.js|style\.css))\?v=[^"]+', rf'\1?v={version}', html)
    html = html.replace('<script src="./app.js?', f'<script src="./browser-client.js?v={version}" defer></script>\n  <script src="./app.js?')
    html = html.replace('Saved automatically on this computer.', 'Saved automatically in this browser.')
    html = html.replace('class="brand" href="/"', 'class="brand" href="./"')
    html = html.replace('<button id="start-game"', '<p id="runtime-status" class="note" role="status">Loading game engine…</p><button id="start-game"')
    html = html.replace('This version supports bank and port trades.', 'Unofficial fan project. This version supports bank and port trades.')
    html = html.replace('</dialog>\n<dialog id="new-dialog">', '<p class="note"><a href="./source.zip">Download source</a> · <a href="./THIRD_PARTY_NOTICES.md">Licenses</a></p></dialog>\n<dialog id="new-dialog">')
    (SITE/'index.html').write_text(html, encoding='utf-8')
    for name in ('browser-client.js', 'game-worker.mjs'):
        source = (WEB/name).read_text(encoding='utf-8')
        source = source.replace("'./game-worker.mjs'", f"'./game-worker.mjs?v={version}'")
        source = source.replace("'./game.zip'", f"'./game.zip?v={version}'")
        (SITE/name).write_text(source, encoding='utf-8')
    runtime = ROOT/'node_modules/pyodide'
    (SITE/'runtime').mkdir(exist_ok=True)
    for name in ('pyodide.mjs', 'pyodide.asm.mjs', 'pyodide.asm.wasm', 'python_stdlib.zip', 'pyodide-lock.json'):
        shutil.copy2(runtime/name, SITE/'runtime'/name)
    modules = ('__init__', 'simulation', 'observation', 'play_core', 'features',
               'strategic_features', 'build_capacity', 'champion_portable', 'browser_game')
    files = [*package_files('catanatron'), *package_files('networkx')]
    files += [(ROOT/f'src/catanzero/{name}.py', f'catanzero/{name}.py') for name in modules]
    files += [(ROOT/'models/champion402/portable.json', 'champion402.json')]
    with zipfile.ZipFile(SITE/'game.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, target in files:
            archive.writestr(target, path.read_bytes())
    shutil.copy2(ROOT/'THIRD_PARTY_NOTICES.md', SITE/'THIRD_PARTY_NOTICES.md')
    shutil.copytree(ROOT/'licenses', SITE/'licenses', dirs_exist_ok=True)
    shutil.copy2(ROOT/'COPYING', SITE/'COPYING')
    shutil.copy2(ROOT/'LICENSE', SITE/'LICENSE')
    # Corresponding application source is available from the same host.
    source_files = []
    for directory in ('src', 'tools', 'tests', 'models', 'docs', 'examples', 'licenses', '.github'):
        source_files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file()
                            and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo'))
    source_files.extend(ROOT/name for name in ('README.md', 'LICENSE', 'COPYING', 'THIRD_PARTY_NOTICES.md',
                        'pyproject.toml', 'uv.lock', 'package.json', 'package-lock.json', 'requirements-play.txt'))
    with zipfile.ZipFile(SITE/'source.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sorted(source_files):
            archive.writestr('CatanZero/' + source.relative_to(ROOT).as_posix(), source.read_bytes())
        for source, target in files:
            if target.startswith(('catanatron/', 'networkx/')):
                archive.writestr('third_party/' + target, source.read_bytes())
    (SITE/'.nojekyll').touch()
    manifest = {p.relative_to(SITE).as_posix(): {'bytes': p.stat().st_size,
                 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(SITE.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    (SITE/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'site': str(SITE), 'files': len(manifest),
                      'total_bytes': sum(p['bytes'] for p in manifest.values())}))


if __name__ == '__main__':
    main()
