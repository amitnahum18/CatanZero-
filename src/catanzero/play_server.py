"""Local HTTP adapter for the CatanZero game session."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import re
import threading
from urllib.parse import urlparse
import webbrowser
from catanzero.play_core import ROOT, CATALOG, Table, Conflict, board_geometry, load_models
from catanzero.simulation import create_game, SimulationConfig

ASSETS = Path(__file__).with_name('play_web')

class PlayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, save_directory):
        super().__init__(address, Handler)
        self.save_directory = save_directory
        self.tables = {}
        # The engine simulation guard patches class methods: serialize all bot inference.
        self.table_lock = threading.RLock()

    def get_table(self, ident):
        if not re.fullmatch(r'[a-f0-9]{32}', ident):
            raise KeyError('unknown game')
        if ident not in self.tables:
            path = self.save_directory / f'{ident}.json'
            if not path.is_file():
                raise KeyError('unknown game')
            self.tables[ident] = Table.restore(path)
        return self.tables[ident]


class Handler(BaseHTTPRequestHandler):
    def send(self, status, data, content_type='application/json; charset=utf-8'):
        body = json.dumps(data, ensure_ascii=False).encode() if not isinstance(data, bytes) else data
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.end_headers()
        self.wfile.write(body)

    def allowed(self):
        host = self.headers.get('Host', '')
        if host not in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'):
            self.send(403, {'error': 'Open the game using its local address.'})
            return False
        origin = self.headers.get('Origin')
        if origin and origin != f'http://{host}':
            self.send(403, {'error': 'This request origin is not allowed.'})
            return False
        return True

    def do_GET(self):
        if not self.allowed(): return
        path = urlparse(self.path).path
        assets = {'/': ('index.html','text/html; charset=utf-8'),
                  '/app.js': ('app.js','text/javascript; charset=utf-8'),
                  '/style.css': ('style.css','text/css; charset=utf-8')}
        try:
            if path in assets:
                name, mime = assets[path]
                self.send(200, (ASSETS/name).read_bytes(), mime)
            elif path == '/api/bootstrap':
                with self.server.table_lock:
                    geometry = board_geometry(create_game(SimulationConfig(), 31415))
                self.send(200, {'models': CATALOG, 'preview': geometry})
            elif path.startswith('/api/game/'):
                with self.server.table_lock:
                    self.send(200, self.server.get_table(path.removeprefix('/api/game/')).public_view())
            else:
                self.send(404, {'error': 'Page not found.'})
        except KeyError:
            self.send(404, {'error': 'Game not found. You can start a new game.'})

    def do_POST(self):
        if not self.allowed(): return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                self.send(400, {'error': 'A valid JSON request is required.'}); return
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict): raise ValueError('Invalid request.')
            with self.server.table_lock:
                if self.path == '/api/new':
                    table = Table(payload.get('bots'), payload.get('name', 'You'))
                    self.server.tables[table.id] = table
                elif self.path in ('/api/action', '/api/advance'):
                    table = self.server.get_table(str(payload.get('id', '')))
                    table.act(payload.get('revision'), payload.get('index'), advance=self.path == '/api/advance')
                else:
                    self.send(404, {'error': 'Action not found.'}); return
                table.save(self.server.save_directory)
                self.send(200, table.public_view())
        except Conflict as exc:
            self.send(409, {'error': str(exc)})
        except (ValueError, TypeError) as exc:
            self.send(400, {'error': str(exc)})
        except KeyError:
            self.send(404, {'error': 'Game not found.'})
        except Exception:
            logging.exception('Game request failed')
            self.send(500, {'error': 'A game error occurred. Reload the page to resume your saved game.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8787)
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    print('Loading frozen model 402...', flush=True)
    load_models()
    server = PlayServer(('127.0.0.1', args.port), ROOT/'runs/human-games')
    url = f'http://127.0.0.1:{server.server_port}/'
    print(f'CatanZero Play is ready: {url}', flush=True)
    if args.open: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__':
    main()
