"""Serve a replay on loopback so it opens as a normal browser page."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import webbrowser

from catanzero.viewer import export_replay


def make_server(replay: Path, port: int = 8765) -> ThreadingHTTPServer:
    content = replay.read_bytes()

    class ReplayHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path not in ("/", "/index.html"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

    return ThreadingHTTPServer(("127.0.0.1", port), ReplayHandler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Open a saved Catan game in your browser")
    parser.add_argument("game", nargs="?", type=Path, help="saved game JSON; default: newest game under runs/")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="serve without launching the browser")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    game = args.game
    if game is None:
        games = list(Path("runs").glob("*/game_*.json"))
        if not games:
            parser.error("No saved game found. Run run.cmd --seed 42 --save-games first.")
        game = max(games, key=lambda path: path.stat().st_mtime_ns)
    try:
        replay = export_replay(game)
        server = make_server(replay, args.port)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    with server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Game: {game.resolve()}", flush=True)
        print(f"Watch: {url}", flush=True)
        print("Keep this process running. Press Ctrl+C to stop.", flush=True)
        if not args.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
