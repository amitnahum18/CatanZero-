from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from catanzero.watch import make_server


def test_serves_only_replay_as_html(tmp_path):
    replay = tmp_path / "replay.html"
    content = '<!doctype html><html lang="he"><body>קאטן</body></html>'.encode("utf-8")
    replay.write_bytes(content)
    server = make_server(replay, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url, timeout=3) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "text/html; charset=utf-8"
            assert response.read() == content
        with pytest.raises(HTTPError) as error:
            urlopen(url + "/pyproject.toml", timeout=3)
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
