"""Minimal local HTTP server for the overlay (stdlib only).

GET /          -> static/overlay.html
GET /live?after=N -> {"active", "session_id", "entries": live.since(N), "error"}

Binds to 127.0.0.1 only: captions are meeting content. In a bigger app mount the same
two routes into its own web server instead (see docs/AGENT.md).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

OVERLAY = Path(__file__).with_name("static") / "overlay.html"


def serve(get_live, host: str = "127.0.0.1", port: int = 8765, session_id: str = "live") -> ThreadingHTTPServer:
    """get_live() -> the current LiveTranscriber or None. Runs in a daemon thread."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.headers.get("Host", "").split(":")[0] not in ("127.0.0.1", "localhost"):
                self.send_error(403)  # DNS rebinding
                return
            url = urlparse(self.path)
            if url.path == "/":
                self._send(OVERLAY.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/live":
                live = get_live()
                try:
                    after = max(0, int(parse_qs(url.query).get("after", ["0"])[0]))
                except ValueError:
                    after = 0
                data = ({"active": True, "session_id": session_id, "entries": live.since(after),
                         "error": live.error} if live else {"active": False, "entries": []})
                self._send(json.dumps(data, ensure_ascii=False).encode(), "application/json")
            else:
                self.send_error(404)

    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="meetvoice-http").start()
    return httpd
