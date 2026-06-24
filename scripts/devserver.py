"""Local dev server that mimics the Vercel routing (static + /api/analyze).

For local preview only — Vercel serves ``public/`` and ``api/*.py`` for you in
production. Run:

    python scripts/devserver.py
    # open http://localhost:8000

No third-party dependencies.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PUBLIC = os.path.join(ROOT, "public")

from bet_assistant.demo import run_demo, supported_sports  # noqa: E402

_CONTENT_TYPES = {".html": "text/html", ".js": "application/javascript",
                  ".css": "text/css", ".json": "application/json"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quieter console
        pass

    def _send(self, status, body, ctype):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/analyze":
            params = parse_qs(parsed.query)
            sport = (params.get("sport", ["basketball"])[0] or "basketball").lower()
            try:
                body = run_demo(sport)
                status = 200
            except ValueError as exc:
                body = {"error": str(exc), "supported_sports": list(supported_sports())}
                status = 400
            self._send(status, json.dumps(body), "application/json; charset=utf-8")
            return

        # Static files from public/
        rel = parsed.path.lstrip("/") or "index.html"
        path = os.path.normpath(os.path.join(PUBLIC, rel))
        if not path.startswith(PUBLIC) or not os.path.isfile(path):
            self._send(404, "Not found", "text/plain")
            return
        ext = os.path.splitext(path)[1]
        with open(path, "rb") as fh:
            self._send(200, fh.read(), _CONTENT_TYPES.get(ext, "application/octet-stream"))


def main():
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Bet Assistant dev server on http://localhost:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
