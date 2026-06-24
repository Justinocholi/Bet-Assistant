"""Vercel Python serverless function: GET /api/analyze?sport=football

Returns a JSON demo analysis produced by ``bet_assistant.demo.run_demo``. Runs
entirely offline against synthetic data — safe to host publicly with no API key.

Vercel detects ``api/*.py`` and serves the ``handler`` class as a function.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Make the repo root importable so ``bet_assistant`` resolves on Vercel, where
# the function runs from the api/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bet_assistant.demo import run_demo, supported_sports  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (Vercel/BaseHTTPRequestHandler naming)
        params = parse_qs(urlparse(self.path).query)
        sport = (params.get("sport", ["basketball"])[0] or "basketball").lower()

        try:
            body = run_demo(sport)
            status = 200
        except ValueError as exc:
            body = {
                "error": str(exc),
                "supported_sports": list(supported_sports()),
            }
            status = 400
        except Exception as exc:  # never leak a stack trace to the client
            body = {"error": f"internal error: {exc.__class__.__name__}"}
            status = 500

        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
