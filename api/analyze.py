"""Vercel Python serverless function: GET /api/analyze?sport=football

Returns a JSON analysis. If API-Football credentials are configured as
environment variables it ingests **real** football fixtures; otherwise it
returns a synthetic demo (always labelled). Safe to host publicly either way.

Environment variables (set in Vercel → Project → Settings → Environment
Variables) to enable live data:
    APIFOOTBALL_KEY      your API-Football key
    APIFOOTBALL_LEAGUE   numeric league id (e.g. 39 = Premier League)
    APIFOOTBALL_SEASON   season year (e.g. 2023)

Vercel auto-detects ``api/*.py`` (zero-config) and serves the ``handler`` class.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Make the repo root importable so ``bet_assistant`` resolves when the function
# runs from the api/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bet_assistant.demo import run_analysis, supported_sports  # noqa: E402


def _int_env(name: str):
    raw = os.environ.get(name)
    try:
        return int(raw) if raw not in (None, "") else None
    except ValueError:
        return None


class handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler naming)
        params = parse_qs(urlparse(self.path).query)
        sport = (params.get("sport", ["basketball"])[0] or "basketball").lower()

        try:
            body = run_analysis(
                sport,
                api_key=os.environ.get("APIFOOTBALL_KEY"),
                league=_int_env("APIFOOTBALL_LEAGUE"),
                season=_int_env("APIFOOTBALL_SEASON"),
            )
            status = 200
        except ValueError as exc:
            body = {"error": str(exc), "supported_sports": list(supported_sports())}
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
