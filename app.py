"""WSGI web-app entrypoint for Vercel.

Vercel's Python builder serves a WSGI/ASGI application exposed as ``app``. This
single app serves both the static UI (``public/index.html``) and the JSON
endpoint ``/api/analyze`` — so the whole thing deploys as one Python web app
rather than a bare serverless function.

Pure standard library (PEP 3333 WSGI) — no Flask/Django dependency.
"""

from __future__ import annotations

import json
import mimetypes
import os
from urllib.parse import parse_qs

from bet_assistant.demo import run_analysis, supported_sports

ROOT = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.path.join(ROOT, "public")


def _int_env(name: str):
    raw = os.environ.get(name)
    try:
        return int(raw) if raw not in (None, "") else None
    except ValueError:
        return None


def _json(start_response, body: dict, status: str = "200 OK"):
    payload = json.dumps(body).encode("utf-8")
    start_response(
        status,
        [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Cache-Control", "no-store"),
            ("Access-Control-Allow-Origin", "*"),
            ("Content-Length", str(len(payload))),
        ],
    )
    return [payload]


def _analyze(query_string: str) -> dict:
    qs = parse_qs(query_string)
    sport = (qs.get("sport", ["basketball"])[0] or "basketball").lower()
    return run_analysis(
        sport,
        api_key=os.environ.get("APIFOOTBALL_KEY"),
        league=_int_env("APIFOOTBALL_LEAGUE"),
        season=_int_env("APIFOOTBALL_SEASON"),
    )


def app(environ, start_response):
    """PEP 3333 WSGI application."""
    path = environ.get("PATH_INFO", "/") or "/"

    if path == "/api/analyze":
        try:
            return _json(start_response, _analyze(environ.get("QUERY_STRING", "")))
        except ValueError as exc:
            return _json(
                start_response,
                {"error": str(exc), "supported_sports": list(supported_sports())},
                "400 Bad Request",
            )
        except Exception as exc:  # never leak a stack trace
            return _json(
                start_response,
                {"error": f"internal error: {exc.__class__.__name__}"},
                "500 Internal Server Error",
            )

    # Static files (served from public/). Default to index.html.
    rel = "index.html" if path in ("/", "") else path.lstrip("/")
    fpath = os.path.normpath(os.path.join(PUBLIC, rel))
    if fpath.startswith(PUBLIC) and os.path.isfile(fpath):
        with open(fpath, "rb") as fh:
            data = fh.read()
        ctype = mimetypes.guess_type(fpath)[0] or "application/octet-stream"
        start_response(
            "200 OK",
            [("Content-Type", ctype), ("Content-Length", str(len(data)))],
        )
        return [data]

    # Fallback (e.g. static bundle missing) — the API still works.
    msg = b"Bet Assistant is running. Try /api/analyze?sport=football"
    start_response("200 OK", [("Content-Type", "text/plain"),
                              ("Content-Length", str(len(msg)))])
    return [msg]
