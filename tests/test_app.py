"""Tests for the WSGI web-app entrypoint (app.py) served by Vercel."""

import json

from wsgiref.util import setup_testing_defaults

import app as webapp


def _call(path, query=""):
    env = {}
    setup_testing_defaults(env)
    env["PATH_INFO"] = path
    env["QUERY_STRING"] = query
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = dict(headers)

    body = b"".join(webapp.app(env, start_response))
    return captured["status"], captured["headers"], body


def test_serves_index_html():
    status, headers, body = _call("/")
    assert status.startswith("200")
    assert headers["Content-Type"].startswith("text/html")
    assert b"<html" in body.lower()


def test_api_analyze_returns_json_payload():
    status, headers, body = _call("/api/analyze", "sport=football")
    assert status.startswith("200")
    assert headers["Content-Type"].startswith("application/json")
    data = json.loads(body)
    assert data["sport"] == "football"
    assert "recommendations" in data and "summary" in data


def test_api_analyze_unknown_sport_is_400():
    status, _, body = _call("/api/analyze", "sport=cricket")
    assert status.startswith("400")
    assert "error" in json.loads(body)


def test_unknown_path_falls_back_without_error():
    status, _, _ = _call("/does-not-exist")
    # Never a 500; static miss falls through to a friendly 200.
    assert status.startswith("200")
