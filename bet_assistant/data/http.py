"""Tiny stdlib HTTP-JSON transport.

Kept separate and injectable so providers can be unit-tested offline with a fake
transport, and so the whole project stays dependency-free (no ``requests``).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Optional

# A transport maps (url, headers) -> parsed JSON dict. Raises on failure.
Transport = Callable[[str, dict], dict]


class HttpError(RuntimeError):
    """Any transport-level failure (network, status, decode)."""


def urllib_transport(timeout: float = 10.0) -> Transport:
    """Default transport built on urllib.

    Respects the standard ``HTTPS_PROXY``/``HTTP_PROXY`` environment variables
    (urllib installs a ProxyHandler from the environment by default), so it
    works behind the agent proxy without extra configuration.
    """

    def _fetch(url: str, headers: dict) -> dict:
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                status = resp.getcode()
        except urllib.error.HTTPError as exc:
            raise HttpError(f"HTTP {exc.code} for {url}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise HttpError(f"network error for {url}: {exc.reason}") from exc
        except Exception as exc:  # timeouts, etc.
            raise HttpError(f"request failed for {url}: {exc!r}") from exc

        if status and status >= 400:
            raise HttpError(f"HTTP {status} for {url}")
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise HttpError(f"invalid JSON from {url}: {exc}") from exc

    return _fetch


def build_url(base_url: str, path: str, params: Optional[dict] = None) -> str:
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    if params:
        # Drop None values so callers can pass optional params freely.
        clean = {k: v for k, v in params.items() if v is not None}
        if clean:
            url += "?" + urllib.parse.urlencode(clean)
    return url
