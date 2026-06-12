"""Shared HTTP sessions for inference (keep-alive, no urllib3 retries)."""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

_NO_RETRY = Retry(total=0, connect=0, read=0, redirect=0, status=0)

_cloud: requests.Session | None = None
_lan: requests.Session | None = None


def _make_session(*, pool_size: int) -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(
        max_retries=_NO_RETRY,
        pool_connections=pool_size,
        pool_maxsize=pool_size,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def cloud_get(url: str, **kwargs):
    global _cloud
    if _cloud is None:
        _cloud = _make_session(pool_size=6)
    return _cloud.get(url, **kwargs)


def lan_get(url: str, **kwargs):
    global _lan
    if _lan is None:
        _lan = _make_session(pool_size=4)
    return _lan.get(url, **kwargs)


def lan_post(url: str, **kwargs):
    global _lan
    if _lan is None:
        _lan = _make_session(pool_size=4)
    return _lan.post(url, **kwargs)
