"""HTTP to Pico boards. Cap Pico requests are serialized."""

from __future__ import annotations

import threading

from hardware.config import PICO_URL_CP
from hardware.http_client import lan_get

_cap_lock = threading.Lock()
_CLOSE = {"Connection": "close"}


def _is_cap_host(base_url: str) -> bool:
    return base_url.rstrip("/") == PICO_URL_CP.rstrip("/")


def pico_get(url: str, *, cap_host: bool = False, **kwargs) -> object:
    """GET with Connection: close; cap host requests are serialized."""
    headers = {**_CLOSE, **kwargs.pop("headers", {})}
    if cap_host:
        with _cap_lock:
            return lan_get(url, headers=headers, **kwargs)
    return lan_get(url, headers=headers, **kwargs)


def pico_get_for_base(base_url: str, path: str, **kwargs) -> object:
    return pico_get(f"{base_url}{path}", cap_host=_is_cap_host(base_url), **kwargs)
