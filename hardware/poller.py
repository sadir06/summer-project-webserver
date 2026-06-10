import requests

from hardware.config import PICO_ENDPOINTS, POLL_TIMEOUT_S


def _poll_endpoint(field: str, base_url: str, path: str) -> float | None:
    try:
        response = requests.get(f"{base_url}{path}", timeout=POLL_TIMEOUT_S)
        response.raise_for_status()
        return float(response.text.strip())
    except Exception as e:
        print(f"Error polling Pico {field}: {e}")
        return None


def poll_hardware(cache: dict, lock) -> None:
    for field, endpoint in PICO_ENDPOINTS.items():
        value = _poll_endpoint(field, endpoint["base_url"], endpoint["path"])
        with lock:
            if value is not None:
                cache[field] = value
            elif field not in cache:
                cache[field] = None
