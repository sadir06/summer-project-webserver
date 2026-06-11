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


def read_hardware() -> dict[str, float | None]:
    readings: dict[str, float | None] = {}
    for field, endpoint in PICO_ENDPOINTS.items():
        readings[field] = _poll_endpoint(field, endpoint["base_url"], endpoint["path"])
    return readings


def poll_hardware(cache: dict, lock) -> None:
    readings = read_hardware()
    with lock:
        for field, value in readings.items():
            if value is not None:
                cache[field] = value
            elif field not in cache:
                cache[field] = None
