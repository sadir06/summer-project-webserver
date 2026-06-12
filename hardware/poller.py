import json

import requests

from hardware.config import PICO_CAP_DATA_URL, PICO_ENDPOINTS, POLL_TIMEOUT_S


def _poll_plain(base_url: str, path: str) -> float | None:
    response = requests.get(f"{base_url}{path}", timeout=POLL_TIMEOUT_S)
    response.raise_for_status()
    text = response.text.strip().splitlines()[0].strip()
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"non-numeric response from {base_url}{path}: {text[:80]!r}") from None


def _poll_cap_voltage() -> float | None:
    response = requests.get(f"{PICO_CAP_DATA_URL}/data", timeout=POLL_TIMEOUT_S)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and "va" in payload:
        return float(payload["va"])
    return None


def read_hardware() -> dict[str, float | None]:
    readings: dict[str, float | None] = {
        "pvout": None,
        "pcout": None,
        "vcap": None,
    }

    for field, endpoint in PICO_ENDPOINTS.items():
        try:
            readings[field] = _poll_plain(endpoint["base_url"], endpoint["path"])
        except Exception as e:
            print(f"Error polling Pico {field}: {e}")

    try:
        readings["vcap"] = _poll_cap_voltage()
    except json.JSONDecodeError as e:
        print(f"Error parsing cap /data JSON: {e}")
    except Exception as e:
        print(f"Error polling Pico vcap: {e}")

    return readings


def poll_hardware(cache: dict, lock) -> None:
    readings = read_hardware()
    with lock:
        for field, value in readings.items():
            if value is not None:
                cache[field] = value
            elif field not in cache:
                cache[field] = None
