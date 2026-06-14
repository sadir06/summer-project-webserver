import concurrent.futures
import json

from hardware.config import (
    INFERENCE_PICO_TIMEOUT_S,
    PICO_CAP_DATA_URL,
    PICO_ENDPOINTS,
    POLL_TIMEOUT_S,
)
from hardware.pico_http import pico_get_for_base

_last_known: dict[str, float | None] = {
    "pvout": None,
    "pcout": None,
    "vcap": None,
}


def _poll_plain(base_url: str, path: str, timeout_s: float) -> float:
    response = pico_get_for_base(base_url, path, timeout=timeout_s)
    response.raise_for_status()
    text = response.text.strip().splitlines()[0].strip()
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"non-numeric response from {base_url}{path}: {text[:80]!r}") from None


def _poll_cap_voltage(timeout_s: float) -> float:
    response = pico_get_for_base(PICO_CAP_DATA_URL, "/data", timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and "va" in payload:
        return float(payload["va"])
    raise ValueError("cap /data missing va")


def read_hardware(
    timeout_s: float | None = None,
    *,
    use_last_known: bool = False,
) -> dict[str, float | None]:
    """Poll Picos in parallel."""
    t = float(timeout_s if timeout_s is not None else POLL_TIMEOUT_S)
    readings: dict[str, float | None] = {
        "pvout": None,
        "pcout": None,
        "vcap": None,
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pv_f = pool.submit(
            _poll_plain,
            PICO_ENDPOINTS["pvout"]["base_url"],
            PICO_ENDPOINTS["pvout"]["path"],
            t,
        )
        pc_f = pool.submit(
            _poll_plain,
            PICO_ENDPOINTS["pcout"]["base_url"],
            PICO_ENDPOINTS["pcout"]["path"],
            t,
        )
        vc_f = pool.submit(_poll_cap_voltage, t)

        for field, fut in (("pvout", pv_f), ("pcout", pc_f)):
            try:
                readings[field] = fut.result()
            except Exception as e:
                print(f"Error polling Pico {field}: {e}")

        try:
            readings["vcap"] = vc_f.result()
        except json.JSONDecodeError as e:
            print(f"Error parsing cap /data JSON: {e}")
        except Exception as e:
            print(f"Error polling Pico vcap: {e}")

    if use_last_known:
        for field, value in readings.items():
            if value is not None:
                _last_known[field] = value
            elif _last_known[field] is not None:
                readings[field] = _last_known[field]

    return readings


def read_hardware_for_inference() -> dict[str, float | None]:
    """PV output and cap voltage for inference."""
    t = float(INFERENCE_PICO_TIMEOUT_S)
    readings: dict[str, float | None] = {"pvout": None, "vcap": None}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pv_f = pool.submit(
            _poll_plain,
            PICO_ENDPOINTS["pvout"]["base_url"],
            PICO_ENDPOINTS["pvout"]["path"],
            t,
        )
        vc_f = pool.submit(_poll_cap_voltage, t)

        try:
            readings["pvout"] = pv_f.result()
        except Exception as e:
            print(f"Error polling Pico pvout: {e}")

        try:
            readings["vcap"] = vc_f.result()
        except json.JSONDecodeError as e:
            print(f"Error parsing cap /data JSON: {e}")
        except Exception as e:
            print(f"Error polling Pico vcap: {e}")

    for field, value in readings.items():
        if value is not None:
            _last_known[field] = value
        elif _last_known.get(field) is not None:
            readings[field] = _last_known[field]

    return readings


def poll_hardware(cache: dict, lock) -> None:
    """Dashboard poll: pvout and vcap only."""
    readings = read_hardware_for_inference()
    with lock:
        for field, value in readings.items():
            if value is not None:
                cache[field] = value
            elif field not in cache:
                cache[field] = None
