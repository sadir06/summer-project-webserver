import requests

from hardware.config import (
    ACTION_QUERY_PARAM,
    ACTION_TIMEOUT_S,
    ACTIONS_ENABLED,
    PICO_ACTION_ENDPOINTS,
    SC_ACTION_WATTS_SCALE,
)


def _scale_sc_action(sc_action: float) -> float:
    return float(sc_action) * SC_ACTION_WATTS_SCALE


def send_action(name: str, value: float) -> bool:
    if not ACTIONS_ENABLED:
        return False

    endpoint = PICO_ACTION_ENDPOINTS[name]
    url = f"{endpoint['base_url']}{endpoint['path']}"
    try:
        response = requests.get(
            url,
            params={ACTION_QUERY_PARAM: float(value)},
            timeout=ACTION_TIMEOUT_S,
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error sending {name} action to {url}: {e}")
        return False


def send_actions(
    sc_action: float,
    *,
    grid_action: float | None = None,
) -> dict[str, bool]:
    """Send supercap (and optional grid) actions directly to Picos.

    Load/defer demand is published separately via cloud.publish_load_demand().
  """
    if not ACTIONS_ENABLED:
        print(
            "actions skipped (ACTIONS_ENABLED=False; Pico firmware has no GET /action yet)"
        )
        results = {"sc": False}
        if grid_action is not None:
            results["grid"] = False
        return results

    results = {
        "sc": send_action("sc", _scale_sc_action(sc_action)),
    }
    if grid_action is not None:
        results["grid"] = send_action("grid", grid_action)
    return results
