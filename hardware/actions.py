import requests

from hardware.config import ACTION_TIMEOUT_S, PICO_ACTION_ENDPOINTS


def send_action(name: str, value: float) -> bool:
    endpoint = PICO_ACTION_ENDPOINTS[name]
    url = f"{endpoint['base_url']}{endpoint['path']}"
    try:
        response = requests.post(
            url,
            data=str(float(value)),
            headers={"Content-Type": "text/plain"},
            timeout=ACTION_TIMEOUT_S,
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error sending {name} action to {url}: {e}")
        return False


def send_actions(grid_action: float, sc_action: float, def_action: float) -> dict[str, bool]:
    return {
        "grid": send_action("grid", grid_action),
        "sc": send_action("sc", sc_action),
        "def": send_action("def", def_action),
    }
