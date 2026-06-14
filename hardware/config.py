# Pico HTTP on the phone hotspot. See hardware/SETUP.md for laptop static IP.

CLOUD_BASE_URL = "https://icelec50015.azurewebsites.net"

LAPTOP_IP = "10.36.4.205"
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8000
HOTSPOT_GATEWAY = "10.36.4.129"

PICO_URL_PV = "http://10.36.4.206"
PICO_URL_CP = "http://10.36.4.215"

PICO_ENDPOINTS = {
    "pvout": {"base_url": PICO_URL_PV, "path": "/pout"},
    "pcout": {"base_url": PICO_URL_CP, "path": "/p"},
}

PICO_CAP_DATA_URL = PICO_URL_CP

ACTIONS_ENABLED = True
ACTION_QUERY_PARAM = "value"

PICO_URL_GRID_ACTION = "http://172.20.10."
PICO_URL_DEF_ACTION = "http://172.20.10.7"
LOAD_DEMAND_PATH = "/demand"
LOAD_DEMAND_POST_ENABLED = False

PICO_ACTION_ENDPOINTS = {
    "grid": {"base_url": PICO_URL_GRID_ACTION, "path": "/action"},
    "sc": {"base_url": PICO_URL_CP, "path": "/action"},
}

SC_ACTION_WATTS_SCALE = 3.0

POLL_TIMEOUT_S = 1
ACTION_TIMEOUT_S = 1
TICK_INTERVAL_S = 5
INFERENCE_PICO_TIMEOUT_S = 0.8
INFERENCE_ACTION_TIMEOUT_S = 0.8
CLOUD_TICK_TIMEOUT_S = 2.0
CLOUD_SNAPSHOT_TIMEOUT_S = 4.0
