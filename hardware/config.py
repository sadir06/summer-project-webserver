# Raspberry Pi Pico boards expose plain-text HTTP endpoints on the local network.

CLOUD_BASE_URL = "https://icelec50015.azurewebsites.net"

# Input Picos (telemetry)
PICO_URL_PV = "http://172.20.10.4"
PICO_URL_CP = "http://172.20.10.8"

PICO_ENDPOINTS = {
    "pvout": {"base_url": PICO_URL_PV, "path": "/pout"},
    "pcout": {"base_url": PICO_URL_CP, "path": "/p"},
    "vcap": {"base_url": PICO_URL_CP, "path": "/va"},
}

# Output Picos (policy actions) — grid/def IPs are placeholders until known.
# Supercap action goes to the same capacitor Pico that reports /p and /va.
PICO_URL_GRID_ACTION = "http://172.20.10.5"  
PICO_URL_DEF_ACTION = "http://172.20.10.7"  

PICO_ACTION_ENDPOINTS = {
    "grid": {"base_url": PICO_URL_GRID_ACTION, "path": "/action"},
    "sc": {"base_url": PICO_URL_CP, "path": "/action"},
    "def": {"base_url": PICO_URL_DEF_ACTION, "path": "/action"},
}

POLL_TIMEOUT_S = 1
ACTION_TIMEOUT_S = 1
TICK_INTERVAL_S = 5
