# Raspberry Pi Pico boards expose plain-text HTTP endpoints on the phone hotspot.
# See hardware/SETUP.md for laptop static-IP instructions.

CLOUD_BASE_URL = "https://icelec50015.azurewebsites.net"

# -----------------------------------------------------------------------------
# Laptop (runs Flask + inference.py)
# -----------------------------------------------------------------------------
# MPPT Pico firmware hardcodes SUN_SERVER_HOST = "172.20.10.4" and polls
#   GET http://172.20.10.4:8000/api/sun_data
# Your laptop Wi-Fi adapter MUST use this IP when connected to the hotspot.
LAPTOP_IP = "172.20.10.4"
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8000
# Local hotspot gateway (run `ipconfig` on laptop when on hotspot — NOT the phone's public IP).
HOTSPOT_GATEWAY = "172.20.10.1"
# -----------------------------------------------------------------------------
# Input Picos (telemetry) — set IPs from each Pico's serial boot log ("Pico IP:")
# Do NOT use LAPTOP_IP here; .4 is the laptop, not a Pico.
# -----------------------------------------------------------------------------
PICO_URL_PV = "http://172.20.10.5"  # MPPT Pico (MPPTfinal) — update after boot
PICO_URL_CP = "http://172.20.10.2"  # Capacitor Pico (capwifi)

PICO_ENDPOINTS = {
    "pvout": {"base_url": PICO_URL_PV, "path": "/pout"},
    "pcout": {"base_url": PICO_URL_CP, "path": "/pout"},
}

# Cap voltage (va) is only available as JSON on GET /data (capwifi firmware).
PICO_CAP_DATA_URL = PICO_URL_CP

# -----------------------------------------------------------------------------
# Output Picos (policy actions) — firmware must handle GET /action?value=<float>
# Set ACTIONS_ENABLED = True once Pico handlers exist.
# -----------------------------------------------------------------------------
ACTIONS_ENABLED = False
ACTION_QUERY_PARAM = "value"

PICO_URL_GRID_ACTION = "http://172.20.10.6"  # PLACEHOLDER
PICO_URL_DEF_ACTION = "http://172.20.10.7"  # PLACEHOLDER

PICO_ACTION_ENDPOINTS = {
    "grid": {"base_url": PICO_URL_GRID_ACTION, "path": "/action"},
    "sc": {"base_url": PICO_URL_CP, "path": "/action"},
    "def": {"base_url": PICO_URL_DEF_ACTION, "path": "/action"},
}

# Model sc_action is [-1, 1]; capwifi pout_ref is watts in roughly [-3, 3].
SC_ACTION_WATTS_SCALE = 3.0

POLL_TIMEOUT_S = 1
ACTION_TIMEOUT_S = 1
TICK_INTERVAL_S = 5
