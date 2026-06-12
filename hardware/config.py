# Raspberry Pi Pico boards expose plain-text HTTP endpoints on the phone hotspot.
# See hardware/SETUP.md for laptop static-IP instructions.

CLOUD_BASE_URL = "https://icelec50015.azurewebsites.net"

# -----------------------------------------------------------------------------
# Laptop (runs Flask + inference.py)
# -----------------------------------------------------------------------------
# Picos poll the laptop at LAPTOP_IP (Load: /api/load_demand, optional MPPT: /api/sun_data).
# Your laptop Wi-Fi adapter MUST use this IP when connected to the hotspot.
LAPTOP_IP = "10.36.4.205"
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8000
# Local hotspot gateway (run `ipconfig` on laptop when on hotspot — NOT the phone's public IP).
HOTSPOT_GATEWAY = "10.36.4.204"
# -----------------------------------------------------------------------------
# Input Picos (telemetry) — set IPs from each Pico's serial boot log ("Pico IP:")
# Do NOT use LAPTOP_IP here; .3 is the laptop, not a Pico.
# -----------------------------------------------------------------------------
PICO_URL_PV = "http://10.36.4.206"  # MPPT Pico (MPPTfinal) — update after boot
PICO_URL_CP = "http://10.36.4.215"  # Capacitor Pico (capwifi)

PICO_ENDPOINTS = {
    "pvout": {"base_url": PICO_URL_PV, "path": "/pout"},
    "pcout": {"base_url": PICO_URL_CP, "path": "/p"},
}

# Cap voltage (va) is only available as JSON on GET /data (capwifi firmware).
PICO_CAP_DATA_URL = PICO_URL_CP

# -----------------------------------------------------------------------------
# Output Picos (policy actions) — firmware must handle GET /action?value=<float>
# Set ACTIONS_ENABLED = True once Pico handlers exist.
# -----------------------------------------------------------------------------
ACTIONS_ENABLED = True
ACTION_QUERY_PARAM = "value"

PICO_URL_GRID_ACTION = "http://172.20.10."  # PLACEHOLDER
PICO_URL_DEF_ACTION = "http://172.20.10.7"  # Load Pico — unused when polling laptop
LOAD_DEMAND_PATH = "/demand"  # POST JSON {day, tick, total_demand}
# Load Pico polls GET /api/load_demand on the laptop — leave False unless firmware accepts POST.
LOAD_DEMAND_POST_ENABLED = False

PICO_ACTION_ENDPOINTS = {
    "grid": {"base_url": PICO_URL_GRID_ACTION, "path": "/action"},
    "sc": {"base_url": PICO_URL_CP, "path": "/action"},
}

# Model sc_action is [-1, 1]; capwifi pout_ref is watts in roughly [-3, 3].
SC_ACTION_WATTS_SCALE = 3.0

POLL_TIMEOUT_S = 1
ACTION_TIMEOUT_S = 1
TICK_INTERVAL_S = 5
