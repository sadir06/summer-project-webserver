# Raspberry Pi Pico boards expose plain-text HTTP endpoints on the local network.

PICO_URL_PV = "http://172.20.10.4"
PICO_URL_CP = "http://172.20.10.8"

PICO_ENDPOINTS = {
    "pvout": {"base_url": PICO_URL_PV, "path": "/pout"},
    "pcout": {"base_url": PICO_URL_CP, "path": "/p"},
    "vcap": {"base_url": PICO_URL_CP, "path": "/va"},
}

POLL_TIMEOUT_S = 1
