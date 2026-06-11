# JS cannot directly read from the web server, so we poll Azure + Pico boards in Python
# and pass the merged cache to the frontend via Flask.

import time
from datetime import datetime, timezone

import requests

from hardware.config import CLOUD_BASE_URL
from hardware.poller import poll_hardware
from webserver.middleware.state import cache, lock

cloud_counter = 0


def poll_loop():
    global cloud_counter

    while True:
        if cloud_counter == 0:
            try:
                response_sun = requests.get(f"{CLOUD_BASE_URL}/sun", timeout=5)
                response_price = requests.get(f"{CLOUD_BASE_URL}/price", timeout=5)
                response_demand = requests.get(f"{CLOUD_BASE_URL}/demand", timeout=5)
                response_deferables = requests.get(f"{CLOUD_BASE_URL}/deferables", timeout=5)
                response_yesterday = requests.get(f"{CLOUD_BASE_URL}/yesterday", timeout=5)

                response_sun.raise_for_status()
                response_price.raise_for_status()
                response_demand.raise_for_status()
                response_deferables.raise_for_status()
                response_yesterday.raise_for_status()

                with lock:
                    cache["sun"] = response_sun.json()
                    cache["price"] = response_price.json()
                    cache["demand"] = response_demand.json()
                    cache["deferables"] = response_deferables.json()
                    cache["yesterday"] = response_yesterday.json()
                    cache["last_updated"] = datetime.now(timezone.utc).isoformat()

            except Exception as e:
                print(f"Error polling main webserver data: {e}")
                with lock:
                    cache["last_updated"] = datetime.now(timezone.utc).isoformat()

        cloud_counter += 1
        if cloud_counter >= 5:
            cloud_counter = 0

        poll_hardware(cache, lock)
        time.sleep(1)
