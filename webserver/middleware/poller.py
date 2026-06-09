# JS cannot directly read from the web server, so we need to poll the server using Python, and then pass the data 
# to the frontend via Flask. 

import requests
import time
from datetime import datetime, timezone
from webserver.middleware.state import cache, lock

BASE_URL = "https://icelec50015.azurewebsites.net"


def poll_loop():
    while True:
        try:    
            response_sun = requests.get(f"{BASE_URL}/sun", timeout=5)
            response_price = requests.get(f"{BASE_URL}/price", timeout=5)
            response_demand = requests.get(f"{BASE_URL}/demand", timeout=5)
            response_deferables = requests.get(f"{BASE_URL}/deferables", timeout=5)
            response_yesterday = requests.get(f"{BASE_URL}/yesterday", timeout=5)

            response_sun.raise_for_status()
            response_price.raise_for_status()
            response_demand.raise_for_status()
            response_deferables.raise_for_status()
            response_yesterday.raise_for_status()

            with lock: # Fetch the data, lock and update the cache

                data_sun = response_sun.json()
                data_price = response_price.json()
                data_demand = response_demand.json()
                data_deferables = response_deferables.json()
                data_yesterday = response_yesterday.json()

                cache["sun"] = data_sun
                cache["price"] = data_price
                cache["demand"] = data_demand
                cache["deferables"] = data_deferables
                cache["yesterday"] = data_yesterday
                cache["last_updated"] = datetime.now(timezone.utc).isoformat() # ISO format is what the frontend expects
        except Exception as e:
            print(f"Error polling data: {e}")
            with lock:
                cache["last_updated"] = datetime.now(timezone.utc).isoformat()
            
        time.sleep(1) # Sleep for 1 second

        




