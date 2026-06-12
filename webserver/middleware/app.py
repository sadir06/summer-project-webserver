from pathlib import Path
import threading
import time

from flask import Flask, jsonify, render_template, request

from hardware.config import FLASK_HOST, FLASK_PORT, LAPTOP_IP
from hardware.inference_telemetry import append_inference_tick, get_inference_telemetry
from hardware.load_demand import get_model_load_demand
from webserver.middleware.poller import poll_loop
from webserver.middleware.state import cache, lock

WEBSERVER_DIR = Path(__file__).resolve().parent.parent

app = Flask(
    __name__,
    template_folder=str(WEBSERVER_DIR / "templates"),
    static_folder=str(WEBSERVER_DIR / "static"),
)


@app.route("/api/state")
def get_state():
    with lock:
        payload = dict(cache)
    payload["inference"] = get_inference_telemetry()
    return jsonify(payload)


@app.route("/api/inference_tick", methods=["POST"])
def post_inference_tick():
    data = request.get_json(force=True, silent=True) or {}
    data.setdefault("ts", time.time())
    append_inference_tick(data)
    return jsonify({"ok": True})


@app.route("/api/load_demand")
def get_load_demand():
    """Load Pico polls this for model total_demand (instant + defer), each tick."""
    return jsonify(get_model_load_demand())


@app.route("/api/sun_data")
def get_sun_data():
    with lock:
        sun_data = cache.get("sun")

        if isinstance(sun_data, dict):
            sun_value = sun_data.get("sun")
            sun_tick = sun_data.get("tick")
        else:
            sun_value = sun_data
            sun_tick = None

    return jsonify({
        "sun": sun_value,
        "tick": sun_tick,
    })


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    threading.Thread(target=poll_loop, daemon=True).start()
    print(f"Flask listening on {FLASK_HOST}:{FLASK_PORT}")
    print(f"MPPT Pico should poll: http://{LAPTOP_IP}:{FLASK_PORT}/api/sun_data")
    print(f"Load Pico should poll: http://{LAPTOP_IP}:{FLASK_PORT}/api/load_demand")
    app.run(host=FLASK_HOST, port=FLASK_PORT)
