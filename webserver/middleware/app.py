import argparse
import atexit
from pathlib import Path
import threading
import time

from flask import Flask, jsonify, render_template, request

from hardware.config import FLASK_HOST, FLASK_PORT, LAPTOP_IP
from hardware.inference_recorder import (
    finalize_run,
    get_recording_state,
    start_new_run,
)
from hardware.inference_telemetry import append_inference_tick, get_inference_telemetry
from hardware.load_demand import get_model_load_demand, set_model_load_demand
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


@app.route("/api/inference_run/start", methods=["POST"])
def post_inference_run_start():
    body = request.get_json(force=True, silent=True) or {}
    run_dir = start_new_run(body.get("run_name"), meta=body.get("meta"))
    return jsonify({"ok": True, "run_dir": str(run_dir)})


@app.route("/api/inference_run")
def get_inference_run():
    return jsonify(get_recording_state())


@app.route("/api/inference_tick", methods=["POST"])
def post_inference_tick():
    data = request.get_json(force=True, silent=True) or {}
    data.setdefault("ts", time.time())
    append_inference_tick(data)
    # Mirror load command for Load Pico polling (inference runs in another process).
    if data.get("total_demand_w") is not None:
        set_model_load_demand(
            day=int(data["day"]),
            tick=int(data["tick"]),
            total_demand_w=float(data["total_demand_w"]),
        )
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


def parse_app_args():
    parser = argparse.ArgumentParser(description="Flask middleware for Pico telemetry + dashboard")
    parser.add_argument(
        "--new-inference-run",
        action="store_true",
        help="Start recording inference ticks to data/inference_runs/<timestamp>/",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional label for the new inference run folder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli = parse_app_args()
    atexit.register(finalize_run)

    if cli.new_inference_run:
        run_dir = start_new_run(cli.run_name)
        print(f"Inference recording -> {run_dir}")
        print(f"  ticks.jsonl (every tick) + latest.json (every 25 ticks)")

    threading.Thread(target=poll_loop, daemon=True).start()
    print(f"Flask listening on {FLASK_HOST}:{FLASK_PORT}")
    print(f"MPPT Pico should poll: http://{LAPTOP_IP}:{FLASK_PORT}/api/sun_data")
    print(f"Load Pico should poll: http://{LAPTOP_IP}:{FLASK_PORT}/api/load_demand")
    app.run(host=FLASK_HOST, port=FLASK_PORT)
