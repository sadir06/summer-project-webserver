from pathlib import Path
import threading

from flask import Flask, jsonify, render_template

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
        return jsonify(cache)


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
    app.run(host="0.0.0.0", port=8000)
