from flask import Flask, jsonify, render_template
import threading
from state import cache, lock
from poller import poll_loop

app = Flask(__name__)

@app.route("/api/state")
def get_state():
    with lock:
        return jsonify(cache) # Return the cache as a JSON object
    
@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    threading.Thread(target=poll_loop, daemon=True).start() # Start the polling loop in a separate thread
    app.run(host="0.0.0.0", port=8000)

