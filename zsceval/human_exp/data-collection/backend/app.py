from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is in path for zsceval.* imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from flask import Flask
from flask_socketio import SocketIO

from api.routes import register

app = Flask(__name__)
app.config["SECRET_KEY"] = "dc-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

register(app, socketio)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
