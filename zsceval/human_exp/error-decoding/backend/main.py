from __future__ import annotations

import os
import sys

from fastapi import FastAPI, WebSocket

from backend.api.websocket import websocket_handler

app = FastAPI(title="Neurocontroller Backend")

# --pygame flag: run with `uvicorn backend.main:app -- --pygame`
# or set env var NEUROCONTROLLER_PYGAME=1
_USE_PYGAME = "--pygame" in sys.argv or os.environ.get("NEUROCONTROLLER_PYGAME") == "1"

if _USE_PYGAME:
    from backend.game.pygame_window import start_pygame_thread

    start_pygame_thread()
    print("[main] Pygame verification window started (Task 2 throwaway tool).")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket_handler(websocket)
