from __future__ import annotations

from fastapi import FastAPI, WebSocket

from backend.api.websocket import websocket_handler

app = FastAPI(title="Neurocontroller Backend")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket_handler(websocket)
