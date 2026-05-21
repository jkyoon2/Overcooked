from __future__ import annotations

import time

from fastapi import WebSocket, WebSocketDisconnect

from backend.data.schema import HelloAckMessage, HelloAckPayload, HelloMessage


async def websocket_handler(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            msg = HelloMessage.model_validate(data)
            if msg.type == "hello":
                # Timestamp captured synchronously — no I/O between receive and send (C1)
                timestamp_ms = int(time.time() * 1000)
                ack = HelloAckMessage(
                    type="hello_ack",
                    payload=HelloAckPayload(
                        echo=msg.payload.message,
                        server_timestamp_ms=timestamp_ms,
                    ),
                )
                await websocket.send_json(ack.model_dump())
    except WebSocketDisconnect:
        pass
