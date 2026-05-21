from __future__ import annotations

from pydantic import BaseModel


class HelloPayload(BaseModel):
    message: str


class HelloMessage(BaseModel):
    type: str
    payload: HelloPayload


class HelloAckPayload(BaseModel):
    echo: str
    server_timestamp_ms: int


class HelloAckMessage(BaseModel):
    type: str
    payload: HelloAckPayload
