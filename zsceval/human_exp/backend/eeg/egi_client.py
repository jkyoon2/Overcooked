from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class EGIConfig:
    enabled: bool = os.environ.get("EGI_ENABLED", "0") == "1"
    ip_ns: str = os.environ.get("EGI_NETSTATION_IP", "10.10.10.42")
    ip_amp: str = os.environ.get("EGI_AMP_IP", "10.10.10.51")
    port_ns: int = int(os.environ.get("EGI_PORT", "55513"))


class EGIClient:
    def __init__(self, config: EGIConfig | None = None):
        self.config = config or EGIConfig()
        self.client = None

    def connect_and_begin(self) -> None:
        if not self.config.enabled:
            print("[EGI] disabled: EGI_ENABLED is not set to 1")
            return

        try:
            from egi_pynetstation.NetStation import NetStation

            self.client = NetStation(self.config.ip_ns, self.config.port_ns)
            self.client.connect(ntp_ip=self.config.ip_amp)
            self.client.begin_rec()
            print("[EGI] connected and recording started")
        except Exception as exc:
            print(f"[EGI] connect_and_begin failed: {exc}")
            self.client = None

    def send_event(self, code: str, label: str | None = None) -> None:
        if not self.config.enabled or self.client is None:
            return

        # EGI event_type은 짧게 유지하는 편이 안전합니다.
        event_type = code[:4]
        self.client.send_event(event_type=event_type, label=label or event_type)

    def end_and_disconnect(self) -> None:
        if not self.config.enabled or self.client is None:
            return

        try:
            self.client.end_rec()
        finally:
            self.client.disconnect()
            self.client = None