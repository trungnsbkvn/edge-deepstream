import os
import json
import threading
from typing import Callable, Optional

import paho.mqtt.client as mqtt


class MQTTListener:
    def __init__(self, host: str, port: int, topic: str, on_message: Callable[[dict], None], username: Optional[str] = None, password: Optional[str] = None):
        self.host = host
        self.port = int(port)
        self.topic = topic
        self.on_message = on_message
        self.username = username
        self.password = password
        self._client = mqtt.Client()
        if username and password:
            self._client.username_pw_set(username, password)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._thread: Optional[threading.Thread] = None

    def _on_connect(self, client, userdata, flags, rc):
        try:
            # Subscribe with QoS 2 to match core_v2 settings
            client.subscribe(self.topic, qos=2)
        except Exception:
            pass

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8', errors='ignore')
            try:
                data = json.loads(payload)
            except Exception:
                data = {'raw': msg.payload}
        except Exception:
            data = {'raw': msg.payload}
        try:
            self.on_message(data)
        except Exception:
            pass

    def start(self):
        self._client.connect(self.host, self.port, keepalive=30)
        # Run network loop in background thread
        self._thread = threading.Thread(target=self._client.loop_forever, daemon=True)
        self._thread.start()

    def stop(self):
        try:
            self._client.disconnect()
        except Exception:
            pass
        if self._thread:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
            self._thread = None

    def publish(self, topic: str, payload: str, qos: int = 2, retain: bool = False):
        try:
            self._client.publish(topic, payload=payload, qos=qos, retain=retain)
        except Exception:
            pass
