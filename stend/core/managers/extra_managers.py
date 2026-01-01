import requests
import threading
import json
from typing import Dict, Any, List

class WebhookManager:
    def __init__(self):
        self.webhooks: List[str] = []
        self._lock = threading.Lock()

    def add_webhook(self, url: str):
        with self._lock:
            if url not in self.webhooks:
                self.webhooks.append(url)

    def remove_webhook(self, url: str):
        with self._lock:
            if url in self.webhooks:
                self.webhooks.remove(url)

    def trigger(self, event_type: str, data: Any):
        payload = {
            "event": event_type,
            "data": data
        }
        # Run in background to avoid blocking
        def _send():
            with self._lock:
                current_webhooks = list(self.webhooks)
            for url in current_webhooks:
                try:
                    requests.post(url, json=payload, timeout=2)
                except Exception as e:
                    print(f"Webhook error ({url}): {e}")
        
        threading.Thread(target=_send).start()

class SharedStateManager:
    def __init__(self):
        self.state: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: Any):
        with self._lock:
            self.state[key] = value

    def get(self, key: str) -> Any:
        with self._lock:
            return self.state.get(key)

    def delete(self, key: str):
        with self._lock:
            if key in self.state:
                del self.state[key]

    def get_all(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.state)
