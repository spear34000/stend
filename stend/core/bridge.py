import websocket
import threading
import json
import time

class IrisBridge:
    def __init__(self, url="ws://localhost:3000/ws", on_message=None):
        self.url = url
        self.ws = None
        self.on_message_callback = on_message
        self.keep_running = False
        self.thread = None

    def start(self):
        self.keep_running = True
        self.thread = threading.Thread(target=self._run_loop)
        self.thread.daemon = True
        self.thread.start()

    def _run_loop(self):
        while self.keep_running:
            try:
                self.ws = websocket.WebSocketApp(
                    self.url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                self.ws.run_forever()
            except Exception as e:
                print(f"[Bridge] Connection failed: {e}")
            
            if self.keep_running:
                print("[Bridge] Reconnecting in 3s...")
                time.sleep(3)

    def _on_open(self, ws):
        print("[Bridge] Connected to Iris Android Subsystem")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if self.on_message_callback:
                self.on_message_callback(data)
        except Exception as e:
            print(f"[Bridge] Message parse error: {e}")

    def _on_error(self, ws, error):
        print(f"[Bridge] Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print("[Bridge] Disconnected")

    def stop(self):
        self.keep_running = False
        if self.ws:
            self.ws.close()
