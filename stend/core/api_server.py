import os
import threading
import asyncio
import json
import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Stend Managers
from stend.core.managers.adb import AdbManager
from stend.core.managers.bridge import IrisBridge
from stend.core.managers.skill_manager import SkillManager
from stend.core.managers.link_manager import KakaoLinkManager
from stend.core.managers.store_manager import StendStore
from stend.core.managers.extra_managers import WebhookManager, SharedStateManager

app = FastAPI(title="Stend API Platform")

# --- Constants & Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = os.path.join(BASE_DIR, "skills")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

# --- Models ---
class SystemStatus(BaseModel):
    android: str
    iris_bridge: str
    skills_active: int

class CommandResponse(BaseModel):
    status: str
    message: str

# --- Global State ---
adb = AdbManager(target="127.0.0.1:5555")
skills = SkillManager(SKILLS_DIR)
links = KakaoLinkManager(iris_url="http://localhost:3000")
store = StendStore()
webhooks = WebhookManager()
shared_state = SharedStateManager()
bridge = None
main_loop = None

SYSTEM_STATUS = {
    "android": "unknown",
    "iris_bridge": "disconnected",
    "skills_active": 0
}

# --- WebSocket Log Broadcasting ---
class LogManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

log_manager = LogManager()

def stend_log(msg: str):
    print(f"[Stend] {msg}")
    if main_loop and main_loop.is_running():
        try:
            asyncio.run_coroutine_threadsafe(log_manager.broadcast(msg), main_loop)
        except:
            pass

# --- Core Lifecycle ---
def lifecycle_start():
    global bridge
    try:
        stend_log("System Lifecycle Starting...")
        SYSTEM_STATUS["android"] = "connecting"
        
        # 1. Device Check
        if not adb.wait_for_device(timeout=15):
             stend_log("ERROR: Android Device not found via ADB.")
             SYSTEM_STATUS["android"] = "error"
             return

        # 2. Deploy & Start Android Subsystem
        # Path to recently built APK
        apk_path = os.path.join(BASE_DIR, "android_project", "output", "Iris-debug.apk")
        if os.path.exists(apk_path):
            stend_log("Subsystem APK found, deploying...")
            if adb.start_iris_process(apk_path):
                SYSTEM_STATUS["android"] = "running"
            else:
                SYSTEM_STATUS["android"] = "error"
        else:
            stend_log("WARNING: Subsystem APK not found. Please build the project.")
            SYSTEM_STATUS["android"] = "error"

        # 3. Port Forwarding for Iris
        stend_log("Setting up port forwarding (3000 -> 3000)...")
        adb.setup_port_forward(3000, 3000)

        # 4. Load Skills
        active = skills.load_skills()
        SYSTEM_STATUS["skills_active"] = len(active)
        stend_log(f"Skills Loaded: {', '.join(active)}")

        # 4. Start Bridge
        def dispatch(raw_data):
            try:
                data = json.loads(raw_data)
                # Handle standard messages
                if "msg" in data:
                    skills.dispatch("message", data)
                    stend_log(f"Message from {data.get('sender', 'Unknown')}")
                
                # [NEW] Handle high-level events (Nickname, Delete, Hide)
                elif data.get("type") == "stend_event":
                    event_name = data.get("event")
                    stend_log(f"Event Detected: {event_name}")
                    skills.dispatch("stend_event", data)
                    # Trigger webhooks
                    webhooks.trigger(event_name, data)
                
                # Trigger webhooks for all messages too if desired
                if "msg" in data:
                    webhooks.trigger("message", data)
                
            except Exception as e:
                stend_log(f"Dispatch Error: {e}")

        bridge = IrisBridge(on_message=dispatch)
        bridge.start()
        SYSTEM_STATUS["iris_bridge"] = "connected"
        stend_log("Stend Platform Ready")
        
    except Exception as e:
        stend_log(f"Fatal Lifecycle Error: {e}")
        SYSTEM_STATUS["android"] = "error"

# --- Endpoints ---
@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    return SYSTEM_STATUS

@app.post("/api/control/start")
async def start_system():
    threading.Thread(target=lifecycle_start).start()
    return CommandResponse(status="starting", message="Lifecycle initiated")

@app.post("/api/control/reload")
async def reload_skills():
    active = skills.load_skills()
    SYSTEM_STATUS["skills_active"] = len(active)
    return active

@app.post("/api/action/shell")
async def api_shell(command: str):
    res = adb.run(["shell"] + command.split())
    return {"output": res}

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await log_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        log_manager.disconnect(websocket)

# --- Mounting & Startup ---
if os.path.exists(DASHBOARD_DIR):
    app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="static")

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    stend_log("Stend API Node Started")
    threading.Thread(target=lifecycle_start).start()

# --- Grand API Proxy ---
@app.get("/api/stend/rooms")
async def get_rooms():
    try:
        r = requests.get("http://localhost:3000/api/v1/rooms", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stend/rooms/{room_id}/members")
async def get_room_members(room_id: int):
    try:
        r = requests.get(f"http://localhost:3000/api/v1/rooms/{room_id}/members", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stend/rooms/{room_id}/history")
async def get_room_history(room_id: int, limit: int = 100):
    try:
        r = requests.get(f"http://localhost:3000/api/v1/rooms/{room_id}/history?limit={limit}", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stend/chats/{chat_id}/context")
async def get_chat_context(chat_id: int, limit: int = 10, dir: str = "prev"):
    try:
        r = requests.get(f"http://localhost:3000/api/v1/chats/{chat_id}/context?limit={limit}&dir={dir}", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stend/rooms/{room_id}/stats")
async def get_room_stats(room_id: int):
    try:
        r = requests.get(f"http://localhost:3000/api/v1/rooms/{room_id}/stats", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/stend/rooms/{room_id}/read")
async def mark_room_read(room_id: int):
    try:
        r = requests.post(f"http://localhost:3000/api/v1/rooms/{room_id}/read", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stend/users/{user_id}")
async def get_user_info(user_id: int):
    try:
        r = requests.get(f"http://localhost:3000/api/v1/users/{user_id}", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stend/friends")
async def get_friends():
    try:
        r = requests.get("http://localhost:3000/api/v1/friends", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stend/aot")
async def get_aot():
    try:
        r = requests.get("http://localhost:3000/aot", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/stend/query")
async def api_query(req: dict):
    try:
        r = requests.post("http://localhost:3000/query", json=req, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/stend/reply")
async def api_reply(req: dict):
    try:
        r = requests.post("http://localhost:3000/reply", json=req, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stend/rooms/{room_id}/link")
async def get_room_link(room_id: int):
    try:
        r = requests.get(f"http://localhost:3000/api/v1/rooms/{room_id}/link", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stend/db/tables")
async def get_db_tables():
    try:
        r = requests.get(f"http://localhost:3000/api/v1/db/tables", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stend/db/columns")
async def get_db_columns(table: str):
    try:
        r = requests.get(f"http://localhost:3000/api/v1/db/columns?table={table}", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/stend/db/clean")
async def clean_db(days: float = 30.0):
    """
    Cleans chat logs older than X days. 
    Equivalent to legacy KakaoDB.clean_chat_logs
    """
    try:
        import time
        now = time.time()
        limit_ts = int(now - (days * 24 * 60 * 60))
        # Delete via proxy query
        payload = {
            "query": "DELETE FROM chat_logs WHERE created_at < ?",
            "bind": [{"content": str(limit_ts)}]
        }
        r = requests.post("http://localhost:3000/api/v1/query", json=payload, timeout=10)
        return {"success": True, "message": f"Deleted logs older than {days} days"}
    except Exception as e:
        return {"error": str(e)}

# --- Store APIs (PyKV Compatible) ---

@app.get("/api/store/get")
async def store_get(key: str):
    return {"key": key, "value": store.get(key)}

@app.post("/api/store/put")
async def store_put(req: dict):
    # { "key": "...", "value": "..." }
    res = store.put(req.get("key"), req.get("value"))
    return {"success": res}

@app.delete("/api/store/delete")
async def store_delete(key: str):
    res = store.delete(key)
    return {"success": res}

@app.get("/api/store/keys")
async def store_list_keys():
    return {"keys": store.list_keys()}

@app.get("/api/stend/rooms/{room_id}/search")
async def search_room(room_id: int, q: str = "", limit: int = 100):
    try:
        r = requests.get(f"http://localhost:3000/api/v1/rooms/{room_id}/search?q={q}&limit={limit}", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stend/chats/{chat_id}/media_info")
async def get_media_info(chat_id: int):
    try:
        r = requests.get(f"http://localhost:3000/api/v1/chats/{chat_id}/media_info", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/stend/link/send")
async def send_kakaolink(req: dict):
    # Expected: { "receiver": "name", "template_id": 123, "template_args": {...}, "app_key": "...", "origin": "..." }
    try:
        res = links.send(
            req.get("receiver"),
            req.get("template_id"),
            req.get("template_args", {}),
            req.get("app_key"),
            req.get("origin")
        )
        return res
    except Exception as e:
        return {"error": str(e)}

# --- Power Feature Proxies ---

@app.post("/api/stend/chats/{chat_id}/send_direct")
async def send_chat_direct(chat_id: int, msg: str):
    try:
        r = requests.post(f"http://localhost:3000/api/v1/chats/{chat_id}/send_direct", data=msg, timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/stend/rooms/{room_id}/read_direct")
async def mark_read_direct(room_id: int):
    try:
        r = requests.post(f"http://localhost:3000/api/v1/rooms/{room_id}/read_direct", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stend/auth/info")
async def get_auth_info():
    try:
        r = requests.get("http://localhost:3000/api/v1/auth/info", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# --- Webhook Management ---

@app.post("/api/webhook/subscribe")
async def webhook_subscribe(url: str):
    webhooks.add_webhook(url)
    return {"status": "subscribed", "url": url}

@app.post("/api/webhook/unsubscribe")
async def webhook_unsubscribe(url: str):
    webhooks.remove_webhook(url)
    return {"status": "unsubscribed", "url": url}

# --- Shared State API ---

@app.post("/api/shared/set")
async def shared_set(req: dict):
    # { "key": "...", "value": "..." }
    shared_state.set(req.get("key"), req.get("value"))
    return {"status": "ok"}

@app.get("/api/shared/get")
async def shared_get(key: str):
    return {"key": key, "value": shared_state.get(key)}

@app.get("/api/shared/all")
async def shared_all():
    return shared_state.get_all()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
