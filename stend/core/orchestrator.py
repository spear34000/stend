import os
import sys
import threading
import time
import requests
import importlib.util
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from core.adb import AdbManager
from core.bridge import IrisBridge

# Stend System Core (Functional)

app = Flask(__name__, static_folder="../dashboard")
CORS(app)

adb = AdbManager()
bridge = None

SYSTEM_STATUS = {
    "android": "unknown",
    "iris_bridge": "disconnected",
    "active_skills": []
}

SKILLS_MODULES = []

# --- Skill Engine ---
def load_skills():
    global SKILLS_MODULES
    SKILLS_MODULES = []
    skills_dir = os.path.join(os.getcwd(), "skills")
    
    loaded_names = []
    if os.path.exists(skills_dir):
        for f in os.listdir(skills_dir):
            if f.endswith(".py") and not f.startswith("__"):
                path = os.path.join(skills_dir, f)
                name = f[:-3]
                try:
                    spec = importlib.util.spec_from_file_location(name, path)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    if hasattr(mod, "on_message"):
                        SKILLS_MODULES.append(mod)
                        loaded_names.append(name)
                        print(f"[Skills] Loaded {name}")
                except Exception as e:
                    print(f"[Skills] Failed to load {name}: {e}")
    
    SYSTEM_STATUS["active_skills"] = loaded_names
    return loaded_names

def dispatch_message(packet):
    # Normalized Chat Object
    try:
        json_payload = packet.get('json', {})
        chat = {
            'room': {'id': json_payload.get('chat_id'), 'name': packet.get('room')},
            'sender': {'id': json_payload.get('user_id'), 'name': packet.get('sender')},
            'message': {'content': packet.get('msg')},
            'raw': packet
        }
        
        reply_func = lambda text: requests.post("http://localhost:3000/reply", json={
            "type": "text", 
            "room": chat['room']['id'], 
            "data": text
        })

        for mod in SKILLS_MODULES:
            try:
                mod.on_message(chat, reply_func)
            except Exception as e:
                print(f"[Skills] Error in {mod}: {e}")
                
    except Exception as e:
        print(f"[Dispatcher] Error: {e}")

# --- Control API ---

@app.route('/api/status')
def get_status():
    return jsonify(SYSTEM_STATUS)

@app.route('/api/control/start_android', methods=['POST'])
def api_start_android():
    threading.Thread(target=lifecycle_start).start()
    return jsonify({"status": "starting"})

@app.route('/api/control/reload_skills', methods=['POST'])
def api_reload_skills():
    load_skills()
    return jsonify({"status": "reloaded", "skills": SYSTEM_STATUS["active_skills"]})

# Dashboard Serving
@app.route('/')
def serve_dashboard():
    return send_from_directory('../dashboard', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../dashboard', path)


def lifecycle_start():
    global bridge
    SYSTEM_STATUS["android"] = "connecting"
    
    # 1. Connect ADB
    adb.fast_connect()
    adb.wait_for_device()
    SYSTEM_STATUS["android"] = "connected"
    
    # 2. Install Iris (Assuming APK is in root or known location)
    # For now, we assume it's pre-installed or user put it in 'env/apk/'
    # adb.install_apk_if_needed("env/apk/Iris.apk", "party.qwer.iris")
    
    # 3. Start App & Port Forward
    adb.setup_port_forward(3000, 3000)
    if adb.start_iris_service():
        SYSTEM_STATUS["android"] = "running_app"
    
    # 4. Connect Bridge
    if bridge:
        bridge.stop()
    
    bridge = IrisBridge(on_message=dispatch_message)
    bridge.start()
    SYSTEM_STATUS["iris_bridge"] = "connected"

if __name__ == '__main__':
    print("Stend Core v1.0 Starting...")
    load_skills()
    # Auto-start lifecycle for convenience
    threading.Thread(target=lifecycle_start).start()
    app.run(host='0.0.0.0', port=5000)
