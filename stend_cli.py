#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import time
import requests

# --- Configuration ---
ADB_TARGET = "localhost:5555"
ANDROID_PROJECT_DIR = os.path.join("stend", "android_project")
DOCKER_ENV_DIR = os.path.join("stend", "env")
API_SERVER_SCRIPT = os.path.join("stend", "run.py")
APK_PATH = os.path.join(ANDROID_PROJECT_DIR, "app/build/outputs/apk/debug/app-debug.apk")
TARGET_APK_REMOTE = "/data/local/tmp/Stend.apk"
MAIN_CLASS = "party.qwer.iris.Main"

def run_cmd(cmd, cwd=None, shell=False, check=True):
    print(f"[CMD] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        return subprocess.run(cmd, cwd=cwd, shell=shell, check=check, text=True, capture_output=True).stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        if check: sys.exit(1)
        return None

def build():
    print("\n[1/3] Building Android Subsystem (Gradle)...")
    gradle_cmd = "gradlew.bat" if os.name == "nt" else "./gradlew"
    run_cmd([gradle_cmd, "assembleDebug"], cwd=ANDROID_PROJECT_DIR, shell=(os.name == "nt"))
    
    if not os.path.exists(APK_PATH):
        print(f"[ERROR] APK not found at {APK_PATH}")
        sys.exit(1)
    print(f"[SUCCESS] APK Built: {APK_PATH}")

def deploy():
    print("\n[2/3] Deploying Subsystem to Redroid...")
    # Ensure connected
    subprocess.run(["adb", "disconnect"], capture_output=True)
    subprocess.run(["adb", "connect", ADB_TARGET], capture_output=True)
    
    print(f"Pushing APK to {TARGET_APK_REMOTE}...")
    run_cmd(["adb", "push", APK_PATH, TARGET_APK_REMOTE])
    
    print("Launching Engine via app_process...")
    # Kill previous
    run_cmd(["adb", "shell", f"pkill -f {MAIN_CLASS}"], check=False)
    
    # Run in background via nohup-like shell command
    launch_cmd = f"export CLASSPATH={TARGET_APK_REMOTE}; app_process /system/bin {MAIN_CLASS} > /data/local/tmp/stend_engine.log 2>&1 &"
    run_cmd(["adb", "shell", launch_cmd])
    
    print("[SUCCESS] Engine Deployed and Running.")

def start_api():
    print("\n[3/3] Launching Stend API Server...")
    # Non-blocking execution of the API server
    try:
        subprocess.Popen([sys.executable, API_SERVER_SCRIPT])
        print("[SUCCESS] API Server Started at http://localhost:5001")
        print("Tip: Use 'python stend_cli.py stop' to shut down everything.")
    except Exception as e:
        print(f"[ERROR] Failed to start API: {e}")

def stop():
    print("\nStopping Stend Platform...")
    # 1. Kill Python processes (run.py/api_server)
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/IM", "python.exe", "/FI", f"WINDOWTITLE eq Stend*"], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", "stend/run.py"], capture_output=True)
    
    # 2. Kill Android process
    subprocess.run(["adb", "shell", f"pkill -f {MAIN_CLASS}"], capture_output=True)
    
    # 3. Docker down
    run_cmd(["docker-compose", "down"], cwd=DOCKER_ENV_DIR)
    print("[SUCCESS] All services stopped.")

def main():
    parser = argparse.ArgumentParser(description="Stend Platform Unified CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    subparsers.add_parser("start", help="Full start (Docker -> Build -> Deploy -> API)")
    subparsers.add_parser("build", help="Build the Android project")
    subparsers.add_parser("deploy", help="Push and run the APK in redroid")
    subparsers.add_parser("stop", help="Stop all services")
    subparsers.add_parser("api", help="Start only the API server")
    
    # Admin/Ban proxies
    parser_admin = subparsers.add_parser("admin", help="Manage admins")
    parser_admin.add_argument("action", choices=["add", "del", "list"])
    parser_admin.add_argument("user_id", nargs="?", type=int)

    args = parser.parse_args()

    if args.command == "start":
        print("Launching Stend Infrastructure (Docker-Compose)...")
        run_cmd(["docker-compose", "up", "-d"], cwd=DOCKER_ENV_DIR)
        time.sleep(5)
        build()
        deploy()
        start_api()
        print("\nStend is READY.")
        
    elif args.command == "build":
        build()
        
    elif args.command == "deploy":
        deploy()
        
    elif args.command == "stop":
        stop()
        
    elif args.command == "api":
        start_api()
        
    elif args.command == "admin":
        # Proxy to Store API for admin management
        try:
            if args.action == "list":
                r = requests.get("http://localhost:5001/api/store/get?key=admins")
                val = r.json().get("value") or []
                print(f"Current Admins: {val}")
            elif args.action == "add":
                if not args.user_id: 
                    print("Error: user_id is required")
                    return
                r = requests.get("http://localhost:5001/api/store/get?key=admins")
                val = r.json().get("value") or []
                if args.user_id not in val:
                    val.append(args.user_id)
                    requests.post("http://localhost:5001/api/store/put", json={"key": "admins", "value": val})
                print(f"Admin Added: {args.user_id}")
            elif args.action == "del":
                if not args.user_id: 
                    print("Error: user_id is required")
                    return
                r = requests.get("http://localhost:5001/api/store/get?key=admins")
                val = r.json().get("value") or []
                if args.user_id in val:
                    val.remove(args.user_id)
                    requests.post("http://localhost:5001/api/store/put", json={"key": "admins", "value": val})
                print(f"Admin Deleted: {args.user_id}")
        except Exception as e:
            print(f"Error communicating with API: {e}. Is the server running?")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
