import subprocess
import os
import time

class AdbManager:
    def __init__(self, adb_path="adb", target="127.0.0.1:5555"):
        self.adb_path = adb_path
        self.target = target

    def run(self, args):
        # Always target the specific device to avoid "more than one device" error
        cmd = [self.adb_path, "-s", self.target] + args
        try:
            return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8').strip()
        except subprocess.CalledProcessError as e:
            err_msg = e.output.decode('utf-8')
            # Ignore errors that are expected during startup/polling
            if "device offline" not in err_msg and "not found" not in err_msg:
                print(f"[ADB Error] {err_msg}")
            return None

    def fast_connect(self):
        # Explicit bypass of run() for 'connect' as it doesn't take -s
        subprocess.run([self.adb_path, "connect", self.target], capture_output=True)
    
    def wait_for_device(self, timeout=10):
        print(f"Waiting for device (timeout {timeout}s)...")
        # In windows, wait-for-device can block. Let's use a simpler check.
        start = time.time()
        while time.time() - start < timeout:
            res = self.run(["get-state"])
            if res == "device":
                print("Device connected.")
                return True
            time.sleep(1)
        print("Device connection timed out.")
        return False

    def install_apk_if_needed(self, apk_path, package_name):
        # Check if installed
        packages = self.run(["shell", "pm", "list", "packages", package_name])
        if packages and package_name in packages:
            print(f"{package_name} is already installed.")
        else:
            print(f"Installing {apk_path}...")
            self.run(["install", "-r", apk_path])

    def push(self, local, remote):
        return self.run(["push", local, remote])

    def start_iris_process(self, apk_path):
        """Ultra-fast startup using app_process (Core Stend technology)"""
        print(f"Deploying Stend Subsystem: {apk_path} ...")
        self.push(apk_path, "/data/local/tmp/Stend.apk")
        
        print("Initializing Android Orchestrator...")
        # Kil existing if any
        self.run(["shell", "pkill -f party.qwer.iris.Main"])
        
        # Start via app_process
        cmd = "export CLASSPATH=/data/local/tmp/Stend.apk; app_process /system/bin party.qwer.iris.Main > /data/local/tmp/stend_log.txt 2>&1 &"
        self.run(["shell", cmd])
        
        time.sleep(2)
        pid = self.run(["shell", "pidof", "party.qwer.iris"])
        if pid:
            print(f"Subsystem ready (PID: {pid})")
            return True
        return False

    def setup_port_forward(self, local=3000, remote=3000):
        print(f"Forwarding tcp:{local} -> tcp:{remote}")
        self.run(["forward", f"tcp:{local}", f"tcp:{remote}"])
