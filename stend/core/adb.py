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

    def start_iris_service(self):
        # Based on previous analysis of Iris, we need to start the service.
        # Since we are automating, we use the raw AM command.
        print("Starting Iris Service...")
        # Assuming the package is party.qwer.iris and Main component
        # Using monkey to launch app is often more reliable for hidden apps
        self.run(["shell", "monkey", "-p", "party.qwer.iris", "-c", "android.intent.category.LAUNCHER", "1"])
        
        # Or explicit service start if we knew the service name.
        # Let's try to finding the pid to confirm
        time.sleep(3)
        pid = self.run(["shell", "pidof", "party.qwer.iris"])
        if pid:
            print(f"Iris is running (PID: {pid})")
            return True
        return False

    def setup_port_forward(self, local=3000, remote=3000):
        print(f"Forwarding tcp:{local} -> tcp:{remote}")
        self.run(["forward", f"tcp:{local}", f"tcp:{remote}"])
