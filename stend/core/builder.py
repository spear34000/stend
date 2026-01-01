import os
import subprocess
import shutil

class AndroidBuilder:
    def __init__(self, project_root):
        self.project_root = project_root
        self.gradlew = os.path.join(project_root, "gradlew.bat" if os.name == 'nt' else "./gradlew")

    def build_debug(self):
        print("[Builder] Building Android App (Debug)...")
        # Ensure executable
        if os.name != 'nt':
            os.chmod(self.gradlew, 0o755)
            
        try:
            subprocess.check_call([self.gradlew, "assembleDebug"], cwd=self.project_root)
            print("[Builder] Build Successful.")
            return self.find_apk("debug")
        except subprocess.CalledProcessError as e:
            print(f"[Builder] Build Failed: {e}")
            return None

    def find_apk(self, variant="debug"):
        # Locates the produced APK
        apk_path = os.path.join(self.project_root, "app", "build", "outputs", "apk", variant, f"app-{variant}.apk")
        if os.path.exists(apk_path):
            return apk_path
        return None

if __name__ == "__main__":
    builder = AndroidBuilder(os.path.abspath("../android_project"))
    apk = builder.build_debug()
    if apk:
        print(f"APK created at: {apk}")
