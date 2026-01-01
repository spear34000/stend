import requests
import json
import base64
import time
import uuid
from urllib.parse import quote

class KakaoLinkManager:
    """
    Port of IrisLink.ts (Legacy KakaoLink v2 implementation)
    Allows sending template-based KakaoLink messages via sharer.kakao.com
    """
    KAKAOTALK_VERSION = '25.2.1'
    ANDROID_SDK_VER = 33
    ANDROID_WEBVIEW_UA = 'Mozilla/5.0 (Linux; Android 13; SM-G998B Build/TP1A.220624.014; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/114.0.5735.60 Mobile Safari/537.36'

    def __init__(self, iris_url="http://localhost:3000"):
        self.iris_url = iris_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': f"{self.ANDROID_WEBVIEW_UA} KAKAOTALK/{self.KAKAOTALK_VERSION} (INAPP)",
            'X-Requested-With': 'com.kakao.talk'
        })

    def _get_ka(self, origin):
        return f"sdk/1.43.5 os/javascript sdk_type/javascript lang/ko-KR device/Linux armv7l origin/{quote(origin)}"

    def _get_authorization(self):
        try:
            r = requests.get(f"{self.iris_url}/aot", timeout=5)
            aot = r.json().get("aot", {})
            return f"{aot.get('access_token')}-{aot.get('d_id')}"
        except Exception as e:
            print(f"[LinkManager] Failed to get Auth: {e}")
            return None

    def send(self, receiver_name, template_id, template_args, app_key, origin):
        """
        Sends a KakaoLink v2 message using the legacy sharer logic.
        """
        auth = self._get_authorization()
        if not auth:
            return {"success": False, "error": "Authorization failed"}

        ka = self._get_ka(origin)
        
        try:
            # 1. Validation (custom template)
            res = self.session.post(
                'https://sharer.kakao.com/picker/link',
                data={
                    'app_key': app_key,
                    'ka': ka,
                    'validation_action': 'custom',
                    'validation_params': json.dumps({
                        'link_ver': '4.0',
                        'template_id': template_id,
                        'template_args': template_args
                    })
                },
                allow_redirects=True
            )

            # 2. Extract server data (Base64 URL encoded)
            import re
            match = re.search(r'window\.serverData = "([^"]+)"', res.text)
            if not match:
                # Might need login logic here if session expired, 
                # but legacy usually passed if AOT was fresh.
                return {"success": False, "error": "Server data not found (Login required?)"}

            # Decode server data to find receivers
            decoded = base64.urlsafe_b64decode(match.group(1) + "===").decode('utf-8')
            server_data = json.loads(decoded)["data"]
            
            # 3. Find receiver
            receivers = server_data.get("chats", []) + server_data.get("friends", [])
            target = None
            for r in receivers:
                name = r.get("title") or r.get("profile_nickname") or ""
                if receiver_name in name:
                    target = r
                    break
            
            if not target:
                return {"success": False, "error": f"Receiver '{receiver_name}' not found"}

            # 4. Send
            send_res = self.session.post(
                'https://sharer.kakao.com/picker/send',
                data={
                    'app_key': app_key,
                    'short_key': server_data['shortKey'],
                    'checksum': server_data['checksum'],
                    '_csrf': server_data['csrfToken'],
                    'receiver': base64.urlsafe_b64encode(json.dumps(target).encode()).decode().strip('=')
                }
            )

            if send_res.status_code == 200:
                return {"success": True}
            else:
                return {"success": False, "error": f"Send failed: {send_res.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}
