"""
Gemini 2.0 Flash API를 사용한 사용자 입력 처리 로직 구현
"""
from typing import Dict, List, Any, Optional, Union
import re

class InputProcessor:
    """
    사용자 입력을 처리하는 클래스
    """
    
    def __init__(self):
        """InputProcessor 초기화"""
        # 명령어 패턴 정의
        self.command_patterns = {
            "reset": r"^(초기화|리셋|reset|clear)$",
            "help": r"^(도움말|도움|help)$",
            "exit": r"^(종료|나가기|exit|quit)$"
        }
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        사용자 입력 처리 및 분석
        
        Args:
            user_input: 사용자 입력 텍스트
            
        Returns:
            처리된 입력 정보
        """
        # 입력 전처리
        cleaned_input = self._preprocess_input(user_input)
        
        # 빈 입력 처리
        if not cleaned_input:
            return {
                "type": "empty",
                "content": "",
                "is_command": False
            }
        
        # 명령어 확인
        command = self._check_command(cleaned_input)
        if command:
            return {
                "type": "command",
                "command": command,
                "content": cleaned_input,
                "is_command": True
            }
        
        # 일반 메시지 처리
        return {
            "type": "message",
            "content": cleaned_input,
            "is_command": False
        }
    
    def _preprocess_input(self, user_input: str) -> str:
        """
        사용자 입력 전처리
        
        Args:
            user_input: 원본 사용자 입력
            
        Returns:
            전처리된 입력 텍스트
        """
        # 공백 제거
        cleaned_input = user_input.strip()
        
        # 특수 문자 처리 등 필요한 전처리 수행
        # (현재는 간단한 공백 제거만 수행)
        
        return cleaned_input
    
    def _check_command(self, input_text: str) -> Optional[str]:
        """
        입력이 명령어인지 확인
        
        Args:
            input_text: 확인할 입력 텍스트
            
        Returns:
            명령어 유형 또는 None
        """
        for command, pattern in self.command_patterns.items():
            if re.match(pattern, input_text, re.IGNORECASE):
                return command
        
        return None
    
    def extract_parameters(self, input_text: str) -> Dict[str, Any]:
        """
        입력에서 매개변수 추출
        
        Args:
            input_text: 매개변수를 추출할 입력 텍스트
            
        Returns:
            추출된 매개변수
        """
        # 기본 매개변수
        params = {}
        
        # 온도 매개변수 추출 (예: "온도:0.8")
        temp_match = re.search(r'온도[:\s]*([\d.]+)', input_text)
        if temp_match:
            try:
                params["temperature"] = float(temp_match.group(1))
            except ValueError:
                pass
        
        # 최대 토큰 수 추출 (예: "최대토큰:100")
        token_match = re.search(r'최대[토큰\s]*[:\s]*(\d+)', input_text)
        if token_match:
            try:
                params["max_tokens"] = int(token_match.group(1))
            except ValueError:
                pass
        
        return params
