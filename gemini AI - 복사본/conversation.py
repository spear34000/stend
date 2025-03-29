"""
Gemini 2.0 Flash API를 사용한 대화 처리 로직 구현
"""
from typing import Dict, List, Any, Optional, Union
import uuid

from .api_client import GeminiAPIClient
from .request_handler import RequestHandler

class ConversationManager:
    """
    대화 컨텍스트를 관리하는 클래스
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        ConversationManager 초기화
        
        Args:
            api_key: Gemini API 키 (None인 경우 환경 변수에서 로드)
        """
        # 요청 핸들러 초기화
        self.request_handler = RequestHandler(api_key)
        
        # 대화 상태 초기화
        self.conversation_id = str(uuid.uuid4())
        self.conversation_history = []
        self.system_prompt = None
        self.max_history_length = 10  # 기본 대화 기록 길이 제한
    
    def set_system_prompt(self, system_prompt: str):
        """
        시스템 프롬프트 설정
        
        Args:
            system_prompt: 시스템 프롬프트 텍스트
        """
        self.system_prompt = system_prompt
    
    def add_user_message(self, message: str) -> Dict[str, Any]:
        """
        사용자 메시지 추가 및 응답 생성
        
        Args:
            message: 사용자 메시지 텍스트
            
        Returns:
            생성된 응답 결과
        """
        # 사용자 메시지 추가
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # 대화 기록 길이 제한 적용
        self._trim_conversation_history()
        
        # API 요청용 대화 기록 준비
        contents = self._prepare_conversation_for_api()
        
        # API 요청 및 응답 생성
        response = self.request_handler.process_request(contents)
        
        # 성공적인 응답인 경우 대화 기록에 추가
        if response.get("success", False):
            self.conversation_history.append({
                "role": "assistant",
                "content": response.get("text", "")
            })
        
        return response
    
    def _prepare_conversation_for_api(self) -> List[Dict[str, Any]]:
        """
        대화 기록을 Gemini API 형식으로 변환
        
        Returns:
            Gemini API 형식의 대화 기록
        """
        # Gemini API 형식으로 변환된 대화 기록
        formatted_conversation = []
        
        # 시스템 프롬프트가 있는 경우 추가
        if self.system_prompt:
            formatted_conversation.append({
                "role": "user",
                "parts": [{"text": self.system_prompt}]
            })
            
            # 시스템 프롬프트에 대한 가상 응답 추가
            formatted_conversation.append({
                "role": "model",
                "parts": [{"text": "이해했습니다. 지시에 따라 도와드리겠습니다."}]
            })
        
        # 대화 기록 추가
        for message in self.conversation_history:
            role = message["role"]
            content = message["content"]
            
            # Gemini API는 'user'와 'model' 역할을 사용
            api_role = "user" if role == "user" else "model"
            
            formatted_conversation.append({
                "role": api_role,
                "parts": [{"text": content}]
            })
        
        return formatted_conversation
    
    def _trim_conversation_history(self):
        """
        대화 기록 길이 제한 적용
        """
        if len(self.conversation_history) > self.max_history_length * 2:
            # 시스템 메시지를 제외한 가장 오래된 사용자-AI 대화 쌍 제거
            self.conversation_history = self.conversation_history[-self.max_history_length * 2:]
    
    def clear_conversation(self):
        """대화 기록 초기화"""
        self.conversation_history = []
        self.conversation_id = str(uuid.uuid4())
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        현재 대화 기록 반환
        
        Returns:
            대화 기록 목록
        """
        return self.conversation_history
    
    def set_max_history_length(self, max_length: int):
        """
        대화 기록 최대 길이 설정
        
        Args:
            max_length: 최대 대화 쌍 수
        """
        if max_length < 1:
            raise ValueError("대화 기록 길이는 1 이상이어야 합니다.")
        
        self.max_history_length = max_length
        self._trim_conversation_history()
