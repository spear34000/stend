"""
Gemini 2.0 Flash API를 사용한 AI 에이전트의 기본 구조
"""
import os
from typing import Dict, List, Any, Optional
from google import genai
from dotenv import load_dotenv

# 설정 파일 로드
from config import (
    GEMINI_API_KEY,
    MODEL_NAME,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K
)

class GeminiAgent:
    """
    Gemini 2.0 Flash API를 사용하는 AI 에이전트 클래스
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        GeminiAgent 초기화
        
        Args:
            api_key: Gemini API 키 (None인 경우 환경 변수에서 로드)
        """
        # API 키 설정
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        
        # Gemini 클라이언트 초기화
        self.client = genai.Client(api_key=self.api_key)
        
        # 모델 설정
        self.model_name = MODEL_NAME
        self.model = self.client.models.get(self.model_name)
        
        # 대화 기록 초기화
        self.conversation_history = []
    
    def generate_response(self, 
                         prompt: str, 
                         max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
                         temperature: float = DEFAULT_TEMPERATURE,
                         top_p: float = DEFAULT_TOP_P,
                         top_k: int = DEFAULT_TOP_K) -> str:
        """
        사용자 입력에 대한 응답 생성
        
        Args:
            prompt: 사용자 입력 텍스트
            max_output_tokens: 최대 출력 토큰 수
            temperature: 온도 (높을수록 더 창의적인 응답)
            top_p: 상위 확률 임계값
            top_k: 상위 k개 토큰 선택
            
        Returns:
            생성된 응답 텍스트
        """
        # 대화 기록에 사용자 입력 추가
        self.conversation_history.append({"role": "user", "content": prompt})
        
        try:
            # 생성 설정
            generation_config = {
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            }
            
            # 대화 기록을 Gemini API 형식으로 변환
            contents = self._prepare_conversation_for_api()
            
            # 응답 생성
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                generation_config=generation_config
            )
            
            # 응답 텍스트 추출
            response_text = response.text
            
            # 대화 기록에 AI 응답 추가
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            return response_text
            
        except Exception as e:
            error_message = f"응답 생성 중 오류 발생: {str(e)}"
            print(error_message)
            return error_message
    
    def _prepare_conversation_for_api(self) -> List[Dict[str, Any]]:
        """
        대화 기록을 Gemini API 형식으로 변환
        
        Returns:
            Gemini API 형식의 대화 기록
        """
        # Gemini API 형식으로 변환된 대화 기록
        formatted_conversation = []
        
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
    
    def clear_conversation(self):
        """대화 기록 초기화"""
        self.conversation_history = []
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        현재 대화 기록 반환
        
        Returns:
            대화 기록 목록
        """
        return self.conversation_history
