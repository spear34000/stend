"""
Gemini 2.0 Flash API 클라이언트 구현
"""
import os
from typing import Dict, List, Any, Optional, Union
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

class GeminiAPIClient:
    """
    Gemini 2.0 Flash API와 통신하는 클라이언트 클래스
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        GeminiAPIClient 초기화
        
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
    
    def generate_content(self, 
                        contents: Union[str, List[Dict[str, Any]]],
                        generation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Gemini API를 사용하여 콘텐츠 생성
        
        Args:
            contents: 생성 요청 내용 (문자열 또는 구조화된 대화)
            generation_config: 생성 설정
            
        Returns:
            API 응답 결과
        """
        try:
            # 기본 생성 설정
            default_config = {
                "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
                "temperature": DEFAULT_TEMPERATURE,
                "top_p": DEFAULT_TOP_P,
                "top_k": DEFAULT_TOP_K
            }
            
            # 사용자 설정이 있으면 기본 설정 업데이트
            if generation_config:
                default_config.update(generation_config)
            
            # 문자열 입력을 처리
            if isinstance(contents, str):
                contents = [{"role": "user", "parts": [{"text": contents}]}]
            
            # API 요청 및 응답 반환
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                generation_config=default_config
            )
            
            return {
                "success": True,
                "text": response.text,
                "response": response
            }
            
        except Exception as e:
            error_message = f"콘텐츠 생성 중 오류 발생: {str(e)}"
            print(error_message)
            return {
                "success": False,
                "error": error_message,
                "text": error_message
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        현재 사용 중인 모델 정보 반환
        
        Returns:
            모델 정보
        """
        try:
            model_info = self.model.get_info()
            return {
                "success": True,
                "model_name": self.model_name,
                "model_info": model_info
            }
        except Exception as e:
            error_message = f"모델 정보 조회 중 오류 발생: {str(e)}"
            print(error_message)
            return {
                "success": False,
                "error": error_message
            }
    
    def validate_connection(self) -> bool:
        """
        API 연결 상태 확인
        
        Returns:
            연결 성공 여부
        """
        try:
            # 간단한 요청으로 연결 확인
            response = self.generate_content("Hello, Gemini!")
            return response.get("success", False)
        except Exception:
            return False
