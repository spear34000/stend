"""
Gemini 2.0 Flash API를 사용한 응답 생성 기능 구현
"""
from typing import Dict, List, Any, Optional, Union
import json
import time

from .api_client import GeminiAPIClient
from .request_handler import RequestHandler
from .conversation import ConversationManager

class ResponseGenerator:
    """
    응답 생성 및 처리를 담당하는 클래스
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        ResponseGenerator 초기화
        
        Args:
            api_key: Gemini API 키 (None인 경우 환경 변수에서 로드)
        """
        # 대화 관리자 초기화
        self.conversation_manager = ConversationManager(api_key)
        
        # 기본 시스템 프롬프트 설정
        self._set_default_system_prompt()
        
        # 응답 생성 설정
        self.default_generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048
        }
        
        # 응답 캐시
        self.response_cache = {}
        self.cache_ttl = 3600  # 캐시 유효 시간 (초)
    
    def _set_default_system_prompt(self):
        """기본 시스템 프롬프트 설정"""
        system_prompt = """
        당신은 사용자가 말한 대로 만들어주는 AI 어시스턴트입니다.
        사용자의 요청을 정확히 이해하고 그에 맞는 결과물을 생성해주세요.
        사용자가 요청한 내용이 불분명하면 추가 정보를 요청하세요.
        항상 친절하고 도움이 되는 응답을 제공하세요.
        """
        self.conversation_manager.set_system_prompt(system_prompt)
    
    def generate_response(self, 
                         user_input: str, 
                         generation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        사용자 입력에 대한 응답 생성
        
        Args:
            user_input: 사용자 입력 텍스트
            generation_config: 응답 생성 설정
            
        Returns:
            생성된 응답 결과
        """
        # 캐시 확인
        cache_key = self._generate_cache_key(user_input, generation_config)
        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            return cached_response
        
        # 생성 설정 준비
        config = self.default_generation_config.copy()
        if generation_config:
            config.update(generation_config)
        
        # 응답 생성
        start_time = time.time()
        response = self.conversation_manager.add_user_message(user_input)
        generation_time = time.time() - start_time
        
        # 응답 결과 구성
        result = {
            "success": response.get("success", False),
            "text": response.get("text", ""),
            "generation_time": generation_time,
            "timestamp": time.time()
        }
        
        # 오류 처리
        if not result["success"]:
            result["error"] = response.get("error", "알 수 없는 오류가 발생했습니다.")
            return self._handle_error(result)
        
        # 캐시에 저장
        self._add_to_cache(cache_key, result)
        
        return result
    
    def _generate_cache_key(self, 
                          user_input: str, 
                          generation_config: Optional[Dict[str, Any]] = None) -> str:
        """
        캐시 키 생성
        
        Args:
            user_input: 사용자 입력 텍스트
            generation_config: 응답 생성 설정
            
        Returns:
            캐시 키
        """
        # 대화 기록의 마지막 몇 개 메시지만 포함
        history = self.conversation_manager.get_conversation_history()[-3:] if self.conversation_manager.get_conversation_history() else []
        
        # 캐시 키 구성 요소
        key_components = {
            "input": user_input,
            "history": history,
            "config": generation_config or {}
        }
        
        # JSON 문자열로 변환하여 캐시 키 생성
        return json.dumps(key_components, sort_keys=True)
    
    def _add_to_cache(self, cache_key: str, response: Dict[str, Any]):
        """
        응답을 캐시에 추가
        
        Args:
            cache_key: 캐시 키
            response: 캐시할 응답
        """
        self.response_cache[cache_key] = {
            "response": response,
            "timestamp": time.time()
        }
        
        # 캐시 크기 제한
        if len(self.response_cache) > 100:
            self._clean_cache()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        캐시에서 응답 조회
        
        Args:
            cache_key: 캐시 키
            
        Returns:
            캐시된 응답 또는 None
        """
        cached_item = self.response_cache.get(cache_key)
        
        if not cached_item:
            return None
        
        # 캐시 유효 시간 확인
        if time.time() - cached_item["timestamp"] > self.cache_ttl:
            del self.response_cache[cache_key]
            return None
        
        return cached_item["response"]
    
    def _clean_cache(self):
        """오래된 캐시 항목 정리"""
        current_time = time.time()
        keys_to_delete = []
        
        for key, item in self.response_cache.items():
            if current_time - item["timestamp"] > self.cache_ttl:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.response_cache[key]
    
    def _handle_error(self, error_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        오류 처리 및 사용자 친화적 메시지 생성
        
        Args:
            error_result: 오류 정보
            
        Returns:
            처리된 오류 응답
        """
        error_message = error_result.get("error", "알 수 없는 오류가 발생했습니다.")
        
        # 오류 유형에 따른 사용자 친화적 메시지
        if "rate limit" in error_message.lower():
            user_message = "요청이 너무 많습니다. 잠시 후 다시 시도해 주세요."
        elif "invalid api key" in error_message.lower():
            user_message = "API 키 오류가 발생했습니다. API 키 설정을 확인해 주세요."
        elif "content filtered" in error_message.lower():
            user_message = "요청하신 내용은 콘텐츠 정책에 따라 처리할 수 없습니다."
        else:
            user_message = "응답 생성 중 오류가 발생했습니다. 다시 시도해 주세요."
        
        error_result["text"] = user_message
        return error_result
    
    def clear_conversation(self):
        """대화 초기화"""
        self.conversation_manager.clear_conversation()
        self._set_default_system_prompt()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        현재 대화 기록 반환
        
        Returns:
            대화 기록 목록
        """
        return self.conversation_manager.get_conversation_history()
