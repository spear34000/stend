"""
Gemini 2.0 Flash API를 사용한 요청 처리 로직 구현
"""
from typing import Dict, List, Any, Optional, Union
import time

from .api_client import GeminiAPIClient

class RequestHandler:
    """
    Gemini API 요청을 처리하는 핸들러 클래스
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        RequestHandler 초기화
        
        Args:
            api_key: Gemini API 키 (None인 경우 환경 변수에서 로드)
        """
        # API 클라이언트 초기화
        self.api_client = GeminiAPIClient(api_key)
        
        # 요청 처리 상태 초기화
        self.last_request_time = 0
        self.request_count = 0
        self.error_count = 0
        self.retry_delay = 1  # 초 단위 재시도 지연 시간
        self.max_retries = 3  # 최대 재시도 횟수
    
    def process_request(self, 
                       contents: Union[str, List[Dict[str, Any]]],
                       generation_config: Optional[Dict[str, Any]] = None,
                       retry_on_error: bool = True) -> Dict[str, Any]:
        """
        Gemini API 요청 처리 및 응답 반환
        
        Args:
            contents: 생성 요청 내용 (문자열 또는 구조화된 대화)
            generation_config: 생성 설정
            retry_on_error: 오류 발생 시 재시도 여부
            
        Returns:
            처리된 응답 결과
        """
        # 요청 간 최소 지연 시간 적용 (비율 제한 준수)
        self._apply_rate_limiting()
        
        # 요청 카운터 증가
        self.request_count += 1
        
        # 재시도 카운터 초기화
        retries = 0
        
        while retries <= self.max_retries:
            # API 요청 실행
            response = self.api_client.generate_content(contents, generation_config)
            
            # 성공 시 응답 반환
            if response.get("success", False):
                return response
            
            # 실패 시 처리
            self.error_count += 1
            
            # 재시도 여부 확인
            if not retry_on_error or retries >= self.max_retries:
                return response
            
            # 재시도 지연 및 카운터 증가
            time.sleep(self.retry_delay * (retries + 1))
            retries += 1
        
        # 모든 재시도 실패 시
        return {
            "success": False,
            "error": "최대 재시도 횟수 초과",
            "text": "요청 처리 중 오류가 발생했습니다. 나중에 다시 시도해 주세요."
        }
    
    def _apply_rate_limiting(self):
        """
        비율 제한 준수를 위한 요청 간 지연 적용
        """
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # 요청 간 최소 간격 (초) - Gemini 2.0 Flash는 분당 15 요청 (약 4초당 1개)
        min_interval = 4.0
        
        # 필요한 경우 지연 적용
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        # 마지막 요청 시간 업데이트
        self.last_request_time = time.time()
    
    def get_request_stats(self) -> Dict[str, Any]:
        """
        요청 처리 통계 반환
        
        Returns:
            요청 처리 통계 정보
        """
        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "success_rate": (self.request_count - self.error_count) / max(1, self.request_count) * 100
        }
