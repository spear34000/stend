"""
Gemini 2.0 Flash API를 사용한 오류 처리 모듈
"""
from typing import Dict, Any, Optional
import logging
import traceback
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_agent_errors.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("gemini_agent")

class ErrorHandler:
    """
    오류 처리 및 로깅을 담당하는 클래스
    """
    
    def __init__(self):
        """ErrorHandler 초기화"""
        self.error_count = 0
        self.last_error_time = 0
        self.error_types = {}
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        오류 처리 및 로깅
        
        Args:
            error: 발생한 예외
            context: 오류 발생 컨텍스트 정보
            
        Returns:
            처리된 오류 정보
        """
        # 오류 정보 수집
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # 오류 통계 업데이트
        self.error_count += 1
        self.last_error_time = time.time()
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        
        # 오류 로깅
        logger.error(f"Error: {error_type} - {error_message}")
        if context:
            logger.error(f"Context: {context}")
        logger.debug(f"Stack trace: {stack_trace}")
        
        # 사용자 친화적 오류 메시지 생성
        user_message = self._generate_user_message(error_type, error_message)
        
        # 오류 응답 구성
        error_response = {
            "success": False,
            "error_type": error_type,
            "error": error_message,
            "text": user_message,
            "timestamp": self.last_error_time
        }
        
        return error_response
    
    def _generate_user_message(self, error_type: str, error_message: str) -> str:
        """
        사용자 친화적 오류 메시지 생성
        
        Args:
            error_type: 오류 유형
            error_message: 오류 메시지
            
        Returns:
            사용자 친화적 오류 메시지
        """
        # API 관련 오류
        if "API" in error_type or "api" in error_message.lower():
            if "key" in error_message.lower():
                return "API 키 오류가 발생했습니다. API 키 설정을 확인해 주세요."
            elif "rate" in error_message.lower() or "limit" in error_message.lower():
                return "요청이 너무 많습니다. 잠시 후 다시 시도해 주세요."
            else:
                return "API 연결 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
        
        # 네트워크 관련 오류
        elif "Connection" in error_type or "Timeout" in error_type:
            return "네트워크 연결 오류가 발생했습니다. 인터넷 연결을 확인하고 다시 시도해 주세요."
        
        # 콘텐츠 필터링 관련 오류
        elif "content" in error_message.lower() and "filter" in error_message.lower():
            return "요청하신 내용은 콘텐츠 정책에 따라 처리할 수 없습니다."
        
        # 기타 오류
        else:
            return "응답 생성 중 오류가 발생했습니다. 다시 시도해 주세요."
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        오류 통계 정보 반환
        
        Returns:
            오류 통계 정보
        """
        return {
            "total_errors": self.error_count,
            "last_error_time": self.last_error_time,
            "error_types": self.error_types
        }
