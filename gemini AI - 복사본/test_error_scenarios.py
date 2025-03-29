"""
Gemini 2.0 Flash API를 사용한 AI 에이전트 오류 시나리오 테스트
"""
import unittest
import os
import sys
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 환경 변수 로드
load_dotenv()

from src.api_client import GeminiAPIClient
from src.request_handler import RequestHandler
from src.conversation import ConversationManager
from src.response_generator import ResponseGenerator
from src.error_handler import ErrorHandler

class TestErrorScenarios(unittest.TestCase):
    """
    오류 시나리오 테스트 클래스
    """
    
    def setUp(self):
        """테스트 설정"""
        # API 키 확인
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key == "YOUR_API_KEY":
            self.skipTest("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        
        # 테스트 컴포넌트 초기화
        self.api_client = GeminiAPIClient(self.api_key)
        self.request_handler = RequestHandler(self.api_key)
        self.conversation_manager = ConversationManager(self.api_key)
        self.response_generator = ResponseGenerator(self.api_key)
        self.error_handler = ErrorHandler()
    
    @patch('src.api_client.genai.Client')
    def test_invalid_api_key(self, mock_client):
        """잘못된 API 키 테스트"""
        # API 키 오류 시뮬레이션
        mock_client.side_effect = Exception("Invalid API key")
        
        # 잘못된 API 키로 클라이언트 초기화 시도
        with self.assertRaises(Exception):
            client = GeminiAPIClient("INVALID_API_KEY")
    
    @patch('src.api_client.GeminiAPIClient.generate_content')
    def test_rate_limit_error(self, mock_generate):
        """비율 제한 오류 테스트"""
        # 비율 제한 오류 시뮬레이션
        mock_generate.return_value = {
            "success": False,
            "error": "Rate limit exceeded",
            "text": "요청이 너무 많습니다. 잠시 후 다시 시도해 주세요."
        }
        
        # 요청 처리
        response = self.request_handler.process_request("테스트 메시지")
        
        # 응답 확인
        self.assertFalse(response.get("success", True), "오류 응답이 성공으로 표시됨")
        self.assertIn("rate limit", response.get("error", "").lower(), "비율 제한 오류 메시지 불일치")
    
    @patch('src.api_client.GeminiAPIClient.generate_content')
    def test_content_filtering(self, mock_generate):
        """콘텐츠 필터링 테스트"""
        # 콘텐츠 필터링 오류 시뮬레이션
        mock_generate.return_value = {
            "success": False,
            "error": "Content filtered due to safety concerns",
            "text": "요청하신 내용은 콘텐츠 정책에 따라 처리할 수 없습니다."
        }
        
        # 응답 생성
        response = self.response_generator.generate_response("부적절한 콘텐츠")
        
        # 응답 확인
        self.assertFalse(response.get("success", True), "오류 응답이 성공으로 표시됨")
        self.assertIn("콘텐츠 정책", response.get("text", ""), "콘텐츠 필터링 오류 메시지 불일치")
    
    @patch('src.api_client.GeminiAPIClient.generate_content')
    def test_network_error(self, mock_generate):
        """네트워크 오류 테스트"""
        # 네트워크 오류 시뮬레이션
        mock_generate.side_effect = Exception("Connection error")
        
        # 오류 처리
        error = Exception("Connection error")
        error_response = self.error_handler.handle_error(error)
        
        # 응답 확인
        self.assertFalse(error_response.get("success", True), "오류 응답이 성공으로 표시됨")
        self.assertEqual(error_response.get("error_type"), "Exception", "오류 유형 불일치")
    
    @patch('src.request_handler.RequestHandler.process_request')
    def test_retry_mechanism(self, mock_process):
        """재시도 메커니즘 테스트"""
        # 첫 번째 호출에서 실패, 두 번째 호출에서 성공하도록 설정
        mock_process.side_effect = [
            {"success": False, "error": "Temporary error"},
            {"success": True, "text": "성공적인 응답"}
        ]
        
        # 요청 핸들러 설정
        self.request_handler.max_retries = 1
        self.request_handler.retry_delay = 0.1
        
        # 요청 처리
        response = self.request_handler.process_request("테스트 메시지", retry_on_error=True)
        
        # 응답 확인
        self.assertTrue(response.get("success", False), "재시도 후 성공 응답이 실패로 표시됨")
        self.assertEqual(response.get("text"), "성공적인 응답", "재시도 후 응답 텍스트 불일치")
    
    def test_empty_input_handling(self):
        """빈 입력 처리 테스트"""
        # 빈 입력으로 응답 생성
        response = self.response_generator.generate_response("")
        
        # 응답 확인
        self.assertFalse(response.get("success", True), "빈 입력에 대한 응답이 성공으로 표시됨")
    
    def test_long_input_handling(self):
        """긴 입력 처리 테스트"""
        # 매우 긴 입력 생성 (10,000자)
        long_input = "테스트 " * 2000
        
        # 긴 입력으로 응답 생성
        response = self.response_generator.generate_response(long_input)
        
        # 응답 확인 (성공 여부는 API 제한에 따라 달라질 수 있음)
        self.assertIsNotNone(response.get("text"), "긴 입력에 대한 응답 텍스트가 없습니다")

if __name__ == "__main__":
    unittest.main()
