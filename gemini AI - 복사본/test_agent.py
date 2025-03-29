"""
Gemini 2.0 Flash API를 사용한 AI 에이전트 테스트 모듈
"""
import unittest
import os
import sys
import time
from dotenv import load_dotenv

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 환경 변수 로드
load_dotenv()

from src.api_client import GeminiAPIClient
from src.request_handler import RequestHandler
from src.conversation import ConversationManager
from src.input_processor import InputProcessor
from src.response_generator import ResponseGenerator
from src.error_handler import ErrorHandler

class TestGeminiAgent(unittest.TestCase):
    """
    Gemini 에이전트 테스트 클래스
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
        self.input_processor = InputProcessor()
        self.response_generator = ResponseGenerator(self.api_key)
        self.error_handler = ErrorHandler()
    
    def test_api_client_connection(self):
        """API 클라이언트 연결 테스트"""
        # 모델 정보 조회
        model_info = self.api_client.get_model_info()
        self.assertTrue(model_info.get("success", False), "API 클라이언트 연결 실패")
        self.assertIn("model_name", model_info, "모델 정보 조회 실패")
    
    def test_simple_content_generation(self):
        """간단한 콘텐츠 생성 테스트"""
        # 간단한 텍스트 생성
        response = self.api_client.generate_content("안녕하세요")
        self.assertTrue(response.get("success", False), "콘텐츠 생성 실패")
        self.assertIsNotNone(response.get("text"), "응답 텍스트가 없습니다")
        self.assertGreater(len(response.get("text", "")), 0, "응답 텍스트가 비어 있습니다")
    
    def test_request_handler(self):
        """요청 핸들러 테스트"""
        # 요청 처리
        response = self.request_handler.process_request("간단한 인사말을 생성해주세요")
        self.assertTrue(response.get("success", False), "요청 처리 실패")
        self.assertIsNotNone(response.get("text"), "응답 텍스트가 없습니다")
    
    def test_conversation_manager(self):
        """대화 관리자 테스트"""
        # 시스템 프롬프트 설정
        self.conversation_manager.set_system_prompt("테스트 시스템 프롬프트")
        
        # 사용자 메시지 추가
        response = self.conversation_manager.add_user_message("안녕하세요")
        self.assertTrue(response.get("success", False), "사용자 메시지 추가 실패")
        
        # 대화 기록 확인
        history = self.conversation_manager.get_conversation_history()
        self.assertGreaterEqual(len(history), 2, "대화 기록이 올바르게 저장되지 않았습니다")
        
        # 대화 초기화
        self.conversation_manager.clear_conversation()
        history = self.conversation_manager.get_conversation_history()
        self.assertEqual(len(history), 0, "대화 초기화 실패")
    
    def test_input_processor(self):
        """입력 처리기 테스트"""
        # 일반 메시지 처리
        result = self.input_processor.process_input("일반 메시지")
        self.assertEqual(result["type"], "message", "일반 메시지 처리 실패")
        self.assertFalse(result["is_command"], "일반 메시지가 명령어로 인식됨")
        
        # 명령어 처리
        result = self.input_processor.process_input("reset")
        self.assertEqual(result["type"], "command", "명령어 처리 실패")
        self.assertTrue(result["is_command"], "명령어가 일반 메시지로 인식됨")
        self.assertEqual(result["command"], "reset", "명령어 인식 실패")
        
        # 빈 입력 처리
        result = self.input_processor.process_input("   ")
        self.assertEqual(result["type"], "empty", "빈 입력 처리 실패")
    
    def test_response_generator(self):
        """응답 생성기 테스트"""
        # 응답 생성
        response = self.response_generator.generate_response("간단한 인사말을 생성해주세요")
        self.assertTrue(response.get("success", False), "응답 생성 실패")
        self.assertIsNotNone(response.get("text"), "응답 텍스트가 없습니다")
        self.assertGreater(len(response.get("text", "")), 0, "응답 텍스트가 비어 있습니다")
        
        # 대화 기록 확인
        history = self.response_generator.get_conversation_history()
        self.assertGreaterEqual(len(history), 2, "대화 기록이 올바르게 저장되지 않았습니다")
        
        # 대화 초기화
        self.response_generator.clear_conversation()
        history = self.response_generator.get_conversation_history()
        self.assertEqual(len(history), 0, "대화 초기화 실패")
    
    def test_error_handler(self):
        """오류 처리기 테스트"""
        # 오류 처리
        error = ValueError("테스트 오류")
        error_response = self.error_handler.handle_error(error, {"context": "테스트"})
        self.assertFalse(error_response.get("success", True), "오류 응답이 성공으로 표시됨")
        self.assertEqual(error_response.get("error_type"), "ValueError", "오류 유형 불일치")
        self.assertIsNotNone(error_response.get("text"), "사용자 오류 메시지가 없습니다")
        
        # 오류 통계
        stats = self.error_handler.get_error_stats()
        self.assertEqual(stats["total_errors"], 1, "오류 카운터 불일치")
        self.assertIn("ValueError", stats["error_types"], "오류 유형 통계 불일치")

if __name__ == "__main__":
    unittest.main()
