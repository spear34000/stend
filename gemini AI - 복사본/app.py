"""
Gemini 2.0 Flash API를 사용한 AI 에이전트의 메인 애플리케이션
"""
import os
import sys
from typing import Optional

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import GeminiAgent

class AIAssistant:
    """
    사용자와 상호작용하는 AI 어시스턴트 클래스
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        AIAssistant 초기화
        
        Args:
            api_key: Gemini API 키 (None인 경우 환경 변수에서 로드)
        """
        # Gemini 에이전트 초기화
        self.agent = GeminiAgent(api_key)
        
        # 시스템 프롬프트 설정
        self.system_prompt = """
        당신은 사용자가 말한 대로 만들어주는 AI 어시스턴트입니다.
        사용자의 요청을 정확히 이해하고 그에 맞는 결과물을 생성해주세요.
        사용자가 요청한 내용이 불분명하면 추가 정보를 요청하세요.
        항상 친절하고 도움이 되는 응답을 제공하세요.
        """
        
        # 시스템 프롬프트 적용
        self._apply_system_prompt()
    
    def _apply_system_prompt(self):
        """시스템 프롬프트를 에이전트에 적용"""
        self.agent.conversation_history.append({
            "role": "system", 
            "content": self.system_prompt
        })
    
    def process_input(self, user_input: str) -> str:
        """
        사용자 입력을 처리하고 응답 생성
        
        Args:
            user_input: 사용자 입력 텍스트
            
        Returns:
            생성된 응답 텍스트
        """
        if not user_input.strip():
            return "입력이 비어 있습니다. 무엇을 도와드릴까요?"
        
        # 에이전트를 통해 응답 생성
        response = self.agent.generate_response(user_input)
        return response
    
    def reset_conversation(self):
        """대화 초기화 및 시스템 프롬프트 재적용"""
        self.agent.clear_conversation()
        self._apply_system_prompt()
    
    def get_conversation(self):
        """현재 대화 기록 반환"""
        return self.agent.get_conversation_history()


def main():
    """메인 함수: 콘솔에서 대화형으로 실행"""
    print("Gemini 2.0 Flash API를 사용한 AI 어시스턴트를 시작합니다.")
    print("종료하려면 'exit' 또는 'quit'를 입력하세요.")
    
    # AI 어시스턴트 초기화
    assistant = AIAssistant()
    
    while True:
        # 사용자 입력 받기
        user_input = input("\n사용자: ")
        
        # 종료 명령 확인
        if user_input.lower() in ["exit", "quit", "종료"]:
            print("AI 어시스턴트를 종료합니다.")
            break
        
        # 대화 초기화 명령 확인
        if user_input.lower() in ["reset", "clear", "초기화"]:
            assistant.reset_conversation()
            print("대화가 초기화되었습니다.")
            continue
        
        # 응답 생성 및 출력
        response = assistant.process_input(user_input)
        print(f"\nAI: {response}")


if __name__ == "__main__":
    main()
