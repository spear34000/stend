"""
Gemini 2.0 Flash API를 사용한 AI 에이전트 메인 실행 파일
"""
import os
import sys
import argparse
from dotenv import load_dotenv

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 환경 변수 로드
load_dotenv()

from src.input_processor import InputProcessor
from src.response_generator import ResponseGenerator
from src.error_handler import ErrorHandler

def main():
    """메인 함수: AI 에이전트 실행"""
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description="Gemini 2.0 Flash API를 사용한 AI 에이전트")
    parser.add_argument("--api_key", help="Gemini API 키 (설정되지 않은 경우 .env 파일에서 로드)")
    parser.add_argument("--temperature", type=float, default=0.7, help="응답 생성 온도 (0.0 ~ 1.0)")
    parser.add_argument("--max_tokens", type=int, default=2048, help="최대 출력 토큰 수")
    args = parser.parse_args()
    
    # API 키 설정
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "YOUR_API_KEY":
        print("오류: API 키가 설정되지 않았습니다.")
        print("API 키를 설정하려면 다음 중 하나를 수행하세요:")
        print("1. .env 파일에 GEMINI_API_KEY 설정")
        print("2. --api_key 인수로 API 키 전달")
        print("\nAPI 키 획득 방법은 API_KEY_SETUP.md 파일을 참조하세요.")
        return
    
    # 생성 설정
    generation_config = {
        "temperature": args.temperature,
        "max_output_tokens": args.max_tokens
    }
    
    # 컴포넌트 초기화
    input_processor = InputProcessor()
    response_generator = ResponseGenerator(api_key)
    error_handler = ErrorHandler()
    
    # 시작 메시지 출력
    print("\n" + "=" * 50)
    print("Gemini 2.0 Flash API를 사용한 AI 에이전트를 시작합니다.")
    print("사용자가 말한 대로 만들어주는 AI 에이전트입니다.")
    print("종료하려면 'exit' 또는 '종료'를 입력하세요.")
    print("대화를 초기화하려면 'reset' 또는 '초기화'를 입력하세요.")
    print("도움말을 보려면 'help' 또는 '도움말'을 입력하세요.")
    print("=" * 50 + "\n")
    
    # 대화 루프
    while True:
        try:
            # 사용자 입력 받기
            user_input = input("\n사용자: ")
            
            # 입력 처리
            processed_input = input_processor.process_input(user_input)
            
            # 명령어 처리
            if processed_input["is_command"]:
                command = processed_input["command"]
                
                # 종료 명령
                if command == "exit":
                    print("\nAI 에이전트를 종료합니다.")
                    break
                
                # 초기화 명령
                elif command == "reset":
                    response_generator.clear_conversation()
                    print("\n대화가 초기화되었습니다.")
                    continue
                
                # 도움말 명령
                elif command == "help":
                    print("\n" + "=" * 50)
                    print("AI 에이전트 도움말:")
                    print("- 종료: 'exit', 'quit', '종료', '나가기'")
                    print("- 초기화: 'reset', 'clear', '초기화', '리셋'")
                    print("- 도움말: 'help', '도움말', '도움'")
                    print("- 매개변수 설정: '온도:0.8', '최대토큰:100' 등")
                    print("=" * 50)
                    continue
            
            # 빈 입력 처리
            if processed_input["type"] == "empty":
                print("\nAI: 무엇을 도와드릴까요?")
                continue
            
            # 매개변수 추출
            params = input_processor.extract_parameters(user_input)
            if params:
                generation_config.update(params)
            
            # 응답 생성
            response = response_generator.generate_response(
                processed_input["content"],
                generation_config
            )
            
            # 응답 출력
            if response.get("success", False):
                print(f"\nAI: {response['text']}")
            else:
                error_message = response.get("text", "응답 생성 중 오류가 발생했습니다.")
                print(f"\nAI: {error_message}")
                
        except KeyboardInterrupt:
            print("\n\nAI 에이전트를 종료합니다.")
            break
            
        except Exception as e:
            # 예상치 못한 오류 처리
            error_response = error_handler.handle_error(e)
            print(f"\nAI: {error_response['text']}")

if __name__ == "__main__":
    main()
