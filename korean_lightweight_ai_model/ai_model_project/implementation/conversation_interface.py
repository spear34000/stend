"""
한국어 특화 경량화 AI 모델 대화 인터페이스
- 사용자와 대화형 인터페이스 제공
- 3GB RAM 환경에서 실행 가능한 경량화 모델 활용
- 자연스러운 한국어 대화 생성
"""

import os
import sys
import time
import logging
import argparse
import torch
from model_loader import KoreanLightweightModel
from tokenizer_optimizer import KoreanTokenizerOptimizer
from conversation_optimizer import ConversationOptimizer
from inference_optimizer import InferenceOptimizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("conversation_interface.log")
    ]
)
logger = logging.getLogger(__name__)

class ConversationInterface:
    """대화 인터페이스 클래스"""
    
    def __init__(self, model_name="EleutherAI/polyglot-ko-1.3b", use_4bit=True, max_memory_gb=2.5):
        """
        대화 인터페이스 초기화
        
        Args:
            model_name: 모델 이름 (Hugging Face 모델 ID)
            use_4bit: 4비트 양자화 사용 여부
            max_memory_gb: 최대 허용 메모리 (GB)
        """
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.max_memory_gb = max_memory_gb
        
        self.model_loader = None
        self.tokenizer_optimizer = None
        self.conversation_optimizer = None
        self.inference_optimizer = None
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"대화 인터페이스 초기화: {model_name}, 4비트 양자화={use_4bit}, 장치={self.device}")
    
    def setup(self):
        """모델 및 최적화 설정"""
        try:
            # 토크나이저 최적화
            logger.info("토크나이저 로드 중...")
            self.tokenizer_optimizer = KoreanTokenizerOptimizer(self.model_name)
            tokenizer = self.tokenizer_optimizer.load_tokenizer()
            
            # 모델 로드
            logger.info("모델 로드 중...")
            self.model_loader = KoreanLightweightModel(
                model_name=self.model_name,
                use_4bit=self.use_4bit,
                device=self.device
            )
            
            model, _ = self.model_loader.load_model(use_sharding=True)
            
            # 추론 최적화
            logger.info("추론 최적화 설정 중...")
            self.inference_optimizer = InferenceOptimizer(
                max_memory_gb=self.max_memory_gb,
                max_cache_size=512
            )
            
            # 추론을 위한 모델 최적화
            self.model_loader.optimize_for_inference()
            
            # 대화 최적화
            logger.info("대화 최적화 설정 중...")
            self.conversation_optimizer = ConversationOptimizer(
                self.model_loader,
                self.tokenizer_optimizer
            )
            
            logger.info("모델 및 최적화 설정 완료")
            return True
            
        except Exception as e:
            logger.error(f"모델 설정 중 오류 발생: {str(e)}", exc_info=True)
            return False
    
    def print_welcome_message(self):
        """환영 메시지 출력"""
        print("\n" + "=" * 60)
        print("  한국어 특화 경량화 AI 모델 대화 인터페이스")
        print("  - 3GB RAM 환경에서 실행 가능한 경량화 모델")
        print("  - 자연스러운 한국어 대화 생성")
        print("=" * 60)
        print("\n대화를 시작합니다. '종료'를 입력하면 대화가 종료됩니다.\n")
    
    def print_model_info(self):
        """모델 정보 출력"""
        print(f"\n모델 정보:")
        print(f"- 모델: {self.model_name}")
        print(f"- 4비트 양자화: {'사용' if self.use_4bit else '미사용'}")
        print(f"- 실행 장치: {self.device}")
        print(f"- 최대 메모리: {self.max_memory_gb}GB\n")
    
    def run_interactive_mode(self):
        """대화형 모드 실행"""
        self.print_welcome_message()
        self.print_model_info()
        
        # 대화 스타일 선택
        print("대화 스타일을 선택하세요:")
        print("1. 정중한 스타일 (formal)")
        print("2. 친근한 스타일 (casual) - 기본값")
        print("3. 전문가 스타일 (professional)")
        
        style_choice = input("선택 (1-3, 기본값: 2): ").strip()
        
        if style_choice == "1":
            self.conversation_optimizer.set_conversation_style("formal")
            print("\n정중한 스타일로 대화합니다.\n")
        elif style_choice == "3":
            self.conversation_optimizer.set_conversation_style("professional")
            print("\n전문가 스타일로 대화합니다.\n")
        else:
            self.conversation_optimizer.set_conversation_style("casual")
            print("\n친근한 스타일로 대화합니다.\n")
        
        # 대화 루프
        while True:
            # 사용자 입력
            user_input = input("\n사용자: ").strip()
            
            # 종료 조건
            if user_input.lower() in ["종료", "exit", "quit", "q"]:
                print("\n대화를 종료합니다. 감사합니다!")
                break
            
            # 빈 입력 처리
            if not user_input:
                print("입력이 없습니다. 다시 입력해주세요.")
                continue
            
            try:
                # 응답 생성 시작 시간
                start_time = time.time()
                
                # 응답 생성
                response = self.conversation_optimizer.generate_response(user_input)
                
                # 응답 생성 완료 시간
                generation_time = time.time() - start_time
                
                # 응답 출력
                print(f"\nAI: {response}")
                print(f"\n(응답 생성 시간: {generation_time:.2f}초)")
                
            except Exception as e:
                logger.error(f"응답 생성 중 오류 발생: {str(e)}", exc_info=True)
                print("\nAI: 죄송합니다. 응답 생성 중 오류가 발생했습니다. 다시 시도해주세요.")
    
    def run_single_query_mode(self, query):
        """단일 쿼리 모드 실행"""
        try:
            # 응답 생성
            response = self.conversation_optimizer.generate_response(query)
            print(f"\n사용자: {query}")
            print(f"\nAI: {response}\n")
            return response
            
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}", exc_info=True)
            print("\nAI: 죄송합니다. 응답 생성 중 오류가 발생했습니다.")
            return None

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="한국어 특화 경량화 AI 모델 대화 인터페이스")
    
    parser.add_argument("--model_name", type=str, default="EleutherAI/polyglot-ko-1.3b",
                        help="사용할 모델 이름 (Hugging Face 모델 ID)")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="4비트 양자화 사용 여부")
    parser.add_argument("--max_memory_gb", type=float, default=2.5,
                        help="최대 허용 메모리 (GB)")
    parser.add_argument("--query", type=str,
                        help="단일 쿼리 모드에서 사용할 질문 (지정하지 않으면 대화형 모드로 실행)")
    
    args = parser.parse_args()
    
    # 대화 인터페이스 생성
    interface = ConversationInterface(
        model_name=args.model_name,
        use_4bit=args.use_4bit,
        max_memory_gb=args.max_memory_gb
    )
    
    # 모델 설정
    if not interface.setup():
        logger.error("모델 설정에 실패했습니다. 프로그램을 종료합니다.")
        return 1
    
    # 실행 모드 결정
    if args.query:
        # 단일 쿼리 모드
        interface.run_single_query_mode(args.query)
    else:
        # 대화형 모드
        interface.run_interactive_mode()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
