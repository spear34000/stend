"""
한국어 특화 경량화 AI 모델 대화 테스트 스크립트
- 자연스러운 한국어 대화 테스트
- 다양한 대화 주제 및 스타일 평가
- 대화 맥락 유지 능력 검증
"""

import os
import time
import json
import logging
import argparse
from model_loader import KoreanLightweightModel
from tokenizer_optimizer import KoreanTokenizerOptimizer
from conversation_optimizer import ConversationOptimizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("conversation_test.log")
    ]
)
logger = logging.getLogger(__name__)

class ConversationTester:
    """대화 테스트 클래스"""
    
    def __init__(self, model_name="EleutherAI/polyglot-ko-1.3b", use_4bit=True):
        """
        대화 테스트 초기화
        
        Args:
            model_name: 모델 이름 (Hugging Face 모델 ID)
            use_4bit: 4비트 양자화 사용 여부
        """
        self.model_name = model_name
        self.use_4bit = use_4bit
        
        self.model_loader = None
        self.tokenizer_optimizer = None
        self.conversation_optimizer = None
        
        self.test_results = {
            "general_conversation": [],
            "context_maintenance": [],
            "style_adaptation": [],
            "korean_language_quality": []
        }
        
        logger.info(f"대화 테스트 초기화: {model_name}, 4비트 양자화={use_4bit}")
    
    def setup(self):
        """모델 및 최적화 설정"""
        # 토크나이저 최적화
        self.tokenizer_optimizer = KoreanTokenizerOptimizer(self.model_name)
        tokenizer = self.tokenizer_optimizer.load_tokenizer()
        
        # 모델 로드
        self.model_loader = KoreanLightweightModel(
            model_name=self.model_name,
            use_4bit=self.use_4bit,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        model, _ = self.model_loader.load_model(use_sharding=True)
        
        # 추론을 위한 모델 최적화
        self.model_loader.optimize_for_inference()
        
        # 대화 최적화
        self.conversation_optimizer = ConversationOptimizer(
            self.model_loader,
            self.tokenizer_optimizer
        )
        
        logger.info("모델 및 최적화 설정 완료")
    
    def test_general_conversation(self, test_prompts):
        """
        일반 대화 테스트
        
        Args:
            test_prompts: 테스트 프롬프트 리스트
        
        Returns:
            테스트 결과 리스트
        """
        logger.info("일반 대화 테스트 시작")
        
        results = []
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"프롬프트 {i+1}/{len(test_prompts)}: {prompt}")
            
            # 대화 이력 초기화
            self.conversation_optimizer.clear_conversation_history()
            
            # 응답 생성
            start_time = time.time()
            response = self.conversation_optimizer.generate_response(prompt)
            generation_time = time.time() - start_time
            
            result = {
                "prompt": prompt,
                "response": response,
                "generation_time": generation_time
            }
            
            results.append(result)
            logger.info(f"응답: {response}")
            logger.info(f"생성 시간: {generation_time:.2f}초")
        
        self.test_results["general_conversation"] = results
        return results
    
    def test_context_maintenance(self, conversation_flows):
        """
        대화 맥락 유지 테스트
        
        Args:
            conversation_flows: 대화 흐름 리스트 (각 흐름은 연속된 프롬프트 리스트)
        
        Returns:
            테스트 결과 리스트
        """
        logger.info("대화 맥락 유지 테스트 시작")
        
        results = []
        
        for i, flow in enumerate(conversation_flows):
            logger.info(f"대화 흐름 {i+1}/{len(conversation_flows)} 테스트")
            
            # 대화 이력 초기화
            self.conversation_optimizer.clear_conversation_history()
            
            flow_result = {
                "conversation_id": i+1,
                "exchanges": []
            }
            
            for j, prompt in enumerate(flow):
                logger.info(f"  턴 {j+1}/{len(flow)}: {prompt}")
                
                # 응답 생성
                response = self.conversation_optimizer.generate_response(prompt)
                
                exchange = {
                    "turn": j+1,
                    "prompt": prompt,
                    "response": response
                }
                
                flow_result["exchanges"].append(exchange)
                logger.info(f"  응답: {response}")
            
            results.append(flow_result)
        
        self.test_results["context_maintenance"] = results
        return results
    
    def test_style_adaptation(self, prompt, styles=None):
        """
        대화 스타일 적응 테스트
        
        Args:
            prompt: 테스트 프롬프트
            styles: 테스트할 스타일 리스트 (기본값: ["formal", "casual", "professional"])
        
        Returns:
            테스트 결과 리스트
        """
        if styles is None:
            styles = ["formal", "casual", "professional"]
        
        logger.info("대화 스타일 적응 테스트 시작")
        logger.info(f"프롬프트: {prompt}")
        
        results = []
        
        for style in styles:
            logger.info(f"스타일 '{style}' 테스트")
            
            # 대화 이력 초기화
            self.conversation_optimizer.clear_conversation_history()
            
            # 스타일 설정
            self.conversation_optimizer.set_conversation_style(style)
            
            # 응답 생성
            response = self.conversation_optimizer.generate_response(prompt)
            
            result = {
                "style": style,
                "prompt": prompt,
                "response": response
            }
            
            results.append(result)
            logger.info(f"응답 ({style}): {response}")
        
        self.test_results["style_adaptation"] = results
        return results
    
    def test_korean_language_quality(self, test_prompts):
        """
        한국어 언어 품질 테스트
        
        Args:
            test_prompts: 테스트 프롬프트 리스트
        
        Returns:
            테스트 결과 리스트
        """
        logger.info("한국어 언어 품질 테스트 시작")
        
        results = []
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"프롬프트 {i+1}/{len(test_prompts)}: {prompt}")
            
            # 대화 이력 초기화
            self.conversation_optimizer.clear_conversation_history()
            
            # 응답 생성
            response = self.conversation_optimizer.generate_response(prompt)
            
            # 한글 문자 비율
            import re
            total_chars = len(response)
            korean_chars = len(re.findall(r'[가-힣]', response))
            korean_char_ratio = korean_chars / total_chars if total_chars > 0 else 0
            
            # 자모 분리 비율
            jamo_chars = len(re.findall(r'[ㄱ-ㅎㅏ-ㅣ]', response))
            jamo_separation_ratio = jamo_chars / total_chars if total_chars > 0 else 0
            
            result = {
                "prompt": prompt,
                "response": response,
                "korean_char_ratio": korean_char_ratio,
                "jamo_separation_ratio": jamo_separation_ratio
            }
            
            results.append(result)
            logger.info(f"응답: {response}")
            logger.info(f"한글 문자 비율: {korean_char_ratio:.2f}, 자모 분리 비율: {jamo_separation_ratio:.2f}")
        
        self.test_results["korean_language_quality"] = results
        return results
    
    def run_all_tests(self, output_dir="./conversation_test_results"):
        """
        모든 테스트 실행
        
        Args:
            output_dir: 결과 저장 디렉토리
        
        Returns:
            전체 테스트 결과 딕셔너리
        """
        logger.info("모든 대화 테스트 시작")
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 모델 설정
        self.setup()
        
        # 1. 일반 대화 테스트
        general_test_prompts = [
            "안녕하세요, 오늘 날씨가 어떤가요?",
            "인공지능에 대해 간단히 설명해주세요.",
            "한국의 전통 음식 중에서 가장 유명한 것은 무엇인가요?",
            "주말에 서울에서 할 만한 활동을 추천해주세요.",
            "요즘 가장 인기있는 영화는 무엇인가요?"
        ]
        self.test_general_conversation(general_test_prompts)
        
        # 2. 대화 맥락 유지 테스트
        conversation_flows = [
            [
                "자기소개를 해주세요.",
                "어떤 기능을 가지고 있나요?",
                "3GB RAM 환경에서도 잘 작동하나요?"
            ],
            [
                "한국 역사에 대해 알려주세요.",
                "조선시대는 언제부터 언제까지인가요?",
                "조선시대의 대표적인 왕은 누구인가요?",
                "세종대왕의 업적에 대해 더 자세히 알려주세요."
            ]
        ]
        self.test_context_maintenance(conversation_flows)
        
        # 3. 대화 스타일 적응 테스트
        style_test_prompt = "인공지능의 미래에 대해 어떻게 생각하시나요?"
        self.test_style_adaptation(style_test_prompt)
        
        # 4. 한국어 언어 품질 테스트
        korean_quality_prompts = [
            "한글의 우수성에 대해 설명해주세요.",
            "한국어와 일본어의 차이점은 무엇인가요?",
            "외국인이 한국어를 배울 때 가장 어려운 점은 무엇인가요?",
            "한국어 속담 중에서 가장 의미있는 것을 알려주세요."
        ]
        self.test_korean_language_quality(korean_quality_prompts)
        
        # 결과 저장
        with open(os.path.join(output_dir, "conversation_test_results.json"), "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        # 대화 예시 저장
        with open(os.path.join(output_dir, "conversation_examples.txt"), "w", encoding="utf-8") as f:
            # 일반 대화 예시
            f.write("=== 일반 대화 예시 ===\n\n")
            for i, result in enumerate(self.test_results["general_conversation"]):
                f.write(f"[대화 {i+1}]\n")
                f.write(f"사용자: {result['prompt']}\n")
                f.write(f"AI: {result['response']}\n\n")
            
            # 대화 맥락 유지 예시
            f.write("\n=== 대화 맥락 유지 예시 ===\n\n")
            for flow in self.test_results["context_maintenance"]:
                f.write(f"[대화 흐름 {flow['conversation_id']}]\n")
                for exchange in flow["exchanges"]:
                    f.write(f"사용자 ({exchange['turn']}): {exchange['prompt']}\n")
                    f.write(f"AI: {exchange['response']}\n")
                f.write("\n")
            
            # 대화 스타일 예시
            f.write("\n=== 대화 스타일 예시 ===\n\n")
            for result in self.test_results["style_adaptation"]:
                f.write(f"[스타일: {result['style']}]\n")
                f.write(f"사용자: {result['prompt']}\n")
                f.write(f"AI: {result['response']}\n\n")
        
        logger.info(f"테스트 결과가 {output_dir}에 저장되었습니다.")
        
        return self.test_results

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="한국어 특화 경량화 AI 모델 대화 테스트")
    
    parser.add_argument("--model_name", type=str, default="EleutherAI/polyglot-ko-1.3b",
                        help="테스트할 모델 이름 (Hugging Face 모델 ID)")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="4비트 양자화 사용 여부")
    parser.add_argument("--output_dir", type=str, default="./conversation_test_results",
                        help="결과 저장 디렉토리")
    
    args = parser.parse_args()
    
    # 대화 테스트
    tester = ConversationTester(
        model_name=args.model_name,
        use_4bit=args.use_4bit
    )
    
    tester.run_all_tests(args.output_dir)

if __name__ == "__main__":
    import torch
    main()
