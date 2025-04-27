"""
한국어 특화 초경량 AI 모델 - 통합 실행 모듈
- 모델 아키텍처 및 극한의 경량화 기법 통합
- 메인 실행 인터페이스 제공
- 벤치마크 및 성능 측정 기능
"""

import os
import sys
import time
import json
import logging
import argparse
import torch
import psutil
from typing import Dict, List, Optional, Union, Any, Tuple

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ultralight_korean_ai.log")
    ]
)
logger = logging.getLogger(__name__)

# 모듈 임포트
from model_architecture import ModelConfig, TokenizerConfig, InferenceConfig, ConversationConfig
from model_architecture import ModelArchitecture, create_extreme_lightweight_architecture
from extreme_optimizer import ExtremeOptimizer
from conversation_optimizer import ConversationOptimizer
from tokenizer_optimizer import KoreanTokenizerOptimizer

class UltralightKoreanAI:
    """초경량 한국어 AI 모델 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        초경량 한국어 AI 모델 초기화
        
        Args:
            config_path: 설정 파일 경로 (없으면 기본 설정 사용)
        """
        self.model = None
        self.tokenizer = None
        self.conversation_optimizer = None
        
        # 설정 로드 또는 생성
        if config_path and os.path.exists(config_path):
            self.architecture = ModelArchitecture.load_config(config_path)
            logger.info(f"설정 파일에서 모델 아키텍처 로드: {config_path}")
        else:
            self.architecture = create_extreme_lightweight_architecture()
            logger.info("기본 초경량 모델 아키텍처 생성")
        
        # 모델 설정 가져오기
        self.model_config = self.architecture.model_config
        
        logger.info("초경량 한국어 AI 모델 초기화 완료")
    
    def initialize(self) -> bool:
        """
        모델 초기화
        
        Returns:
            초기화 성공 여부
        """
        try:
            logger.info("모델 초기화 시작")
            
            # 시작 시간 기록
            start_time = time.time()
            
            # 시작 메모리 사용량 기록
            start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            
            # 1. 토크나이저 초기화
            logger.info("토크나이저 초기화 중...")
            tokenizer_optimizer = KoreanTokenizerOptimizer(self.model_config.model_name)
            self.tokenizer = tokenizer_optimizer.load_tokenizer()
            
            # 2. 극한의 최적화 적용
            logger.info("극한의 최적화 적용 중...")
            optimizer = ExtremeOptimizer(self.model_config.to_dict())
            self.model = optimizer.optimize_model(self.model_config.model_name)
            
            # 3. 대화 최적화 초기화
            logger.info("대화 최적화 초기화 중...")
            model_loader = type('ModelLoader', (), {
                'model': self.model,
                'tokenizer': self.tokenizer,
                'generate_text': self.generate_text
            })
            self.conversation_optimizer = ConversationOptimizer(model_loader, tokenizer_optimizer)
            
            # 초기화 시간 및 메모리 사용량 계산
            init_time = time.time() - start_time
            current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            memory_used = current_memory - start_memory
            
            # 모델 크기 정보 가져오기
            model_size_info = optimizer.get_model_size_info(self.model)
            
            logger.info(f"모델 초기화 완료: {init_time:.2f}초, {memory_used:.2f}MB 사용")
            logger.info(f"모델 크기: {model_size_info['memory_mb']:.2f}MB, 파라미터: {model_size_info['param_count']:,}, 희소성: {model_size_info['sparsity']:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"모델 초기화 중 오류 발생: {str(e)}", exc_info=True)
            return False
    
    def generate_text(self, prompt: str, max_length: int = 256, temperature: float = 0.7,
                     top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.1) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 프롬프트 텍스트
            max_length: 최대 생성 길이
            temperature: 생성 온도 (높을수록 다양한 결과)
            top_p: 누적 확률 임계값
            top_k: 상위 k개 토큰 선택
            repetition_penalty: 반복 패널티
        
        Returns:
            생성된 텍스트
        """
        if self.model is None or self.tokenizer is None:
            logger.error("모델이 초기화되지 않았습니다.")
            return ""
        
        try:
            # 입력 인코딩
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.model_config.device)
            attention_mask = inputs.attention_mask.to(self.model_config.device)
            
            # 생성 설정
            gen_kwargs = {
                "max_length": min(max_length + len(input_ids[0]), 2048),
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            }
            
            # 텍스트 생성
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )
            
            # 생성된 텍스트 디코딩
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            return generated_text
            
        except Exception as e:
            logger.error(f"텍스트 생성 중 오류 발생: {str(e)}", exc_info=True)
            return ""
    
    def chat(self, user_input: str, conversation_style: str = "casual") -> str:
        """
        대화 응답 생성
        
        Args:
            user_input: 사용자 입력 텍스트
            conversation_style: 대화 스타일 ("formal", "casual", "professional")
        
        Returns:
            생성된 응답 텍스트
        """
        if self.conversation_optimizer is None:
            logger.error("대화 최적화가 초기화되지 않았습니다.")
            return ""
        
        try:
            # 대화 스타일 설정
            self.conversation_optimizer.set_conversation_style(conversation_style)
            
            # 응답 생성
            start_time = time.time()
            response = self.conversation_optimizer.generate_response(user_input)
            generation_time = time.time() - start_time
            
            logger.info(f"응답 생성 완료: {generation_time:.2f}초")
            return response
            
        except Exception as e:
            logger.error(f"대화 응답 생성 중 오류 발생: {str(e)}", exc_info=True)
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다."
    
    def benchmark(self, test_prompts: List[str], output_path: str = "benchmark_results.json") -> Dict[str, Any]:
        """
        성능 벤치마크
        
        Args:
            test_prompts: 테스트 프롬프트 리스트
            output_path: 결과 저장 경로
        
        Returns:
            벤치마크 결과 딕셔너리
        """
        if self.model is None or self.tokenizer is None:
            logger.error("모델이 초기화되지 않았습니다.")
            return {}
        
        try:
            logger.info(f"성능 벤치마크 시작: {len(test_prompts)} 프롬프트")
            
            results = {
                "model_name": self.model_config.model_name,
                "quantization_bits": self.model_config.quantization_bits,
                "device": self.model_config.device,
                "prompts": []
            }
            
            total_tokens = 0
            total_time = 0
            total_memory = 0
            
            for i, prompt in enumerate(test_prompts):
                logger.info(f"프롬프트 {i+1}/{len(test_prompts)} 벤치마크 중...")
                
                # 메모리 사용량 측정 시작
                start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                
                # 시간 측정 시작
                start_time = time.time()
                
                # 텍스트 생성
                generated_text = self.generate_text(prompt)
                
                # 시간 측정 종료
                generation_time = time.time() - start_time
                
                # 메모리 사용량 측정 종료
                end_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                memory_used = end_memory - start_memory
                
                # 토큰 수 계산
                input_tokens = len(self.tokenizer.encode(prompt))
                output_tokens = len(self.tokenizer.encode(generated_text)) - input_tokens
                
                # 토큰당 시간 계산
                tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
                
                # 결과 저장
                prompt_result = {
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "generation_time": generation_time,
                    "tokens_per_second": tokens_per_second,
                    "memory_used_mb": memory_used
                }
                
                results["prompts"].append(prompt_result)
                
                # 통계 업데이트
                total_tokens += output_tokens
                total_time += generation_time
                total_memory += memory_used
                
                logger.info(f"프롬프트 {i+1} 완료: {generation_time:.2f}초, {tokens_per_second:.2f} 토큰/초, {memory_used:.2f}MB")
            
            # 평균 계산
            avg_tokens = total_tokens / len(test_prompts) if test_prompts else 0
            avg_time = total_time / len(test_prompts) if test_prompts else 0
            avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
            avg_memory = total_memory / len(test_prompts) if test_prompts else 0
            
            # 요약 통계 추가
            results["summary"] = {
                "total_prompts": len(test_prompts),
                "total_tokens": total_tokens,
                "total_time": total_time,
                "avg_tokens": avg_tokens,
                "avg_time": avg_time,
                "avg_tokens_per_second": avg_tokens_per_second,
                "avg_memory_mb": avg_memory
            }
            
            # 결과 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"벤치마크 완료: 평균 {avg_tokens_per_second:.2f} 토큰/초, {avg_memory:.2f}MB")
            logger.info(f"결과가 {output_path}에 저장되었습니다.")
            
            return results
            
        except Exception as e:
            logger.error(f"벤치마크 중 오류 발생: {str(e)}", exc_info=True)
            return {}
    
    def compare_with_other_models(self, other_model_results: Dict[str, Dict[str, Any]],
                                 output_path: str = "model_comparison.json") -> Dict[str, Any]:
        """
        다른 모델과 비교
        
        Args:
            other_model_results: 다른 모델의 벤치마크 결과 딕셔너리
            output_path: 결과 저장 경로
        
        Returns:
            비교 결과 딕셔너리
        """
        try:
            logger.info(f"다른 모델과 비교 시작: {len(other_model_results)} 모델")
            
            # 현재 모델의 벤치마크 결과
            our_model_name = f"UltralightKorean-{self.model_config.quantization_bits}bit"
            our_results = self.benchmark(
                [
                    "안녕하세요, 오늘 날씨가 어떤가요?",
                    "인공지능에 대해 간단히 설명해주세요.",
                    "한국의 전통 음식 중에서 가장 유명한 것은 무엇인가요?",
                    "주말에 서울에서 할 만한 활동을 추천해주세요.",
                    "요즘 가장 인기있는 영화는 무엇인가요?"
                ],
                output_path="our_model_benchmark.json"
            )
            
            # 비교 결과
            comparison = {
                "models": {
                    our_model_name: our_results.get("summary", {})
                },
                "comparison": {
                    "speed": {},
                    "memory": {},
                    "quality": {}
                }
            }
            
            # 다른 모델 결과 추가
            for model_name, results in other_model_results.items():
                comparison["models"][model_name] = results.get("summary", {})
            
            # 속도 비교
            our_speed = our_results.get("summary", {}).get("avg_tokens_per_second", 0)
            for model_name, results in other_model_results.items():
                other_speed = results.get("summary", {}).get("avg_tokens_per_second", 0)
                speed_ratio = our_speed / other_speed if other_speed > 0 else float('inf')
                comparison["comparison"]["speed"][model_name] = {
                    "our_speed": our_speed,
                    "other_speed": other_speed,
                    "ratio": speed_ratio,
                    "faster_by": f"{(speed_ratio - 1) * 100:.1f}%" if speed_ratio > 1 else f"{(1 - speed_ratio) * 100:.1f}% 느림"
                }
            
            # 메모리 비교
            our_memory = our_results.get("summary", {}).get("avg_memory_mb", 0)
            for model_name, results in other_model_results.items():
                other_memory = results.get("summary", {}).get("avg_memory_mb", 0)
                memory_ratio = other_memory / our_memory if our_memory > 0 else float('inf')
                comparison["comparison"]["memory"][model_name] = {
                    "our_memory": our_memory,
                    "other_memory": other_memory,
                    "ratio": memory_ratio,
                    "smaller_by": f"{(memory_ratio - 1) * 100:.1f}%" if memory_ratio > 1 else f"{(1 - memory_ratio) * 100:.1f}% 더 큼"
                }
            
            # 결과 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)
            
            logger.info(f"비교 완료: 결과가 {output_path}에 저장되었습니다.")
            
            return comparison
            
        except Exception as e:
            logger.error(f"모델 비교 중 오류 발생: {str(e)}", exc_info=True)
            return {}
    
    def save_model(self, output_dir: str) -> bool:
        """
        모델 저장
        
        Args:
            output_dir: 저장 디렉토리
        
        Returns:
            저장 성공 여부
        """
        if self.model is None or self.tokenizer is None:
            logger.error("모델이 초기화되지 않았습니다.")
            return False
        
        try:
            logger.info(f"모델 저장 시작: {output_dir}")
            
            # 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 모델 설정 저장
            self.architecture.save_config(output_dir)
            
            # 모델 저장
            self.model.save_pretrained(output_dir)
            
            # 토크나이저 저장
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"모델 저장 완료: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"모델 저장 중 오류 발생: {str(e)}", exc_info=True)
            return False

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="초경량 한국어 AI 모델")
    
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    parser.add_argument("--mode", type=str, choices=["chat", "benchmark", "compare"], default="chat",
                        help="실행 모드 (chat, benchmark, compare)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="출력 디렉토리")
    parser.add_argument("--save_model", action="store_true",
                        help="모델 저장 여부")
    
    args = parser.parse_args()
    
    # 초경량 한국어 AI 모델 생성
    model = UltralightKoreanAI(config_path=args.config)
    
    # 모델 초기화
    if not model.initialize():
        logger.error("모델 초기화에 실패했습니다.")
        return 1
    
    # 실행 모드에 따라 처리
    if args.mode == "chat":
        # 대화 모드
        print("\n" + "=" * 60)
        print("  초경량 한국어 AI 모델 대화 인터페이스")
        print("  - 3GB RAM 이하 환경에서 실행 가능한 초경량 모델")
        print("  - 자연스러운 한국어 대화 생성")
        print("=" * 60)
        print("\n대화를 시작합니다. '종료'를 입력하면 대화가 종료됩니다.\n")
        
        while True:
            user_input = input("\n사용자: ").strip()
            
            if user_input.lower() in ["종료", "exit", "quit", "q"]:
                print("\n대화를 종료합니다. 감사합니다!")
                break
            
            if not user_input:
                print("입력이 없습니다. 다시 입력해주세요.")
                continue
            
            response = model.chat(user_input)
            print(f"\nAI: {response}")
    
    elif args.mode == "benchmark":
        # 벤치마크 모드
        test_prompts = [
            "안녕하세요, 오늘 날씨가 어떤가요?",
            "인공지능에 대해 간단히 설명해주세요.",
            "한국의 전통 음식 중에서 가장 유명한 것은 무엇인가요?",
            "주말에 서울에서 할 만한 활동을 추천해주세요.",
            "요즘 가장 인기있는 영화는 무엇인가요?",
            "한국어와 일본어의 차이점은 무엇인가요?",
            "인공지능의 미래에 대해 어떻게 생각하시나요?",
            "한글의 우수성에 대해 설명해주세요.",
            "외국인이 한국어를 배울 때 가장 어려운 점은 무엇인가요?",
            "한국어 속담 중에서 가장 의미있는 것을 알려주세요."
        ]
        
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "benchmark_results.json")
        
        model.benchmark(test_prompts, output_path=output_path)
    
    elif args.mode == "compare":
        # 비교 모드
        # 다른 모델의 벤치마크 결과 (예시)
        other_model_results = {
            "Polyglot-ko-1.3B": {
                "summary": {
                    "avg_tokens_per_second": 10.5,
                    "avg_memory_mb": 1500
                }
            },
            "KoGPT-2": {
                "summary": {
                    "avg_tokens_per_second": 15.2,
                    "avg_memory_mb": 1200
                }
            },
            "ETRI-Eagle-3B": {
                "summary": {
                    "avg_tokens_per_second": 8.3,
                    "avg_memory_mb": 2800
                }
            }
        }
        
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "model_comparison.json")
        
        model.compare_with_other_models(other_model_results, output_path=output_path)
    
    # 모델 저장 (옵션)
    if args.save_model:
        model_dir = os.path.join(args.output_dir, "model")
        model.save_model(model_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
