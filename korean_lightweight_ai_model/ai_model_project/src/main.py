"""
한국어 특화 초경량 AI 모델 - 메인 모듈

이 모듈은 한국어 특화 초경량 AI 모델의 메인 진입점입니다.
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Optional, Union, Any, Tuple

# 프로젝트 루트 디렉토리 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 모듈 임포트
from src.models.korean_model import ChatGPTLevelKoreanModel
from src.tokenizers.korean_tokenizer_optimizer import KoreanTokenizerOptimizer
from src.optimizers.extreme_optimizer import ExtremeModelOptimizer
from src.inference.inference_optimizer import ChatGPTLevelInferenceOptimizer
from src.conversation.conversation_optimizer import ChatGPTLevelConversationOptimizer
from src.utils.utils import ConfigManager, MemoryMonitor, PerformanceTracker, TorchUtils

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraLightKoreanAI:
    """한국어 특화 초경량 AI 모델 메인 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        한국어 특화 초경량 AI 모델 초기화
        
        Args:
            config_path: 설정 파일 경로 (None인 경우 기본 설정 사용)
        """
        # 설정 관리자 초기화
        self.config_manager = ConfigManager(config_path)
        
        # 모델 설정 가져오기
        self.model_config = self.config_manager.get_config("model_config")
        self.tokenizer_config = self.config_manager.get_config("tokenizer_config")
        self.inference_config = self.config_manager.get_config("inference_config")
        self.conversation_config = self.config_manager.get_config("conversation_config")
        
        # 컴포넌트 초기화
        self.model = None
        self.tokenizer = None
        self.inference_optimizer = None
        self.conversation_optimizer = None
        self.memory_monitor = MemoryMonitor()
        self.performance_tracker = PerformanceTracker()
        
        logger.info("한국어 특화 초경량 AI 모델 초기화 완료")
    
    def load(self) -> None:
        """모델 및 관련 컴포넌트 로드"""
        logger.info("모델 및 관련 컴포넌트 로드 시작")
        
        try:
            # 메모리 모니터링 시작
            self.memory_monitor.start_monitoring()
            
            # 성능 추적 시작
            self.performance_tracker.start_tracking()
            
            # 1. 토크나이저 초기화 및 최적화
            logger.info("토크나이저 초기화 및 최적화 시작")
            self.tokenizer = KoreanTokenizerOptimizer(
                model_name=self.model_config.get("model_name", "polyglot-ko-410m"),
                **self.tokenizer_config
            ).load_tokenizer()
            
            # 2. 모델 초기화 및 로드
            logger.info("모델 초기화 및 로드 시작")
            self.model = ChatGPTLevelKoreanModel(self.model_config).load_model()
            
            # 3. 모델 최적화
            logger.info("모델 최적화 시작")
            model_optimizer = ExtremeModelOptimizer(
                quantization_bits=self.model_config.get("quantization_bits", 2),
                pruning_ratio=self.model_config.get("pruning_ratio", 0.5)
            )
            self.model = model_optimizer.optimize_model(self.model)
            
            # 4. 추론 최적화기 초기화
            logger.info("추론 최적화기 초기화")
            self.inference_optimizer = ChatGPTLevelInferenceOptimizer(
                kv_cache_size_mb=self.inference_config.get("kv_cache_size_mb", 128),
                use_memory_mapping=self.inference_config.get("use_memory_mapping", True),
                enable_inplace_operations=self.inference_config.get("enable_inplace_operations", True),
                max_memory_gb=self.model_config.get("max_memory_gb", 2.5),
                use_flash_attention=True,
                use_grouped_query_attention=True
            )
            self.model = self.inference_optimizer.optimize_for_inference(self.model)
            
            # 5. 대화 최적화기 초기화
            logger.info("대화 최적화기 초기화")
            self.conversation_optimizer = ChatGPTLevelConversationOptimizer(
                context_length=self.conversation_config.get("context_length", 5),
                response_quality_enhancement=self.conversation_config.get("response_quality_enhancement", True),
                default_style=self.conversation_config.get("default_style", "casual"),
                personality="helpful",
                enable_context_awareness=True
            )
            
            # 성능 추적 종료
            self.performance_tracker.stop_tracking()
            
            # 메모리 사용량 통계 기록
            memory_stats = self.memory_monitor.get_memory_usage_stats()
            for key, value in memory_stats.items():
                self.performance_tracker.add_metric(key, value)
            
            # 모델 크기 정보 기록
            model_size_info = TorchUtils.get_model_size(self.model)
            for key, value in model_size_info.items():
                self.performance_tracker.add_metric(key, value)
            
            # 성능 메트릭 로깅
            self.performance_tracker.log_metrics()
            
            logger.info("모델 및 관련 컴포넌트 로드 완료")
            
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}", exc_info=True)
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 프롬프트
            **kwargs: 추가 생성 설정
        
        Returns:
            생성된 텍스트
        """
        if self.model is None or self.tokenizer is None:
            logger.error("모델이 로드되지 않았습니다. generate() 호출 전에 load()를 호출하세요.")
            return ""
        
        try:
            # 생성 설정
            generation_config = {
                "max_new_tokens": kwargs.get("max_new_tokens", self.inference_config.get("max_new_tokens", 512)),
                "do_sample": kwargs.get("do_sample", True),
                "temperature": kwargs.get("temperature", self.conversation_config.get("temperature", 0.7)),
                "top_p": kwargs.get("top_p", self.conversation_config.get("top_p", 0.9)),
                "repetition_penalty": kwargs.get("repetition_penalty", self.conversation_config.get("repetition_penalty", 1.1))
            }
            
            # 텍스트 생성
            from src.inference.inference_optimizer import MemoryEfficientInference
            
            memory_efficient_inference = MemoryEfficientInference(
                max_memory_gb=self.model_config.get("max_memory_gb", 2.5),
                batch_size=self.inference_config.get("batch_size", 1),
                max_new_tokens=generation_config["max_new_tokens"]
            )
            
            generated_text = memory_efficient_inference.generate_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                generation_config=generation_config
            )
            
            return generated_text
            
        except Exception as e:
            logger.error(f"텍스트 생성 중 오류 발생: {str(e)}", exc_info=True)
            return ""
    
    def chat(self, user_input: str, **kwargs) -> str:
        """
        대화 응답 생성
        
        Args:
            user_input: 사용자 입력
            **kwargs: 추가 대화 설정
        
        Returns:
            대화 응답
        """
        if self.model is None or self.tokenizer is None:
            logger.error("모델이 로드되지 않았습니다. chat() 호출 전에 load()를 호출하세요.")
            return ""
        
        try:
            # 대화 스타일 및 성격 설정
            style = kwargs.get("style", self.conversation_config.get("default_style", "casual"))
            personality = kwargs.get("personality", "helpful")
            
            # 대화 컨텍스트 가져오기
            context = self.conversation_optimizer.get_conversation_context()
            
            # 프롬프트 구성
            if context:
                # 이전 대화 컨텍스트가 있는 경우
                prompt = "이전 대화:\n"
                for i, conv in enumerate(context):
                    prompt += f"사용자: {conv['user']}\n"
                    prompt += f"AI: {conv['assistant']}\n"
                prompt += f"사용자: {user_input}\nAI: "
            else:
                # 첫 대화인 경우
                prompt = f"사용자: {user_input}\nAI: "
            
            # 응답 생성
            model_response = self.generate(prompt, **kwargs)
            
            # AI 응답 부분만 추출
            if "AI: " in model_response:
                model_response = model_response.split("AI: ")[-1].strip()
            
            # 대화 최적화 적용
            optimized_response = self.conversation_optimizer.process_conversation(
                user_input=user_input,
                model_response=model_response,
                style=style,
                personality=personality
            )
            
            return optimized_response
            
        except Exception as e:
            logger.error(f"대화 응답 생성 중 오류 발생: {str(e)}", exc_info=True)
            return ""
    
    def save(self, output_dir: str) -> None:
        """
        모델 및 설정 저장
        
        Args:
            output_dir: 출력 디렉토리
        """
        if self.model is None:
            logger.error("모델이 로드되지 않았습니다. save() 호출 전에 load()를 호출하세요.")
            return
        
        try:
            # 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 모델 저장
            model_dir = os.path.join(output_dir, "model")
            self.model.save_pretrained(model_dir)
            logger.info(f"모델 저장 완료: {model_dir}")
            
            # 토크나이저 저장
            tokenizer_dir = os.path.join(output_dir, "tokenizer")
            self.tokenizer.save_pretrained(tokenizer_dir)
            logger.info(f"토크나이저 저장 완료: {tokenizer_dir}")
            
            # 설정 저장
            config_path = os.path.join(output_dir, "config.json")
            self.config_manager.save_config(config_path)
            logger.info(f"설정 저장 완료: {config_path}")
            
        except Exception as e:
            logger.error(f"모델 및 설정 저장 중 오류 발생: {str(e)}", exc_info=True)

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="한국어 특화 초경량 AI 모델")
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    parser.add_argument("--prompt", type=str, help="생성 프롬프트")
    parser.add_argument("--chat", type=str, help="대화 입력")
    parser.add_argument("--output", type=str, help="출력 디렉토리")
    args = parser.parse_args()
    
    # 모델 초기화 및 로드
    model = UltraLightKoreanAI(config_path=args.config)
    model.load()
    
    # 텍스트 생성 또는 대화
    if args.prompt:
        generated_text = model.generate(args.prompt)
        print(f"생성된 텍스트: {generated_text}")
    elif args.chat:
        response = model.chat(args.chat)
        print(f"AI 응답: {response}")
    
    # 모델 저장
    if args.output:
        model.save(args.output)

if __name__ == "__main__":
    main()
