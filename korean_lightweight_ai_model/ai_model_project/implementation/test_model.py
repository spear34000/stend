"""
한국어 특화 경량화 AI 모델 테스트 스크립트
- 간단한 테스트 프롬프트로 모델 작동 확인
- 메모리 사용량 및 추론 속도 측정
"""

import os
import time
import psutil
import logging
from model_loader import KoreanLightweightModel
from tokenizer_optimizer import KoreanTokenizerOptimizer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_memory_usage():
    """현재 메모리 사용량 출력"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / (1024 * 1024)
    logger.info(f"현재 메모리 사용량: {memory_usage_mb:.2f} MB")
    return memory_usage_mb

def main():
    """메인 테스트 함수"""
    logger.info("한국어 특화 경량화 AI 모델 테스트 시작")
    
    # 초기 메모리 사용량
    initial_memory = print_memory_usage()
    
    # 테스트 프롬프트
    test_prompt = "안녕하세요, 한국어 AI 모델입니다. 저는 3GB RAM 환경에서도 작동할 수 있습니다."
    
    try:
        # 1. 토크나이저 로드
        logger.info("1. 토크나이저 로드 중...")
        tokenizer_optimizer = KoreanTokenizerOptimizer("EleutherAI/polyglot-ko-1.3b")
        tokenizer = tokenizer_optimizer.load_tokenizer()
        
        # 토크나이저 로드 후 메모리 사용량
        after_tokenizer_memory = print_memory_usage()
        logger.info(f"토크나이저 로드로 인한 메모리 증가: {after_tokenizer_memory - initial_memory:.2f} MB")
        
        # 2. 모델 로드
        logger.info("2. 모델 로드 중...")
        model_loader = KoreanLightweightModel(
            model_name="EleutherAI/polyglot-ko-1.3b",
            use_4bit=True,
            device="cpu"
        )
        
        # 모델 로드 시작 시간
        model_load_start_time = time.time()
        
        # 모델 로드
        model, _ = model_loader.load_model(use_sharding=True)
        
        # 모델 로드 완료 시간
        model_load_time = time.time() - model_load_start_time
        logger.info(f"모델 로드 시간: {model_load_time:.2f}초")
        
        # 모델 로드 후 메모리 사용량
        after_model_memory = print_memory_usage()
        logger.info(f"모델 로드로 인한 메모리 증가: {after_model_memory - after_tokenizer_memory:.2f} MB")
        
        # 3. 추론 최적화
        logger.info("3. 추론 최적화 적용 중...")
        model_loader.optimize_for_inference()
        
        # 4. 텍스트 생성 테스트
        logger.info("4. 텍스트 생성 테스트 중...")
        logger.info(f"프롬프트: {test_prompt}")
        
        # 토큰화 효율성 분석
        tokenization_analysis = tokenizer_optimizer.analyze_tokenization_efficiency(test_prompt)
        logger.info(f"토큰화 효율성: {tokenization_analysis['korean_char_per_token']:.2f} 한글 문자/토큰")
        
        # 텍스트 생성 시작 시간
        generation_start_time = time.time()
        
        # 텍스트 생성
        generated_text = model_loader.generate_text(
            test_prompt,
            max_length=128,
            temperature=0.7,
            top_p=0.9
        )
        
        # 텍스트 생성 완료 시간
        generation_time = time.time() - generation_start_time
        
        # 생성된 토큰 수
        input_tokens = tokenizer.tokenize(test_prompt)
        output_tokens = tokenizer.tokenize(generated_text)
        generated_tokens = len(output_tokens) - len(input_tokens)
        
        # 초당 토큰 생성 속도
        tokens_per_second = generated_tokens / generation_time if generation_time > 0 else 0
        
        logger.info(f"생성된 텍스트: {generated_text}")
        logger.info(f"생성 시간: {generation_time:.2f}초")
        logger.info(f"생성된 토큰 수: {generated_tokens}")
        logger.info(f"초당 토큰 생성 속도: {tokens_per_second:.2f}")
        
        # 생성 후 메모리 사용량
        after_generation_memory = print_memory_usage()
        logger.info(f"텍스트 생성으로 인한 메모리 증가: {after_generation_memory - after_model_memory:.2f} MB")
        
        # 5. 결과 요약
        logger.info("5. 테스트 결과 요약")
        logger.info(f"총 메모리 사용량: {after_generation_memory:.2f} MB")
        logger.info(f"모델 로드 시간: {model_load_time:.2f}초")
        logger.info(f"텍스트 생성 시간: {generation_time:.2f}초")
        logger.info(f"초당 토큰 생성 속도: {tokens_per_second:.2f}")
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}", exc_info=True)
        return 1
    
    logger.info("한국어 특화 경량화 AI 모델 테스트 완료")
    return 0

if __name__ == "__main__":
    main()
