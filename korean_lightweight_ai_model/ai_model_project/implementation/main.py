"""
한국어 특화 경량화 AI 모델 메인 실행 모듈
- 모델 로더, 추론 최적화, 토크나이저 최적화 통합
- 3GB RAM 환경에서 실행 가능한 추론 파이프라인
"""

import os
import sys
import time
import logging
import argparse
import torch
import psutil
from model_loader import KoreanLightweightModel
from inference_optimizer import InferenceOptimizer
from tokenizer_optimizer import KoreanTokenizerOptimizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("korean_lightweight_model.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="한국어 특화 경량화 AI 모델 실행")
    
    parser.add_argument("--model_name", type=str, default="EleutherAI/polyglot-ko-1.3b",
                        help="사용할 기본 모델 이름 (Hugging Face 모델 ID)")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="4비트 양자화 사용 여부")
    parser.add_argument("--use_sharding", action="store_true", default=True,
                        help="모델 샤딩 사용 여부")
    parser.add_argument("--max_memory_gb", type=float, default=2.5,
                        help="최대 허용 메모리 (GB)")
    parser.add_argument("--max_cache_size", type=int, default=512,
                        help="최대 KV 캐시 크기 (토큰 수)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="출력 디렉토리 경로")
    parser.add_argument("--prompt", type=str, default="안녕하세요, 저는 한국어 AI 모델입니다.",
                        help="테스트용 프롬프트")
    parser.add_argument("--max_length", type=int, default=128,
                        help="최대 생성 길이")
    
    return parser.parse_args()

def print_memory_usage():
    """현재 메모리 사용량 출력"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / (1024 * 1024)
    logger.info(f"현재 메모리 사용량: {memory_usage_mb:.2f} MB")
    return memory_usage_mb

def main():
    """메인 실행 함수"""
    args = parse_arguments()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("한국어 특화 경량화 AI 모델 실행 시작")
    logger.info(f"설정: 모델={args.model_name}, 4비트 양자화={args.use_4bit}, "
                f"모델 샤딩={args.use_sharding}, 최대 메모리={args.max_memory_gb}GB")
    
    # 초기 메모리 사용량 확인
    initial_memory = print_memory_usage()
    
    try:
        # 1. 토크나이저 최적화
        logger.info("1. 토크나이저 최적화 시작")
        tokenizer_optimizer = KoreanTokenizerOptimizer(args.model_name)
        tokenizer = tokenizer_optimizer.load_tokenizer()
        
        # 토큰화 효율성 분석
        tokenization_analysis = tokenizer_optimizer.analyze_tokenization_efficiency(args.prompt)
        logger.info(f"토큰화 효율성: {tokenization_analysis['korean_char_per_token']:.2f} 한글 문자/토큰")
        
        # 2. 모델 로드 및 최적화
        logger.info("2. 모델 로드 및 최적화 시작")
        model_loader = KoreanLightweightModel(
            model_name=args.model_name,
            use_4bit=args.use_4bit,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 모델 로드
        model, _ = model_loader.load_model(use_sharding=args.use_sharding)
        
        # 모델 로드 후 메모리 사용량 확인
        after_load_memory = print_memory_usage()
        logger.info(f"모델 로드로 인한 메모리 증가: {after_load_memory - initial_memory:.2f} MB")
        
        # 3. 추론 최적화 설정
        logger.info("3. 추론 최적화 설정")
        inference_optimizer = InferenceOptimizer(
            max_memory_gb=args.max_memory_gb,
            max_cache_size=args.max_cache_size
        )
        
        # 추론을 위한 모델 최적화
        model_loader.optimize_for_inference()
        
        # 4. 텍스트 생성 테스트
        logger.info("4. 텍스트 생성 테스트 시작")
        logger.info(f"프롬프트: {args.prompt}")
        
        # 시간 측정 시작
        start_time = time.time()
        
        # 텍스트 생성
        generated_text = model_loader.generate_text(
            args.prompt,
            max_length=args.max_length,
            temperature=0.7,
            top_p=0.9
        )
        
        # 시간 측정 종료
        end_time = time.time()
        generation_time = end_time - start_time
        
        logger.info(f"생성된 텍스트: {generated_text}")
        logger.info(f"생성 시간: {generation_time:.2f}초")
        
        # 생성 후 메모리 사용량 확인
        after_generation_memory = print_memory_usage()
        logger.info(f"텍스트 생성으로 인한 메모리 증가: {after_generation_memory - after_load_memory:.2f} MB")
        
        # 5. 결과 저장
        logger.info("5. 결과 저장")
        with open(os.path.join(args.output_dir, "generation_result.txt"), "w", encoding="utf-8") as f:
            f.write(f"프롬프트: {args.prompt}\n\n")
            f.write(f"생성된 텍스트: {generated_text}\n\n")
            f.write(f"생성 시간: {generation_time:.2f}초\n")
            f.write(f"토큰화 효율성: {tokenization_analysis['korean_char_per_token']:.2f} 한글 문자/토큰\n")
            f.write(f"초기 메모리 사용량: {initial_memory:.2f} MB\n")
            f.write(f"모델 로드 후 메모리 사용량: {after_load_memory:.2f} MB\n")
            f.write(f"텍스트 생성 후 메모리 사용량: {after_generation_memory:.2f} MB\n")
        
        logger.info(f"결과가 {os.path.join(args.output_dir, 'generation_result.txt')}에 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)
        return 1
    
    logger.info("한국어 특화 경량화 AI 모델 실행 완료")
    return 0

if __name__ == "__main__":
    sys.exit(main())
