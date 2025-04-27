"""
한국어 특화 경량화 AI 모델 성능 평가 모듈
- 메모리 사용량 평가
- 추론 속도 평가
- 한국어 생성 품질 평가
"""

import os
import time
import json
import logging
import argparse
import torch
import numpy as np
import psutil
from typing import Dict, List, Optional, Tuple, Union

from model_loader import KoreanLightweightModel
from inference_optimizer import InferenceOptimizer
from tokenizer_optimizer import KoreanTokenizerOptimizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_evaluation.log")
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """모델 성능 평가 클래스"""
    
    def __init__(self, model_name, use_4bit=True, use_sharding=True, max_memory_gb=2.5):
        """
        모델 평가 초기화
        
        Args:
            model_name: 모델 이름 (Hugging Face 모델 ID)
            use_4bit: 4비트 양자화 사용 여부
            use_sharding: 모델 샤딩 사용 여부
            max_memory_gb: 최대 허용 메모리 (GB)
        """
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.use_sharding = use_sharding
        self.max_memory_gb = max_memory_gb
        
        self.model_loader = None
        self.inference_optimizer = None
        self.tokenizer_optimizer = None
        
        self.model = None
        self.tokenizer = None
        
        self.evaluation_results = {}
        
        logger.info(f"모델 평가 초기화: {model_name}, 4비트 양자화={use_4bit}, 샤딩={use_sharding}")
    
    def setup(self):
        """모델 및 최적화 설정"""
        # 토크나이저 최적화
        self.tokenizer_optimizer = KoreanTokenizerOptimizer(self.model_name)
        self.tokenizer = self.tokenizer_optimizer.load_tokenizer()
        
        # 모델 로드
        self.model_loader = KoreanLightweightModel(
            model_name=self.model_name,
            use_4bit=self.use_4bit,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.model, _ = self.model_loader.load_model(use_sharding=self.use_sharding)
        
        # 추론 최적화
        self.inference_optimizer = InferenceOptimizer(
            max_memory_gb=self.max_memory_gb,
            max_cache_size=512
        )
        
        # 추론을 위한 모델 최적화
        self.model_loader.optimize_for_inference()
        
        logger.info("모델 및 최적화 설정 완료")
    
    def evaluate_memory_usage(self, prompts):
        """
        메모리 사용량 평가
        
        Args:
            prompts: 평가할 프롬프트 리스트
        
        Returns:
            메모리 사용량 평가 결과 딕셔너리
        """
        logger.info("메모리 사용량 평가 시작")
        
        memory_results = {
            "initial_memory_mb": 0,
            "peak_memory_mb": 0,
            "average_memory_mb": [],
            "memory_per_token": []
        }
        
        # 초기 메모리 사용량
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)
        memory_results["initial_memory_mb"] = initial_memory
        
        peak_memory = initial_memory
        
        for i, prompt in enumerate(prompts):
            logger.info(f"프롬프트 {i+1}/{len(prompts)} 평가 중")
            
            # 텍스트 생성
            _ = self.model_loader.generate_text(prompt, max_length=128)
            
            # 현재 메모리 사용량
            current_memory = process.memory_info().rss / (1024 * 1024)
            memory_results["average_memory_mb"].append(current_memory)
            
            # 토큰 당 메모리 사용량
            tokens = self.tokenizer.tokenize(prompt)
            memory_per_token = (current_memory - initial_memory) / len(tokens) if tokens else 0
            memory_results["memory_per_token"].append(memory_per_token)
            
            # 최대 메모리 사용량 업데이트
            peak_memory = max(peak_memory, current_memory)
            
            logger.info(f"프롬프트 {i+1} 메모리 사용량: {current_memory:.2f} MB, 토큰 당: {memory_per_token:.2f} MB")
        
        memory_results["peak_memory_mb"] = peak_memory
        memory_results["average_memory_mb"] = np.mean(memory_results["average_memory_mb"])
        memory_results["memory_per_token"] = np.mean(memory_results["memory_per_token"])
        
        logger.info(f"메모리 사용량 평가 결과: 초기={memory_results['initial_memory_mb']:.2f} MB, "
                   f"최대={memory_results['peak_memory_mb']:.2f} MB, "
                   f"평균={memory_results['average_memory_mb']:.2f} MB, "
                   f"토큰 당={memory_results['memory_per_token']:.2f} MB")
        
        self.evaluation_results["memory_usage"] = memory_results
        return memory_results
    
    def evaluate_inference_speed(self, prompts, max_length=128):
        """
        추론 속도 평가
        
        Args:
            prompts: 평가할 프롬프트 리스트
            max_length: 최대 생성 길이
        
        Returns:
            추론 속도 평가 결과 딕셔너리
        """
        logger.info("추론 속도 평가 시작")
        
        speed_results = {
            "total_time_seconds": 0,
            "tokens_per_second": [],
            "latency_seconds": []
        }
        
        total_tokens = 0
        total_time = 0
        
        for i, prompt in enumerate(prompts):
            logger.info(f"프롬프트 {i+1}/{len(prompts)} 평가 중")
            
            # 시간 측정 시작
            start_time = time.time()
            
            # 텍스트 생성
            generated_text = self.model_loader.generate_text(prompt, max_length=max_length)
            
            # 시간 측정 종료
            end_time = time.time()
            generation_time = end_time - start_time
            
            # 생성된 토큰 수
            input_tokens = self.tokenizer.tokenize(prompt)
            output_tokens = self.tokenizer.tokenize(generated_text)
            generated_tokens = len(output_tokens) - len(input_tokens)
            
            # 토큰 당 생성 시간
            tokens_per_second = generated_tokens / generation_time if generation_time > 0 else 0
            
            speed_results["tokens_per_second"].append(tokens_per_second)
            speed_results["latency_seconds"].append(generation_time)
            
            total_tokens += generated_tokens
            total_time += generation_time
            
            logger.info(f"프롬프트 {i+1} 생성 시간: {generation_time:.2f}초, "
                       f"생성된 토큰: {generated_tokens}, "
                       f"초당 토큰: {tokens_per_second:.2f}")
        
        speed_results["total_time_seconds"] = total_time
        speed_results["average_tokens_per_second"] = np.mean(speed_results["tokens_per_second"])
        speed_results["average_latency_seconds"] = np.mean(speed_results["latency_seconds"])
        
        logger.info(f"추론 속도 평가 결과: 총 시간={speed_results['total_time_seconds']:.2f}초, "
                   f"평균 초당 토큰={speed_results['average_tokens_per_second']:.2f}, "
                   f"평균 지연 시간={speed_results['average_latency_seconds']:.2f}초")
        
        self.evaluation_results["inference_speed"] = speed_results
        return speed_results
    
    def evaluate_korean_quality(self, prompts, reference_texts=None):
        """
        한국어 생성 품질 평가
        
        Args:
            prompts: 평가할 프롬프트 리스트
            reference_texts: 참조 텍스트 리스트 (있는 경우)
        
        Returns:
            한국어 생성 품질 평가 결과 딕셔너리
        """
        logger.info("한국어 생성 품질 평가 시작")
        
        quality_results = {
            "generated_texts": [],
            "korean_char_ratio": [],
            "jamo_separation_ratio": [],
            "tokenization_efficiency": []
        }
        
        for i, prompt in enumerate(prompts):
            logger.info(f"프롬프트 {i+1}/{len(prompts)} 평가 중")
            
            # 텍스트 생성
            generated_text = self.model_loader.generate_text(prompt, max_length=128)
            quality_results["generated_texts"].append(generated_text)
            
            # 한글 문자 비율
            import re
            total_chars = len(generated_text)
            korean_chars = len(re.findall(r'[가-힣]', generated_text))
            korean_char_ratio = korean_chars / total_chars if total_chars > 0 else 0
            quality_results["korean_char_ratio"].append(korean_char_ratio)
            
            # 자모 분리 비율
            jamo_chars = len(re.findall(r'[ㄱ-ㅎㅏ-ㅣ]', generated_text))
            jamo_separation_ratio = jamo_chars / total_chars if total_chars > 0 else 0
            quality_results["jamo_separation_ratio"].append(jamo_separation_ratio)
            
            # 토큰화 효율성
            tokenization_analysis = self.tokenizer_optimizer.analyze_tokenization_efficiency(generated_text)
            quality_results["tokenization_efficiency"].append(tokenization_analysis["korean_char_per_token"])
            
            logger.info(f"프롬프트 {i+1} 한글 문자 비율: {korean_char_ratio:.2f}, "
                       f"자모 분리 비율: {jamo_separation_ratio:.2f}, "
                       f"토큰화 효율성: {tokenization_analysis['korean_char_per_token']:.2f}")
        
        # 평균 계산
        quality_results["average_korean_char_ratio"] = np.mean(quality_results["korean_char_ratio"])
        quality_results["average_jamo_separation_ratio"] = np.mean(quality_results["jamo_separation_ratio"])
        quality_results["average_tokenization_efficiency"] = np.mean(quality_results["tokenization_efficiency"])
        
        logger.info(f"한국어 생성 품질 평가 결과: "
                   f"평균 한글 문자 비율={quality_results['average_korean_char_ratio']:.2f}, "
                   f"평균 자모 분리 비율={quality_results['average_jamo_separation_ratio']:.2f}, "
                   f"평균 토큰화 효율성={quality_results['average_tokenization_efficiency']:.2f}")
        
        self.evaluation_results["korean_quality"] = quality_results
        return quality_results
    
    def run_full_evaluation(self, test_prompts, output_dir="./evaluation_results"):
        """
        전체 평가 실행
        
        Args:
            test_prompts: 테스트 프롬프트 리스트
            output_dir: 결과 저장 디렉토리
        
        Returns:
            전체 평가 결과 딕셔너리
        """
        logger.info("전체 모델 평가 시작")
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 모델 설정
        self.setup()
        
        # 메모리 사용량 평가
        memory_results = self.evaluate_memory_usage(test_prompts)
        
        # 추론 속도 평가
        speed_results = self.evaluate_inference_speed(test_prompts)
        
        # 한국어 생성 품질 평가
        quality_results = self.evaluate_korean_quality(test_prompts)
        
        # 결과 저장
        with open(os.path.join(output_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
            json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)
        
        # 생성된 텍스트 저장
        with open(os.path.join(output_dir, "generated_texts.txt"), "w", encoding="utf-8") as f:
            for i, (prompt, text) in enumerate(zip(test_prompts, quality_results["generated_texts"])):
                f.write(f"프롬프트 {i+1}: {prompt}\n\n")
                f.write(f"생성된 텍스트: {text}\n\n")
                f.write("-" * 80 + "\n\n")
        
        logger.info(f"평가 결과가 {output_dir}에 저장되었습니다.")
        
        # 요약 결과
        summary = {
            "model_name": self.model_name,
            "use_4bit": self.use_4bit,
            "use_sharding": self.use_sharding,
            "max_memory_gb": self.max_memory_gb,
            "peak_memory_mb": memory_results["peak_memory_mb"],
            "average_tokens_per_second": speed_results["average_tokens_per_second"],
            "average_latency_seconds": speed_results["average_latency_seconds"],
            "average_korean_char_ratio": quality_results["average_korean_char_ratio"],
            "average_tokenization_efficiency": quality_results["average_tokenization_efficiency"]
        }
        
        logger.info("전체 모델 평가 완료")
        logger.info(f"요약: 최대 메모리={summary['peak_memory_mb']:.2f}MB, "
                   f"초당 토큰={summary['average_tokens_per_second']:.2f}, "
                   f"한글 비율={summary['average_korean_char_ratio']:.2f}")
        
        return summary

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="한국어 특화 경량화 AI 모델 평가")
    
    parser.add_argument("--model_name", type=str, default="EleutherAI/polyglot-ko-1.3b",
                        help="평가할 모델 이름 (Hugging Face 모델 ID)")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="4비트 양자화 사용 여부")
    parser.add_argument("--use_sharding", action="store_true", default=True,
                        help="모델 샤딩 사용 여부")
    parser.add_argument("--max_memory_gb", type=float, default=2.5,
                        help="최대 허용 메모리 (GB)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="결과 저장 디렉토리")
    
    args = parser.parse_args()
    
    # 테스트 프롬프트
    test_prompts = [
        "안녕하세요, 오늘 날씨가 어떤가요?",
        "인공지능 기술의 발전 방향에 대해 설명해주세요.",
        "한국의 전통 음식 중에서 외국인들에게 가장 인기 있는 것은 무엇인가요?",
        "프로그래밍 언어 중에서 초보자가 배우기 좋은 언어는 무엇인가요?",
        "지구 온난화가 생태계에 미치는 영향에 대해 설명해주세요."
    ]
    
    # 모델 평가
    evaluator = ModelEvaluator(
        model_name=args.model_name,
        use_4bit=args.use_4bit,
        use_sharding=args.use_sharding,
        max_memory_gb=args.max_memory_gb
    )
    
    evaluator.run_full_evaluation(test_prompts, args.output_dir)

if __name__ == "__main__":
    main()
