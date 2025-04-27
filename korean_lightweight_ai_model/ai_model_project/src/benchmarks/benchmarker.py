"""
한국어 특화 초경량 AI 모델 - 벤치마크 모듈

이 모듈은 한국어 특화 초경량 AI 모델의 성능을 벤치마크하고 다른 모델과 비교합니다.
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Tuple

# 프로젝트 루트 디렉토리 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 모듈 임포트
from src.main import UltraLightKoreanAI
from src.utils.utils import ConfigManager, MemoryMonitor, PerformanceTracker, FileUtils

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelBenchmarker:
    """모델 벤치마크 클래스"""
    
    def __init__(self, model_path: Optional[str] = None, output_dir: str = "benchmarks"):
        """
        모델 벤치마크 초기화
        
        Args:
            model_path: 모델 경로 (None인 경우 기본 모델 사용)
            output_dir: 출력 디렉토리
        """
        self.model_path = model_path
        self.output_dir = os.path.join(project_root, output_dir)
        self.model = None
        self.memory_monitor = MemoryMonitor()
        self.performance_tracker = PerformanceTracker()
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"모델 벤치마크 초기화: {model_path if model_path else '기본 모델'}")
    
    def load_model(self) -> None:
        """모델 로드"""
        logger.info("모델 로드 시작")
        
        try:
            # 메모리 모니터링 시작
            self.memory_monitor.start_monitoring()
            
            # 성능 추적 시작
            self.performance_tracker.start_tracking()
            
            # 모델 초기화 및 로드
            self.model = UltraLightKoreanAI(config_path=self.model_path)
            self.model.load()
            
            # 성능 추적 종료
            self.performance_tracker.stop_tracking()
            
            # 로드 시간 기록
            load_time = self.performance_tracker.get_metrics().get("elapsed_time", 0)
            logger.info(f"모델 로드 완료: {load_time:.2f}초")
            
            # 메모리 사용량 통계
            memory_stats = self.memory_monitor.get_memory_usage_stats()
            for key, value in memory_stats.items():
                self.performance_tracker.add_metric(key, value)
            
            # 성능 메트릭 로깅
            self.performance_tracker.log_metrics()
            
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}", exc_info=True)
            raise
    
    def run_inference_benchmark(self, num_samples: int = 10) -> Dict[str, Any]:
        """
        추론 벤치마크 실행
        
        Args:
            num_samples: 샘플 수
        
        Returns:
            벤치마크 결과
        """
        if self.model is None:
            logger.error("모델이 로드되지 않았습니다. run_inference_benchmark() 호출 전에 load_model()을 호출하세요.")
            return {}
        
        logger.info(f"추론 벤치마크 시작: {num_samples}개 샘플")
        
        try:
            # 벤치마크 데이터 로드
            benchmark_data = self._load_benchmark_data(num_samples)
            
            # 결과 저장용 딕셔너리
            results = {
                "inference_times": [],
                "token_generation_speeds": [],
                "memory_usages": []
            }
            
            # 각 샘플에 대해 추론 실행
            for i, prompt in enumerate(benchmark_data):
                logger.info(f"샘플 {i+1}/{num_samples} 추론 시작")
                
                # 메모리 모니터링 시작
                self.memory_monitor.start_monitoring()
                
                # 성능 추적 시작
                self.performance_tracker.start_tracking()
                
                # 추론 실행
                generated_text = self.model.generate(prompt)
                
                # 성능 추적 종료
                self.performance_tracker.stop_tracking()
                
                # 추론 시간 기록
                inference_time = self.performance_tracker.get_metrics().get("elapsed_time", 0)
                results["inference_times"].append(inference_time)
                
                # 토큰 생성 속도 계산 (토큰/초)
                num_tokens = len(generated_text.split())
                token_generation_speed = num_tokens / inference_time if inference_time > 0 else 0
                results["token_generation_speeds"].append(token_generation_speed)
                
                # 메모리 사용량 기록
                memory_stats = self.memory_monitor.get_memory_usage_stats()
                results["memory_usages"].append(memory_stats["process_rss_mb"])
                
                logger.info(f"샘플 {i+1}/{num_samples} 추론 완료: {inference_time:.2f}초, {token_generation_speed:.2f} 토큰/초, {memory_stats['process_rss_mb']:.2f} MB")
            
            # 평균 계산
            avg_inference_time = np.mean(results["inference_times"])
            avg_token_generation_speed = np.mean(results["token_generation_speeds"])
            avg_memory_usage = np.mean(results["memory_usages"])
            
            # 결과 요약
            summary = {
                "avg_inference_time": avg_inference_time,
                "avg_token_generation_speed": avg_token_generation_speed,
                "avg_memory_usage": avg_memory_usage,
                "num_samples": num_samples,
                "raw_results": results
            }
            
            # 결과 저장
            self._save_benchmark_results(summary, "inference_benchmark")
            
            # 결과 시각화
            self._visualize_benchmark_results(summary, "inference_benchmark")
            
            logger.info(f"추론 벤치마크 완료: 평균 {avg_inference_time:.2f}초, {avg_token_generation_speed:.2f} 토큰/초, {avg_memory_usage:.2f} MB")
            
            return summary
            
        except Exception as e:
            logger.error(f"추론 벤치마크 중 오류 발생: {str(e)}", exc_info=True)
            return {}
    
    def run_conversation_benchmark(self, num_conversations: int = 5, turns_per_conversation: int = 3) -> Dict[str, Any]:
        """
        대화 벤치마크 실행
        
        Args:
            num_conversations: 대화 수
            turns_per_conversation: 대화당 턴 수
        
        Returns:
            벤치마크 결과
        """
        if self.model is None:
            logger.error("모델이 로드되지 않았습니다. run_conversation_benchmark() 호출 전에 load_model()을 호출하세요.")
            return {}
        
        logger.info(f"대화 벤치마크 시작: {num_conversations}개 대화, 각 {turns_per_conversation}턴")
        
        try:
            # 벤치마크 데이터 로드
            conversation_starters = self._load_conversation_starters(num_conversations)
            
            # 결과 저장용 딕셔너리
            results = {
                "response_times": [],
                "response_lengths": [],
                "memory_usages": [],
                "conversations": []
            }
            
            # 각 대화에 대해 벤치마크 실행
            for i, starter in enumerate(conversation_starters):
                logger.info(f"대화 {i+1}/{num_conversations} 시작")
                
                # 대화 기록
                conversation = []
                
                # 첫 번째 사용자 메시지
                user_message = starter
                conversation.append({"role": "user", "message": user_message})
                
                # 대화 턴 반복
                for turn in range(turns_per_conversation):
                    logger.info(f"대화 {i+1}/{num_conversations}, 턴 {turn+1}/{turns_per_conversation}")
                    
                    # 메모리 모니터링 시작
                    self.memory_monitor.start_monitoring()
                    
                    # 성능 추적 시작
                    self.performance_tracker.start_tracking()
                    
                    # 응답 생성
                    response = self.model.chat(user_message)
                    
                    # 성능 추적 종료
                    self.performance_tracker.stop_tracking()
                    
                    # 응답 시간 기록
                    response_time = self.performance_tracker.get_metrics().get("elapsed_time", 0)
                    results["response_times"].append(response_time)
                    
                    # 응답 길이 기록
                    response_length = len(response.split())
                    results["response_lengths"].append(response_length)
                    
                    # 메모리 사용량 기록
                    memory_stats = self.memory_monitor.get_memory_usage_stats()
                    results["memory_usages"].append(memory_stats["process_rss_mb"])
                    
                    # 대화 기록에 추가
                    conversation.append({"role": "assistant", "message": response})
                    
                    # 다음 사용자 메시지 생성 (간단한 예시)
                    if turn < turns_per_conversation - 1:
                        user_message = self._generate_follow_up_question(response)
                        conversation.append({"role": "user", "message": user_message})
                    
                    logger.info(f"턴 {turn+1}/{turns_per_conversation} 완료: {response_time:.2f}초, {response_length} 토큰, {memory_stats['process_rss_mb']:.2f} MB")
                
                # 대화 기록 저장
                results["conversations"].append(conversation)
            
            # 평균 계산
            avg_response_time = np.mean(results["response_times"])
            avg_response_length = np.mean(results["response_lengths"])
            avg_memory_usage = np.mean(results["memory_usages"])
            
            # 결과 요약
            summary = {
                "avg_response_time": avg_response_time,
                "avg_response_length": avg_response_length,
                "avg_memory_usage": avg_memory_usage,
                "num_conversations": num_conversations,
                "turns_per_conversation": turns_per_conversation,
                "raw_results": results
            }
            
            # 결과 저장
            self._save_benchmark_results(summary, "conversation_benchmark")
            
            # 결과 시각화
            self._visualize_benchmark_results(summary, "conversation_benchmark")
            
            logger.info(f"대화 벤치마크 완료: 평균 {avg_response_time:.2f}초, {avg_response_length:.2f} 토큰, {avg_memory_usage:.2f} MB")
            
            return summary
            
        except Exception as e:
            logger.error(f"대화 벤치마크 중 오류 발생: {str(e)}", exc_info=True)
            return {}
    
    def run_memory_benchmark(self, max_tokens: int = 1000, step: int = 100) -> Dict[str, Any]:
        """
        메모리 벤치마크 실행
        
        Args:
            max_tokens: 최대 토큰 수
            step: 단계별 토큰 수 증가량
        
        Returns:
            벤치마크 결과
        """
        if self.model is None:
            logger.error("모델이 로드되지 않았습니다. run_memory_benchmark() 호출 전에 load_model()을 호출하세요.")
            return {}
        
        logger.info(f"메모리 벤치마크 시작: 최대 {max_tokens}토큰, 단계 {step}토큰")
        
        try:
            # 결과 저장용 딕셔너리
            results = {
                "token_counts": [],
                "memory_usages": [],
                "inference_times": []
            }
            
            # 기본 프롬프트
            base_prompt = "한국어 특화 초경량 AI 모델에 대해 설명해주세요."
            
            # 토큰 수를 늘려가며 벤치마크 실행
            for tokens in range(step, max_tokens + step, step):
                logger.info(f"토큰 수 {tokens} 벤치마크 시작")
                
                # 메모리 모니터링 시작
                self.memory_monitor.start_monitoring()
                
                # 성능 추적 시작
                self.performance_tracker.start_tracking()
                
                # 추론 실행
                generated_text = self.model.generate(
                    base_prompt,
                    max_new_tokens=tokens
                )
                
                # 성능 추적 종료
                self.performance_tracker.stop_tracking()
                
                # 추론 시간 기록
                inference_time = self.performance_tracker.get_metrics().get("elapsed_time", 0)
                results["inference_times"].append(inference_time)
                
                # 실제 생성된 토큰 수 계산
                actual_tokens = len(generated_text.split())
                results["token_counts"].append(actual_tokens)
                
                # 메모리 사용량 기록
                memory_stats = self.memory_monitor.get_memory_usage_stats()
                results["memory_usages"].append(memory_stats["process_rss_mb"])
                
                logger.info(f"토큰 수 {tokens} 벤치마크 완료: {actual_tokens} 토큰, {memory_stats['process_rss_mb']:.2f} MB, {inference_time:.2f}초")
            
            # 결과 요약
            summary = {
                "token_counts": results["token_counts"],
                "memory_usages": results["memory_usages"],
                "inference_times": results["inference_times"],
                "max_tokens": max_tokens,
                "step": step
            }
            
            # 결과 저장
            self._save_benchmark_results(summary, "memory_benchmark")
            
            # 결과 시각화
            self._visualize_memory_benchmark(summary)
            
            logger.info("메모리 벤치마크 완료")
            
            return summary
            
        except Exception as e:
            logger.error(f"메모리 벤치마크 중 오류 발생: {str(e)}", exc_info=True)
            return {}
    
    def compare_with_other_models(self, model_names: List[str]) -> Dict[str, Any]:
        """
        다른 모델과 비교
        
        Args:
            model_names: 비교할 모델 이름 리스트
        
        Returns:
            비교 결과
        """
        logger.info(f"모델 비교 시작: {', '.join(model_names)}")
        
        try:
            # 비교 데이터
            comparison_data = {
                "model_names": ["UltraLightKorean"] + model_names,
                "memory_usages": [],
                "inference_times": [],
                "model_sizes": []
            }
            
            # 현재 모델 데이터 추가
            if self.model is not None:
                # 메모리 사용량
                memory_stats = self.memory_monitor.get_memory_usage_stats()
                comparison_data["memory_usages"].append(memory_stats["process_rss_mb"])
                
                # 추론 시간 (간단한 벤치마크)
                self.performance_tracker.start_tracking()
                self.model.generate("한국어 특화 초경량 AI 모델에 대해 간단히 설명해주세요.")
                self.performance_tracker.stop_tracking()
                inference_time = self.performance_tracker.get_metrics().get("elapsed_time", 0)
                comparison_data["inference_times"].append(inference_time)
                
                # 모델 크기
                from src.utils.utils import TorchUtils
                model_size_info = TorchUtils.get_model_size(self.model.model)
                comparison_data["model_sizes"].append(model_size_info["total_size_mb"])
            else:
                # 모델이 로드되지 않은 경우 더미 데이터 추가
                comparison_data["memory_usages"].append(0)
                comparison_data["inference_times"].append(0)
                comparison_data["model_sizes"].append(0)
            
            # 다른 모델 데이터 추가 (더미 데이터, 실제로는 각 모델을 로드하고 벤치마크해야 함)
            # 여기서는 예시 데이터만 제공
            comparison_data["memory_usages"].extend([
                1024,  # 1GB
                2048,  # 2GB
                4096   # 4GB
            ])
            
            comparison_data["inference_times"].extend([
                0.8,   # 0.8초
                0.5,   # 0.5초
                0.3    # 0.3초
            ])
            
            comparison_data["model_sizes"].extend([
                1024,  # 1GB
                2048,  # 2GB
                4096   # 4GB
            ])
            
            # 결과 저장
            self._save_benchmark_results(comparison_data, "model_comparison")
            
            # 결과 시각화
            self._visualize_model_comparison(comparison_data)
            
            logger.info("모델 비교 완료")
            
            return comparison_data
            
        except Exception as e:
            logger.error(f"모델 비교 중 오류 발생: {str(e)}", exc_info=True)
            return {}
    
    def _load_benchmark_data(self, num_samples: int) -> List[str]:
        """
        벤치마크 데이터 로드
        
        Args:
            num_samples: 샘플 수
        
        Returns:
            벤치마크 데이터 리스트
        """
        # 벤치마크 데이터 파일 경로
        benchmark_data_path = os.path.join(project_root, "src", "benchmarks", "data", "benchmark_prompts.json")
        
        # 파일이 존재하는 경우 로드
        if os.path.exists(benchmark_data_path):
            try:
                with open(benchmark_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    prompts = data.get("prompts", [])
                    return prompts[:num_samples]
            except Exception as e:
                logger.error(f"벤치마크 데이터 로드 중 오류 발생: {str(e)}", exc_info=True)
        
        # 파일이 존재하지 않는 경우 기본 데이터 생성
        default_prompts = [
            "한국어 특화 초경량 AI 모델의 장점은 무엇인가요?",
            "인공지능 기술의 발전 방향에 대해 설명해주세요.",
            "한국어 자연어 처리의 어려운 점은 무엇인가요?",
            "경량화 AI 모델이 중요한 이유는 무엇인가요?",
            "온디바이스 AI의 미래에 대해 전망해주세요.",
            "한국어와 영어의 언어적 차이점은 무엇인가요?",
            "AI 모델의 메모리 최적화 방법에 대해 설명해주세요.",
            "양자화 기법이 AI 모델에 미치는 영향은 무엇인가요?",
            "프루닝 기법을 통한 모델 경량화 방법을 설명해주세요.",
            "지식 증류 기법의 원리와 장점은 무엇인가요?"
        ]
        
        # 필요한 샘플 수만큼 반환 (부족한 경우 반복)
        result = []
        while len(result) < num_samples:
            result.extend(default_prompts[:min(num_samples - len(result), len(default_prompts))])
        
        return result
    
    def _load_conversation_starters(self, num_conversations: int) -> List[str]:
        """
        대화 시작 메시지 로드
        
        Args:
            num_conversations: 대화 수
        
        Returns:
            대화 시작 메시지 리스트
        """
        # 대화 시작 메시지 파일 경로
        conversation_starters_path = os.path.join(project_root, "src", "benchmarks", "data", "conversation_starters.json")
        
        # 파일이 존재하는 경우 로드
        if os.path.exists(conversation_starters_path):
            try:
                with open(conversation_starters_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    starters = data.get("starters", [])
                    return starters[:num_conversations]
            except Exception as e:
                logger.error(f"대화 시작 메시지 로드 중 오류 발생: {str(e)}", exc_info=True)
        
        # 파일이 존재하지 않는 경우 기본 데이터 생성
        default_starters = [
            "안녕하세요! 오늘 날씨가 어때요?",
            "인공지능에 대해 알려주세요.",
            "한국어 학습에 좋은 방법이 있을까요?",
            "요즘 가장 인기있는 영화는 무엇인가요?",
            "건강을 유지하는 좋은 방법이 있을까요?",
            "프로그래밍을 배우고 싶은데 어떻게 시작해야 할까요?",
            "여행 계획을 세우는 좋은 방법이 있을까요?",
            "책 추천 좀 해주세요.",
            "요리를 잘하는 비결이 있을까요?",
            "취미로 할 만한 것들을 추천해주세요."
        ]
        
        # 필요한 대화 수만큼 반환 (부족한 경우 반복)
        result = []
        while len(result) < num_conversations:
            result.extend(default_starters[:min(num_conversations - len(result), len(default_starters))])
        
        return result
    
    def _generate_follow_up_question(self, response: str) -> str:
        """
        후속 질문 생성
        
        Args:
            response: 이전 응답
        
        Returns:
            후속 질문
        """
        # 간단한 후속 질문 생성 로직 (실제로는 더 복잡한 로직이 필요할 수 있음)
        follow_up_templates = [
            "그것에 대해 더 자세히 설명해주세요.",
            "왜 그렇게 생각하시나요?",
            "다른 관점에서는 어떻게 볼 수 있을까요?",
            "구체적인 예시를 들어주실 수 있나요?",
            "그 외에 다른 의견이 있으신가요?"
        ]
        
        import random
        return random.choice(follow_up_templates)
    
    def _save_benchmark_results(self, results: Dict[str, Any], benchmark_type: str) -> None:
        """
        벤치마크 결과 저장
        
        Args:
            results: 벤치마크 결과
            benchmark_type: 벤치마크 유형
        """
        # 결과 파일 경로
        results_path = os.path.join(self.output_dir, f"{benchmark_type}_results.json")
        
        try:
            # 결과 저장
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"벤치마크 결과 저장 완료: {results_path}")
        except Exception as e:
            logger.error(f"벤치마크 결과 저장 중 오류 발생: {str(e)}", exc_info=True)
    
    def _visualize_benchmark_results(self, results: Dict[str, Any], benchmark_type: str) -> None:
        """
        벤치마크 결과 시각화
        
        Args:
            results: 벤치마크 결과
            benchmark_type: 벤치마크 유형
        """
        try:
            # 그래프 생성
            plt.figure(figsize=(12, 8))
            
            if benchmark_type == "inference_benchmark":
                # 추론 시간 그래프
                plt.subplot(2, 2, 1)
                plt.plot(results["raw_results"]["inference_times"])
                plt.title("추론 시간 (초)")
                plt.xlabel("샘플 번호")
                plt.ylabel("시간 (초)")
                plt.grid(True)
                
                # 토큰 생성 속도 그래프
                plt.subplot(2, 2, 2)
                plt.plot(results["raw_results"]["token_generation_speeds"])
                plt.title("토큰 생성 속도 (토큰/초)")
                plt.xlabel("샘플 번호")
                plt.ylabel("속도 (토큰/초)")
                plt.grid(True)
                
                # 메모리 사용량 그래프
                plt.subplot(2, 2, 3)
                plt.plot(results["raw_results"]["memory_usages"])
                plt.title("메모리 사용량 (MB)")
                plt.xlabel("샘플 번호")
                plt.ylabel("메모리 (MB)")
                plt.grid(True)
                
                # 평균 결과 그래프
                plt.subplot(2, 2, 4)
                avg_data = [results["avg_inference_time"], results["avg_token_generation_speed"], results["avg_memory_usage"]]
                labels = ["평균 추론 시간 (초)", "평균 토큰 생성 속도 (토큰/초)", "평균 메모리 사용량 (MB)"]
                plt.bar(labels, avg_data)
                plt.title("평균 결과")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                
            elif benchmark_type == "conversation_benchmark":
                # 응답 시간 그래프
                plt.subplot(2, 2, 1)
                plt.plot(results["raw_results"]["response_times"])
                plt.title("응답 시간 (초)")
                plt.xlabel("턴 번호")
                plt.ylabel("시간 (초)")
                plt.grid(True)
                
                # 응답 길이 그래프
                plt.subplot(2, 2, 2)
                plt.plot(results["raw_results"]["response_lengths"])
                plt.title("응답 길이 (토큰)")
                plt.xlabel("턴 번호")
                plt.ylabel("길이 (토큰)")
                plt.grid(True)
                
                # 메모리 사용량 그래프
                plt.subplot(2, 2, 3)
                plt.plot(results["raw_results"]["memory_usages"])
                plt.title("메모리 사용량 (MB)")
                plt.xlabel("턴 번호")
                plt.ylabel("메모리 (MB)")
                plt.grid(True)
                
                # 평균 결과 그래프
                plt.subplot(2, 2, 4)
                avg_data = [results["avg_response_time"], results["avg_response_length"], results["avg_memory_usage"]]
                labels = ["평균 응답 시간 (초)", "평균 응답 길이 (토큰)", "평균 메모리 사용량 (MB)"]
                plt.bar(labels, avg_data)
                plt.title("평균 결과")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
            
            # 그래프 저장
            plt.savefig(os.path.join(self.output_dir, f"{benchmark_type}_results.png"))
            logger.info(f"벤치마크 결과 시각화 완료: {os.path.join(self.output_dir, f'{benchmark_type}_results.png')}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"벤치마크 결과 시각화 중 오류 발생: {str(e)}", exc_info=True)
    
    def _visualize_memory_benchmark(self, results: Dict[str, Any]) -> None:
        """
        메모리 벤치마크 결과 시각화
        
        Args:
            results: 메모리 벤치마크 결과
        """
        try:
            # 그래프 생성
            plt.figure(figsize=(12, 8))
            
            # 토큰 수에 따른 메모리 사용량 그래프
            plt.subplot(2, 1, 1)
            plt.plot(results["token_counts"], results["memory_usages"], marker='o')
            plt.title("토큰 수에 따른 메모리 사용량")
            plt.xlabel("토큰 수")
            plt.ylabel("메모리 사용량 (MB)")
            plt.grid(True)
            
            # 토큰 수에 따른 추론 시간 그래프
            plt.subplot(2, 1, 2)
            plt.plot(results["token_counts"], results["inference_times"], marker='o')
            plt.title("토큰 수에 따른 추론 시간")
            plt.xlabel("토큰 수")
            plt.ylabel("추론 시간 (초)")
            plt.grid(True)
            
            plt.tight_layout()
            
            # 그래프 저장
            plt.savefig(os.path.join(self.output_dir, "memory_benchmark_results.png"))
            logger.info(f"메모리 벤치마크 결과 시각화 완료: {os.path.join(self.output_dir, 'memory_benchmark_results.png')}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"메모리 벤치마크 결과 시각화 중 오류 발생: {str(e)}", exc_info=True)
    
    def _visualize_model_comparison(self, results: Dict[str, Any]) -> None:
        """
        모델 비교 결과 시각화
        
        Args:
            results: 모델 비교 결과
        """
        try:
            # 그래프 생성
            plt.figure(figsize=(15, 10))
            
            # 메모리 사용량 비교 그래프
            plt.subplot(2, 2, 1)
            plt.bar(results["model_names"], results["memory_usages"])
            plt.title("메모리 사용량 비교 (MB)")
            plt.xlabel("모델")
            plt.ylabel("메모리 사용량 (MB)")
            plt.xticks(rotation=45, ha="right")
            
            # 추론 시간 비교 그래프
            plt.subplot(2, 2, 2)
            plt.bar(results["model_names"], results["inference_times"])
            plt.title("추론 시간 비교 (초)")
            plt.xlabel("모델")
            plt.ylabel("추론 시간 (초)")
            plt.xticks(rotation=45, ha="right")
            
            # 모델 크기 비교 그래프
            plt.subplot(2, 2, 3)
            plt.bar(results["model_names"], results["model_sizes"])
            plt.title("모델 크기 비교 (MB)")
            plt.xlabel("모델")
            plt.ylabel("모델 크기 (MB)")
            plt.xticks(rotation=45, ha="right")
            
            # 메모리 효율성 비교 그래프 (모델 크기 대비 메모리 사용량)
            plt.subplot(2, 2, 4)
            efficiency = [m / s if s > 0 else 0 for m, s in zip(results["memory_usages"], results["model_sizes"])]
            plt.bar(results["model_names"], efficiency)
            plt.title("메모리 효율성 비교 (메모리 사용량 / 모델 크기)")
            plt.xlabel("모델")
            plt.ylabel("메모리 효율성")
            plt.xticks(rotation=45, ha="right")
            
            plt.tight_layout()
            
            # 그래프 저장
            plt.savefig(os.path.join(self.output_dir, "model_comparison_results.png"))
            logger.info(f"모델 비교 결과 시각화 완료: {os.path.join(self.output_dir, 'model_comparison_results.png')}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"모델 비교 결과 시각화 중 오류 발생: {str(e)}", exc_info=True)

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="한국어 특화 초경량 AI 모델 벤치마크")
    parser.add_argument("--model", type=str, help="모델 경로")
    parser.add_argument("--output", type=str, default="benchmarks", help="출력 디렉토리")
    parser.add_argument("--inference", action="store_true", help="추론 벤치마크 실행")
    parser.add_argument("--conversation", action="store_true", help="대화 벤치마크 실행")
    parser.add_argument("--memory", action="store_true", help="메모리 벤치마크 실행")
    parser.add_argument("--compare", action="store_true", help="모델 비교 실행")
    parser.add_argument("--all", action="store_true", help="모든 벤치마크 실행")
    args = parser.parse_args()
    
    # 벤치마크 초기화
    benchmarker = ModelBenchmarker(model_path=args.model, output_dir=args.output)
    
    # 모델 로드
    benchmarker.load_model()
    
    # 벤치마크 실행
    if args.inference or args.all:
        benchmarker.run_inference_benchmark()
    
    if args.conversation or args.all:
        benchmarker.run_conversation_benchmark()
    
    if args.memory or args.all:
        benchmarker.run_memory_benchmark()
    
    if args.compare or args.all:
        benchmarker.compare_with_other_models(["Model1", "Model2", "Model3"])

if __name__ == "__main__":
    main()
