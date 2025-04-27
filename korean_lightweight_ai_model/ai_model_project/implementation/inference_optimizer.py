"""
한국어 특화 경량화 AI 모델 추론 최적화 모듈
- 메모리 사용량 최적화
- KV 캐시 관리
- 모델 샤딩 및 메모리 매핑
"""

import os
import gc
import torch
import psutil
import logging
from typing import Dict, List, Optional, Tuple, Union

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """메모리 사용량 최적화 및 모니터링 클래스"""
    
    def __init__(self, max_memory_gb=3.0):
        """
        메모리 최적화 초기화
        
        Args:
            max_memory_gb: 최대 허용 메모리 (GB)
        """
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.initial_memory = self._get_memory_usage()
        logger.info(f"메모리 최적화 초기화: 최대 허용 메모리 {max_memory_gb}GB")
    
    def _get_memory_usage(self) -> int:
        """현재 메모리 사용량 반환 (바이트)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    
    def get_memory_usage_mb(self) -> float:
        """현재 메모리 사용량 반환 (MB)"""
        current_memory = self._get_memory_usage()
        return (current_memory - self.initial_memory) / (1024 * 1024)
    
    def get_memory_usage_percentage(self) -> float:
        """최대 허용 메모리 대비 현재 사용량 비율 반환"""
        current_memory = self._get_memory_usage()
        memory_used = current_memory - self.initial_memory
        return (memory_used / self.max_memory_bytes) * 100
    
    def is_memory_critical(self, threshold_percentage=90) -> bool:
        """메모리 사용량이 임계치를 초과하는지 확인"""
        usage_percentage = self.get_memory_usage_percentage()
        return usage_percentage > threshold_percentage
    
    def clear_cache(self):
        """메모리 캐시 정리"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("메모리 캐시 정리 완료")
    
    def log_memory_status(self):
        """현재 메모리 상태 로깅"""
        usage_mb = self.get_memory_usage_mb()
        usage_percentage = self.get_memory_usage_percentage()
        logger.info(f"메모리 사용량: {usage_mb:.2f}MB ({usage_percentage:.2f}%)")


class KVCacheManager:
    """KV 캐시 관리 클래스"""
    
    def __init__(self, max_cache_size=512):
        """
        KV 캐시 관리자 초기화
        
        Args:
            max_cache_size: 최대 캐시 크기 (토큰 수)
        """
        self.max_cache_size = max_cache_size
        self.current_cache_size = 0
        logger.info(f"KV 캐시 관리자 초기화: 최대 캐시 크기 {max_cache_size} 토큰")
    
    def manage_kv_cache(self, model, input_length):
        """
        KV 캐시 관리
        
        Args:
            model: 모델 객체
            input_length: 입력 시퀀스 길이
        """
        # 캐시 크기 업데이트
        self.current_cache_size += input_length
        
        # 캐시 크기가 최대값을 초과하면 캐시 정리
        if self.current_cache_size > self.max_cache_size:
            self._clear_kv_cache(model)
            logger.info(f"KV 캐시 정리: 크기 초과 ({self.current_cache_size} > {self.max_cache_size})")
            self.current_cache_size = input_length
    
    def _clear_kv_cache(self, model):
        """
        모델의 KV 캐시 정리
        
        Args:
            model: 모델 객체
        """
        # 모델의 past_key_values 초기화
        if hasattr(model, "past_key_values"):
            model.past_key_values = None
        
        # 캐시 정리를 위한 가비지 컬렉션 실행
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ModelShardingManager:
    """모델 샤딩 및 메모리 매핑 관리 클래스"""
    
    def __init__(self, model_dir, num_shards=4):
        """
        모델 샤딩 관리자 초기화
        
        Args:
            model_dir: 모델 디렉토리 경로
            num_shards: 샤드 수
        """
        self.model_dir = model_dir
        self.num_shards = num_shards
        self.current_shard = None
        self.shard_mapping = {}
        logger.info(f"모델 샤딩 관리자 초기화: 샤드 수 {num_shards}")
    
    def prepare_model_sharding(self, model):
        """
        모델 샤딩 준비
        
        Args:
            model: 모델 객체
        
        Returns:
            샤딩된 모델
        """
        # 모델 레이어를 샤드로 분할
        layers = model.transformer.h if hasattr(model, "transformer") else model.model.layers
        num_layers = len(layers)
        
        # 레이어를 샤드로 그룹화
        layers_per_shard = num_layers // self.num_shards
        for i in range(self.num_shards):
            start_idx = i * layers_per_shard
            end_idx = (i + 1) * layers_per_shard if i < self.num_shards - 1 else num_layers
            self.shard_mapping[i] = (start_idx, end_idx)
        
        logger.info(f"모델 샤딩 준비 완료: {self.shard_mapping}")
        return model
    
    def load_shard(self, model, shard_idx):
        """
        특정 샤드 로드
        
        Args:
            model: 모델 객체
            shard_idx: 로드할 샤드 인덱스
        
        Returns:
            업데이트된 모델
        """
        if shard_idx not in self.shard_mapping:
            logger.error(f"유효하지 않은 샤드 인덱스: {shard_idx}")
            return model
        
        # 현재 샤드가 이미 로드되어 있으면 스킵
        if self.current_shard == shard_idx:
            return model
        
        # 이전 샤드 언로드
        if self.current_shard is not None:
            self._unload_shard(model, self.current_shard)
        
        # 새 샤드 로드
        start_idx, end_idx = self.shard_mapping[shard_idx]
        layers = model.transformer.h if hasattr(model, "transformer") else model.model.layers
        
        # 메모리 매핑을 사용하여 샤드 로드
        for i in range(start_idx, end_idx):
            # 실제 구현에서는 메모리 매핑 로직 추가
            pass
        
        self.current_shard = shard_idx
        logger.info(f"샤드 {shard_idx} 로드 완료 (레이어 {start_idx}-{end_idx-1})")
        return model
    
    def _unload_shard(self, model, shard_idx):
        """
        특정 샤드 언로드
        
        Args:
            model: 모델 객체
            shard_idx: 언로드할 샤드 인덱스
        """
        start_idx, end_idx = self.shard_mapping[shard_idx]
        layers = model.transformer.h if hasattr(model, "transformer") else model.model.layers
        
        # 메모리에서 샤드 언로드
        for i in range(start_idx, end_idx):
            # 실제 구현에서는 언로드 로직 추가
            pass
        
        # 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"샤드 {shard_idx} 언로드 완료")


class InferenceOptimizer:
    """추론 최적화 클래스"""
    
    def __init__(self, max_memory_gb=3.0, max_cache_size=512, num_shards=4):
        """
        추론 최적화 초기화
        
        Args:
            max_memory_gb: 최대 허용 메모리 (GB)
            max_cache_size: 최대 KV 캐시 크기 (토큰 수)
            num_shards: 모델 샤드 수
        """
        self.memory_optimizer = MemoryOptimizer(max_memory_gb)
        self.kv_cache_manager = KVCacheManager(max_cache_size)
        self.model_sharding_manager = None  # 모델 로드 후 초기화
        
        logger.info("추론 최적화 초기화 완료")
    
    def setup_model_sharding(self, model, model_dir):
        """
        모델 샤딩 설정
        
        Args:
            model: 모델 객체
            model_dir: 모델 디렉토리 경로
        
        Returns:
            샤딩 준비된 모델
        """
        self.model_sharding_manager = ModelShardingManager(model_dir, num_shards=4)
        return self.model_sharding_manager.prepare_model_sharding(model)
    
    def optimize_inference(self, model, input_ids):
        """
        추론 최적화 적용
        
        Args:
            model: 모델 객체
            input_ids: 입력 토큰 ID
        
        Returns:
            최적화된 모델
        """
        # 메모리 상태 로깅
        self.memory_optimizer.log_memory_status()
        
        # KV 캐시 관리
        input_length = input_ids.shape[1]
        self.kv_cache_manager.manage_kv_cache(model, input_length)
        
        # 메모리가 임계치에 도달하면 캐시 정리
        if self.memory_optimizer.is_memory_critical():
            logger.warning("메모리 사용량 임계치 초과, 캐시 정리 실행")
            self.memory_optimizer.clear_cache()
        
        return model
    
    def get_memory_status(self):
        """
        현재 메모리 상태 반환
        
        Returns:
            메모리 사용량 정보 딕셔너리
        """
        return {
            "usage_mb": self.memory_optimizer.get_memory_usage_mb(),
            "usage_percentage": self.memory_optimizer.get_memory_usage_percentage(),
            "is_critical": self.memory_optimizer.is_memory_critical()
        }
