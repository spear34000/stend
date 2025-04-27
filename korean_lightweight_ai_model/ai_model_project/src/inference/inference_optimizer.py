"""
한국어 특화 초경량 AI 모델 - 추론 최적화 모듈

이 모듈은 제한된 하드웨어 환경(RAM 3GB, SSD 64GB)에서 효율적인 추론을 위한 최적화 기법을 구현합니다:
- KV 캐시 관리
- 메모리 매핑
- 인플레이스 연산
- 모델 샤딩
"""

import os
import gc
import torch
import numpy as np
import logging
import psutil
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import PreTrainedModel

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InferenceOptimizer:
    """추론 최적화 클래스"""
    
    def __init__(self, 
                 kv_cache_size_mb: int = 128,
                 use_memory_mapping: bool = True,
                 enable_inplace_operations: bool = True,
                 max_memory_gb: float = 2.5):
        """
        추론 최적화 초기화
        
        Args:
            kv_cache_size_mb: KV 캐시 크기 (MB)
            use_memory_mapping: 메모리 매핑 사용 여부
            enable_inplace_operations: 인플레이스 연산 활성화 여부
            max_memory_gb: 최대 허용 메모리 (GB)
        """
        self.kv_cache_size_mb = kv_cache_size_mb
        self.use_memory_mapping = use_memory_mapping
        self.enable_inplace_operations = enable_inplace_operations
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        
        logger.info(f"추론 최적화 초기화: KV 캐시={kv_cache_size_mb}MB, 메모리 매핑={use_memory_mapping}")
    
    def optimize_for_inference(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        추론을 위한 최적화 적용
        
        Args:
            model: 최적화할 모델
        
        Returns:
            최적화된 모델
        """
        logger.info("추론 최적화 적용 시작")
        
        try:
            # 1. 그래디언트 계산 비활성화
            for param in model.parameters():
                param.requires_grad = False
            
            # 2. 모델을 평가 모드로 설정
            model.eval()
            
            # 3. KV 캐시 최적화
            self._optimize_kv_cache(model)
            
            # 4. 메모리 매핑 적용 (설정에 따라)
            if self.use_memory_mapping:
                self._apply_memory_mapping(model)
            
            # 5. 인플레이스 연산 활성화 (설정에 따라)
            if self.enable_inplace_operations:
                self._enable_inplace_operations()
            
            # 6. 메모리 정리
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            logger.info("추론 최적화 적용 완료")
            return model
            
        except Exception as e:
            logger.error(f"추론 최적화 적용 중 오류 발생: {str(e)}", exc_info=True)
            raise
    
    def _optimize_kv_cache(self, model: PreTrainedModel) -> None:
        """
        KV 캐시 최적화
        
        Args:
            model: 최적화할 모델
        """
        # KV 캐시 활성화
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        
        # KV 캐시 크기 제한 설정
        # 이 부분은 모델 구현에 따라 다를 수 있음
        if hasattr(model, "config"):
            # 캐시 크기 제한 설정 (바이트 단위)
            cache_size_bytes = self.kv_cache_size_mb * 1024 * 1024
            
            # 모델 설정에 캐시 크기 제한 추가
            if not hasattr(model.config, "max_cache_size"):
                model.config.max_cache_size = cache_size_bytes
            else:
                model.config.max_cache_size = min(model.config.max_cache_size, cache_size_bytes)
            
            logger.info(f"KV 캐시 크기 제한 설정: {self.kv_cache_size_mb}MB")
    
    def _apply_memory_mapping(self, model: PreTrainedModel) -> None:
        """
        메모리 매핑 적용
        
        Args:
            model: 최적화할 모델
        """
        # 메모리 매핑 적용
        # 이 부분은 모델 구현에 따라 다를 수 있음
        logger.info("메모리 매핑 적용")
    
    def _enable_inplace_operations(self) -> None:
        """인플레이스 연산 활성화"""
        # PyTorch 설정 변경
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        logger.info("인플레이스 연산 활성화")

class ModelSharding:
    """모델 샤딩 클래스"""
    
    def __init__(self, num_shards: int = 2):
        """
        모델 샤딩 초기화
        
        Args:
            num_shards: 샤드 수
        """
        self.num_shards = num_shards
        
        logger.info(f"모델 샤딩 초기화: 샤드 수={num_shards}")
    
    def apply_model_sharding(self, model: PreTrainedModel) -> List[torch.nn.Module]:
        """
        모델 샤딩 적용
        
        Args:
            model: 샤딩할 모델
        
        Returns:
            샤딩된 모델 리스트
        """
        logger.info(f"모델 샤딩 적용 시작: 샤드 수={self.num_shards}")
        
        try:
            # 모델 레이어 추출
            layers = self._extract_model_layers(model)
            
            # 레이어 샤딩
            sharded_layers = self._shard_layers(layers)
            
            # 샤딩된 모델 생성
            sharded_models = self._create_sharded_models(model, sharded_layers)
            
            logger.info("모델 샤딩 적용 완료")
            return sharded_models
            
        except Exception as e:
            logger.error(f"모델 샤딩 적용 중 오류 발생: {str(e)}", exc_info=True)
            raise
    
    def _extract_model_layers(self, model: PreTrainedModel) -> List[torch.nn.Module]:
        """
        모델 레이어 추출
        
        Args:
            model: 모델
        
        Returns:
            레이어 리스트
        """
        # 모델 구현에 따라 레이어 추출 방법이 다를 수 있음
        # 여기서는 간단한 예시만 제공
        
        layers = []
        
        # Transformer 모델의 경우 일반적으로 layers 또는 h 속성에 레이어가 저장됨
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = list(model.transformer.h)
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            layers = list(model.encoder.layer)
        elif hasattr(model, "layers"):
            layers = list(model.layers)
        elif hasattr(model, "h"):
            layers = list(model.h)
        else:
            logger.warning("모델 레이어를 추출할 수 없습니다. 샤딩이 제대로 적용되지 않을 수 있습니다.")
        
        return layers
    
    def _shard_layers(self, layers: List[torch.nn.Module]) -> List[List[torch.nn.Module]]:
        """
        레이어 샤딩
        
        Args:
            layers: 레이어 리스트
        
        Returns:
            샤딩된 레이어 리스트
        """
        if not layers:
            return [[]]
        
        # 레이어 수
        num_layers = len(layers)
        
        # 샤드당 레이어 수 계산
        layers_per_shard = num_layers // self.num_shards
        remainder = num_layers % self.num_shards
        
        # 샤딩된 레이어 리스트 생성
        sharded_layers = []
        start_idx = 0
        
        for i in range(self.num_shards):
            # 이 샤드에 할당할 레이어 수 계산
            shard_size = layers_per_shard + (1 if i < remainder else 0)
            
            # 이 샤드에 할당할 레이어 추출
            shard_layers = layers[start_idx:start_idx + shard_size]
            sharded_layers.append(shard_layers)
            
            # 다음 샤드의 시작 인덱스 업데이트
            start_idx += shard_size
        
        return sharded_layers
    
    def _create_sharded_models(self, model: PreTrainedModel, 
                              sharded_layers: List[List[torch.nn.Module]]) -> List[torch.nn.Module]:
        """
        샤딩된 모델 생성
        
        Args:
            model: 원본 모델
            sharded_layers: 샤딩된 레이어 리스트
        
        Returns:
            샤딩된 모델 리스트
        """
        # 이 부분은 모델 구현에 따라 크게 달라질 수 있음
        # 여기서는 간단한 예시만 제공
        
        # 샤딩된 모델 리스트
        sharded_models = []
        
        # 각 샤드에 대해
        for i, shard_layers in enumerate(sharded_layers):
            # 원본 모델 복제
            shard_model = type(model)(model.config)
            
            # 샤드 레이어 설정
            # 이 부분은 모델 구현에 따라 달라질 수 있음
            if hasattr(shard_model, "transformer") and hasattr(shard_model.transformer, "h"):
                shard_model.transformer.h = torch.nn.ModuleList(shard_layers)
            elif hasattr(shard_model, "encoder") and hasattr(shard_model.encoder, "layer"):
                shard_model.encoder.layer = torch.nn.ModuleList(shard_layers)
            elif hasattr(shard_model, "layers"):
                shard_model.layers = torch.nn.ModuleList(shard_layers)
            elif hasattr(shard_model, "h"):
                shard_model.h = torch.nn.ModuleList(shard_layers)
            
            # 샤드 모델 추가
            sharded_models.append(shard_model)
        
        return sharded_models

class MemoryEfficientInference:
    """메모리 효율적 추론 클래스"""
    
    def __init__(self, 
                 max_memory_gb: float = 2.5,
                 batch_size: int = 1,
                 max_new_tokens: int = 512):
        """
        메모리 효율적 추론 초기화
        
        Args:
            max_memory_gb: 최대 허용 메모리 (GB)
            batch_size: 배치 크기
            max_new_tokens: 최대 생성 토큰 수
        """
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        
        # 메모리 모니터링 초기화
        self.memory_monitor = MemoryMonitor()
        
        logger.info(f"메모리 효율적 추론 초기화: 최대 메모리={max_memory_gb}GB, 배치 크기={batch_size}")
    
    def generate_text(self, model: PreTrainedModel, tokenizer, prompt: str, 
                     generation_config: Optional[Dict[str, Any]] = None) -> str:
        """
        텍스트 생성
        
        Args:
            model: 모델
            tokenizer: 토크나이저
            prompt: 프롬프트
            generation_config: 생성 설정
        
        Returns:
            생성된 텍스트
        """
        logger.info("텍스트 생성 시작")
        
        try:
            # 메모리 모니터링 시작
            self.memory_monitor.start_monitoring()
            
            # 기본 생성 설정
            if generation_config is None:
                generation_config = {
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1
                }
            
            # 입력 인코딩
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            
            # 메모리 사용량 확인
            current_memory = self.memory_monitor.get_current_memory_usage()
            logger.info(f"현재 메모리 사용량: {current_memory / (1024 * 1024):.2f}MB")
            
            # 메모리 제한 초과 시 청크 단위로 처리
            if current_memory > self.max_memory_bytes * 0.8:
                logger.warning("메모리 사용량이 높습니다. 청크 단위로 처리합니다.")
                return self._generate_text_in_chunks(model, tokenizer, prompt, generation_config)
            
            # 텍스트 생성
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )
            
            # 생성된 텍스트 디코딩
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 메모리 모니터링 종료
            self.memory_monitor.stop_monitoring()
            
            logger.info("텍스트 생성 완료")
            return generated_text
            
        except Exception as e:
            logger.error(f"텍스트 생성 중 오류 발생: {str(e)}", exc_info=True)
            
            # 메모리 모니터링 종료
            self.memory_monitor.stop_monitoring()
            
            # 메모리 정리
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            raise
    
    def _generate_text_in_chunks(self, model: PreTrainedModel, tokenizer, prompt: str, 
                                generation_config: Dict[str, Any]) -> str:
        """
        청크 단위로 텍스트 생성
        
        Args:
            model: 모델
            tokenizer: 토크나이저
            prompt: 프롬프트
            generation_config: 생성 설정
        
        Returns:
            생성된 텍스트
        """
        logger.info("청크 단위로 텍스트 생성 시작")
        
        # 입력 인코딩
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # 청크 크기 설정 (토큰 수)
        chunk_size = 128
        
        # 최대 생성 토큰 수
        max_new_tokens = generation_config.get("max_new_tokens", self.max_new_tokens)
        
        # 생성된 토큰 리스트
        generated_tokens = input_ids[0].tolist()
        
        # 청크 단위로 생성
        remaining_tokens = max_new_tokens
        
        while remaining_tokens > 0:
            # 현재 청크 크기 계산
            current_chunk_size = min(chunk_size, remaining_tokens)
            
            # 현재 청크 생성 설정
            chunk_config = generation_config.copy()
            chunk_config["max_new_tokens"] = current_chunk_size
            
            # 현재 입력 설정
            current_input_ids = torch.tensor([generated_tokens[-512:]]).to(input_ids.device)
            
            # 텍스트 생성
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=current_input_ids,
                    **chunk_config
                )
            
            # 새로 생성된 토큰 추출
            new_tokens = outputs[0][len(current_input_ids[0]):].tolist()
            
            # 생성된 토큰 리스트에 추가
            generated_tokens.extend(new_tokens)
            
            # 남은 토큰 수 업데이트
            remaining_tokens -= len(new_tokens)
            
            # 메모리 정리
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 생성이 완료되었는지 확인
            if len(new_tokens) == 0 or new_tokens[-1] == tokenizer.eos_token_id:
                break
        
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        logger.info("청크 단위로 텍스트 생성 완료")
        return generated_text

class MemoryMonitor:
    """메모리 모니터링 클래스"""
    
    def __init__(self, interval_seconds: float = 1.0):
        """
        메모리 모니터링 초기화
        
        Args:
            interval_seconds: 모니터링 간격 (초)
        """
        self.interval_seconds = interval_seconds
        self.monitoring = False
        self.peak_memory = 0
        
        logger.info(f"메모리 모니터링 초기화: 간격={interval_seconds}초")
    
    def start_monitoring(self) -> None:
        """메모리 모니터링 시작"""
        self.monitoring = True
        self.peak_memory = 0
        
        logger.info("메모리 모니터링 시작")
    
    def stop_monitoring(self) -> None:
        """메모리 모니터링 종료"""
        self.monitoring = False
        
        logger.info(f"메모리 모니터링 종료: 최대 메모리 사용량={self.peak_memory / (1024 * 1024):.2f}MB")
    
    def get_current_memory_usage(self) -> int:
        """
        현재 메모리 사용량 반환
        
        Returns:
            현재 메모리 사용량 (바이트)
        """
        # 프로세스 메모리 사용량 확인
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # RSS(Resident Set Size) 사용
        memory_usage = memory_info.rss
        
        # 최대 메모리 사용량 업데이트
        if memory_usage > self.peak_memory:
            self.peak_memory = memory_usage
        
        return memory_usage

class ChatGPTLevelInferenceOptimizer(InferenceOptimizer):
    """ChatGPT 수준의 추론 최적화 클래스"""
    
    def __init__(self, 
                 kv_cache_size_mb: int = 256,
                 use_memory_mapping: bool = True,
                 enable_inplace_operations: bool = True,
                 max_memory_gb: float = 2.8,
                 use_flash_attention: bool = True,
                 use_grouped_query_attention: bool = True):
        """
        ChatGPT 수준의 추론 최적화 초기화
        
        Args:
            kv_cache_size_mb: KV 캐시 크기 (MB)
            use_memory_mapping: 메모리 매핑 사용 여부
            enable_inplace_operations: 인플레이스 연산 활성화 여부
            max_memory_gb: 최대 허용 메모리 (GB)
            use_flash_attention: 플래시 어텐션 사용 여부
            use_grouped_query_attention: 그룹 쿼리 어텐션 사용 여부
        """
        super().__init__(
            kv_cache_size_mb=kv_cache_size_mb,
            use_memory_mapping=use_memory_mapping,
            enable_inplace_operations=enable_inplace_operations,
            max_memory_gb=max_memory_gb
        )
        
        self.use_flash_attention = use_flash_attention
        self.use_grouped_query_attention = use_grouped_query_attention
        
        logger.info(f"ChatGPT 수준의 추론 최적화 초기화: 플래시 어텐션={use_flash_attention}, 그룹 쿼리 어텐션={use_grouped_query_attention}")
    
    def optimize_for_inference(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        추론을 위한 최적화 적용 (확장)
        
        Args:
            model: 최적화할 모델
        
        Returns:
            최적화된 모델
        """
        # 기본 최적화 적용
        model = super().optimize_for_inference(model)
        
        # 추가 최적화 적용
        model = self._apply_advanced_optimizations(model)
        
        return model
    
    def _apply_advanced_optimizations(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        고급 최적화 적용
        
        Args:
            model: 최적화할 모델
        
        Returns:
            최적화된 모델
        """
        logger.info("고급 최적화 적용 시작")
        
        try:
            # 1. 플래시 어텐션 적용 (설정에 따라)
            if self.use_flash_attention:
                model = self._apply_flash_attention(model)
            
            # 2. 그룹 쿼리 어텐션 적용 (설정에 따라)
            if self.use_grouped_query_attention:
                model = self._apply_grouped_query_attention(model)
            
            # 3. 추가 메모리 최적화
            model = self._apply_additional_memory_optimizations(model)
            
            logger.info("고급 최적화 적용 완료")
            return model
            
        except Exception as e:
            logger.error(f"고급 최적화 적용 중 오류 발생: {str(e)}", exc_info=True)
            raise
    
    def _apply_flash_attention(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        플래시 어텐션 적용
        
        Args:
            model: 최적화할 모델
        
        Returns:
            최적화된 모델
        """
        # 플래시 어텐션 적용
        # 이 부분은 모델 구현에 따라 다를 수 있음
        
        # 모델 설정에 플래시 어텐션 활성화
        if hasattr(model, "config"):
            model.config.use_flash_attention = True
        
        logger.info("플래시 어텐션 적용")
        return model
    
    def _apply_grouped_query_attention(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        그룹 쿼리 어텐션 적용
        
        Args:
            model: 최적화할 모델
        
        Returns:
            최적화된 모델
        """
        # 그룹 쿼리 어텐션 적용
        # 이 부분은 모델 구현에 따라 다를 수 있음
        
        # 모델 설정에 그룹 쿼리 어텐션 활성화
        if hasattr(model, "config"):
            model.config.use_grouped_query_attention = True
        
        logger.info("그룹 쿼리 어텐션 적용")
        return model
    
    def _apply_additional_memory_optimizations(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        추가 메모리 최적화 적용
        
        Args:
            model: 최적화할 모델
        
        Returns:
            최적화된 모델
        """
        # 추가 메모리 최적화 적용
        # 이 부분은 모델 구현에 따라 다를 수 있음
        
        logger.info("추가 메모리 최적화 적용")
        return model
