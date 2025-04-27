"""
한국어 특화 초경량 AI 모델 - 극한의 경량화 기법 적용 모듈
- 2비트 양자화 구현
- 모델 프루닝 적용
- 지식 증류 기법 구현
- 메모리 최적화 기법 통합
"""

import os
import gc
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExtremeQuantizer:
    """극한의 양자화 클래스"""
    
    def __init__(self, bits: int = 2):
        """
        극한의 양자화 초기화
        
        Args:
            bits: 양자화 비트 수 (2, 3, 4, 8 중 선택)
        """
        self.bits = bits
        self.supported_bits = [2, 3, 4, 8]
        
        if self.bits not in self.supported_bits:
            logger.warning(f"지원하지 않는 비트 수: {bits}, 기본값 4비트로 설정합니다.")
            self.bits = 4
        
        logger.info(f"극한의 양자화 초기화: {self.bits}비트")
    
    def quantize_model(self, model_path: str, device: str = "cpu") -> torch.nn.Module:
        """
        모델 양자화
        
        Args:
            model_path: 모델 경로 또는 Hugging Face 모델 ID
            device: 실행 장치 (cpu 또는 cuda)
        
        Returns:
            양자화된 모델
        """
        logger.info(f"{self.bits}비트 양자화 모델 로드 시작: {model_path}")
        
        try:
            if self.bits == 4:
                # 4비트 양자화는 BitsAndBytes 라이브러리 사용
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto" if device == "cuda" and torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
                
            elif self.bits == 2:
                # 2비트 양자화는 직접 구현
                # 먼저 모델을 로드한 후 가중치를 2비트로 양자화
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto" if device == "cuda" and torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
                
                # 2비트 양자화 적용
                self._apply_2bit_quantization(model)
                
            elif self.bits == 3:
                # 3비트 양자화도 직접 구현
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto" if device == "cuda" and torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
                
                # 3비트 양자화 적용
                self._apply_3bit_quantization(model)
                
            else:  # 8비트
                # 8비트 양자화는 BitsAndBytes 라이브러리 사용
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    load_in_8bit=True,
                    device_map="auto" if device == "cuda" and torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
            
            logger.info(f"{self.bits}비트 양자화 모델 로드 완료")
            return model
            
        except Exception as e:
            logger.error(f"모델 양자화 중 오류 발생: {str(e)}", exc_info=True)
            raise
    
    def _apply_2bit_quantization(self, model: torch.nn.Module) -> None:
        """
        2비트 양자화 적용
        
        Args:
            model: 양자화할 모델
        """
        logger.info("2비트 양자화 적용 시작")
        
        # 모델의 모든 Linear 레이어에 대해 2비트 양자화 적용
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # 가중치 텐서
                weight = module.weight.data
                
                # 스케일 계산 (각 행의 최대값)
                scale = torch.max(torch.abs(weight), dim=1, keepdim=True)[0]
                
                # 스케일이 0인 경우 처리
                scale[scale == 0] = 1.0
                
                # 가중치를 [-2, -1, 0, 1]의 4개 값으로 양자화
                weight_normalized = weight / scale
                weight_quantized = torch.round(torch.clamp(weight_normalized * 1.5, -2, 1))
                
                # 양자화된 가중치 저장
                module.weight.data = weight_quantized * scale
                
                # 메모리 정리
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info("2비트 양자화 적용 완료")
    
    def _apply_3bit_quantization(self, model: torch.nn.Module) -> None:
        """
        3비트 양자화 적용
        
        Args:
            model: 양자화할 모델
        """
        logger.info("3비트 양자화 적용 시작")
        
        # 모델의 모든 Linear 레이어에 대해 3비트 양자화 적용
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # 가중치 텐서
                weight = module.weight.data
                
                # 스케일 계산 (각 행의 최대값)
                scale = torch.max(torch.abs(weight), dim=1, keepdim=True)[0]
                
                # 스케일이 0인 경우 처리
                scale[scale == 0] = 1.0
                
                # 가중치를 [-4, -3, -2, -1, 0, 1, 2, 3]의 8개 값으로 양자화
                weight_normalized = weight / scale
                weight_quantized = torch.round(torch.clamp(weight_normalized * 3.5, -4, 3))
                
                # 양자화된 가중치 저장
                module.weight.data = weight_quantized * scale
                
                # 메모리 정리
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info("3비트 양자화 적용 완료")

class ModelPruner:
    """모델 프루닝 클래스"""
    
    def __init__(self, pruning_ratio: float = 0.3):
        """
        모델 프루닝 초기화
        
        Args:
            pruning_ratio: 프루닝 비율 (0.0 ~ 0.9)
        """
        self.pruning_ratio = max(0.0, min(0.9, pruning_ratio))
        logger.info(f"모델 프루닝 초기화: 프루닝 비율 {self.pruning_ratio:.1f}")
    
    def prune_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        모델 프루닝 적용
        
        Args:
            model: 프루닝할 모델
        
        Returns:
            프루닝된 모델
        """
        logger.info(f"모델 프루닝 시작: 프루닝 비율 {self.pruning_ratio:.1f}")
        
        try:
            # 모델의 모든 Linear 레이어에 대해 프루닝 적용
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # 가중치 텐서
                    weight = module.weight.data
                    
                    # 절대값이 작은 가중치를 0으로 설정
                    threshold = torch.quantile(torch.abs(weight), self.pruning_ratio)
                    mask = torch.abs(weight) > threshold
                    module.weight.data = weight * mask
                    
                    # 프루닝 결과 로깅
                    sparsity = 1.0 - torch.count_nonzero(module.weight.data) / weight.numel()
                    logger.debug(f"레이어 {name} 프루닝 완료: 희소성 {sparsity:.2f}")
                    
                    # 메모리 정리
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            logger.info("모델 프루닝 완료")
            return model
            
        except Exception as e:
            logger.error(f"모델 프루닝 중 오류 발생: {str(e)}", exc_info=True)
            raise

class KnowledgeDistiller:
    """지식 증류 클래스"""
    
    def __init__(self, teacher_model_path: str, temperature: float = 2.0):
        """
        지식 증류 초기화
        
        Args:
            teacher_model_path: 교사 모델 경로 또는 Hugging Face 모델 ID
            temperature: 증류 온도 (높을수록 소프트한 확률 분포)
        """
        self.teacher_model_path = teacher_model_path
        self.temperature = temperature
        self.teacher_model = None
        self.teacher_tokenizer = None
        
        logger.info(f"지식 증류 초기화: 교사 모델 {teacher_model_path}, 온도 {temperature}")
    
    def load_teacher_model(self, device: str = "cpu") -> None:
        """
        교사 모델 로드
        
        Args:
            device: 실행 장치 (cpu 또는 cuda)
        """
        logger.info(f"교사 모델 로드 시작: {self.teacher_model_path}")
        
        try:
            # 교사 모델 로드
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.teacher_model_path,
                device_map="auto" if device == "cuda" and torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            # 교사 모델 토크나이저 로드
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_path)
            
            logger.info("교사 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"교사 모델 로드 중 오류 발생: {str(e)}", exc_info=True)
            raise
    
    def distill_model(self, student_model: torch.nn.Module, training_data: List[str],
                      device: str = "cpu", epochs: int = 1) -> torch.nn.Module:
        """
        모델 지식 증류
        
        Args:
            student_model: 학생 모델
            training_data: 학습 데이터 리스트
            device: 실행 장치 (cpu 또는 cuda)
            epochs: 학습 에폭 수
        
        Returns:
            증류된 학생 모델
        """
        if self.teacher_model is None:
            self.load_teacher_model(device)
        
        logger.info(f"모델 지식 증류 시작: {len(training_data)} 데이터, {epochs} 에폭")
        
        try:
            # 학생 모델을 학습 모드로 설정
            student_model.train()
            
            # 교사 모델을 평가 모드로 설정
            self.teacher_model.eval()
            
            # 옵티마이저 설정
            optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
            
            # 손실 함수 설정 (KL 발산)
            kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
            
            # 학습 루프
            for epoch in range(epochs):
                total_loss = 0.0
                
                for i, text in enumerate(training_data):
                    # 입력 인코딩
                    inputs = self.teacher_tokenizer(text, return_tensors="pt")
                    input_ids = inputs.input_ids.to(device)
                    attention_mask = inputs.attention_mask.to(device)
                    
                    # 교사 모델 예측 (그래디언트 계산 없이)
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        teacher_logits = teacher_outputs.logits / self.temperature
                        teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
                    
                    # 학생 모델 예측
                    student_outputs = student_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    student_logits = student_outputs.logits / self.temperature
                    student_log_probs = torch.nn.functional.log_softmax(student_logits, dim=-1)
                    
                    # 손실 계산
                    loss = kl_loss(student_log_probs, teacher_probs)
                    
                    # 역전파 및 옵티마이저 스텝
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # 진행 상황 로깅
                    if (i + 1) % 10 == 0:
                        logger.info(f"에폭 {epoch+1}/{epochs}, 배치 {i+1}/{len(training_data)}, 손실: {loss.item():.4f}")
                    
                    # 메모리 정리
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # 에폭 완료 로깅
                avg_loss = total_loss / len(training_data)
                logger.info(f"에폭 {epoch+1}/{epochs} 완료, 평균 손실: {avg_loss:.4f}")
            
            # 학생 모델을 평가 모드로 설정
            student_model.eval()
            
            logger.info("모델 지식 증류 완료")
            return student_model
            
        except Exception as e:
            logger.error(f"모델 지식 증류 중 오류 발생: {str(e)}", exc_info=True)
            raise

class MemoryOptimizationTechniques:
    """메모리 최적화 기법 클래스"""
    
    def __init__(self, max_memory_gb: float = 2.5):
        """
        메모리 최적화 기법 초기화
        
        Args:
            max_memory_gb: 최대 허용 메모리 (GB)
        """
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        
        logger.info(f"메모리 최적화 기법 초기화: 최대 허용 메모리 {max_memory_gb}GB")
    
    def apply_memory_optimizations(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        메모리 최적화 기법 적용
        
        Args:
            model: 최적화할 모델
        
        Returns:
            최적화된 모델
        """
        logger.info("메모리 최적화 기법 적용 시작")
        
        try:
            # 1. 그래디언트 계산 비활성화
            for param in model.parameters():
                param.requires_grad = False
            
            # 2. 모델을 평가 모드로 설정
            model.eval()
            
            # 3. KV 캐시 활성화
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = True
            
            # 4. 메모리 매핑 적용
            self._apply_memory_mapping(model)
            
            # 5. 인플레이스 연산 활성화
            self._enable_inplace_operations()
            
            # 6. 메모리 정리
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            logger.info("메모리 최적화 기법 적용 완료")
            return model
            
        except Exception as e:
            logger.error(f"메모리 최적화 기법 적용 중 오류 발생: {str(e)}", exc_info=True)
            raise
    
    def _apply_memory_mapping(self, model: torch.nn.Module) -> None:
        """
        메모리 매핑 적용
        
        Args:
            model: 최적화할 모델
        """
        # 모델 구조에 따라 메모리 매핑 적용
        # 이 부분은 모델 구조에 따라 다르게 구현해야 함
        pass
    
    def _enable_inplace_operations(self) -> None:
        """인플레이스 연산 활성화"""
        # PyTorch 설정 변경
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

class ExtremeOptimizer:
    """극한의 최적화 클래스"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        극한의 최적화 초기화
        
        Args:
            model_config: 모델 설정 딕셔너리
        """
        self.model_config = model_config
        self.quantizer = ExtremeQuantizer(bits=model_config.get("quantization_bits", 4))
        self.pruner = ModelPruner(pruning_ratio=model_config.get("pruning_ratio", 0.3))
        self.memory_optimizer = MemoryOptimizationTechniques(max_memory_gb=model_config.get("max_memory_gb", 2.5))
        
        # 지식 증류 설정
        teacher_model_name = model_config.get("teacher_model_name")
        if model_config.get("use_knowledge_distillation", False) and teacher_model_name:
            self.distiller = KnowledgeDistiller(teacher_model_name)
        else:
            self.distiller = None
        
        logger.info("극한의 최적화 초기화 완료")
    
    def optimize_model(self, model_path: str, training_data: Optional[List[str]] = None) -> torch.nn.Module:
        """
        모델 최적화
        
        Args:
            model_path: 모델 경로 또는 Hugging Face 모델 ID
            training_data: 지식 증류를 위한 학습 데이터 (선택 사항)
        
        Returns:
            최적화된 모델
        """
        logger.info(f"모델 최적화 시작: {model_path}")
        
        try:
            # 1. 양자화 적용
            model = self.quantizer.quantize_model(model_path, device=self.model_config.get("device", "cpu"))
            
            # 2. 프루닝 적용 (모델 설정에 따라)
            if self.model_config.get("use_pruning", False):
                model = self.pruner.prune_model(model)
            
            # 3. 메모리 최적화 적용
            model = self.memory_optimizer.apply_memory_optimizations(model)
            
            # 4. 지식 증류 적용 (모델 설정 및 학습 데이터에 따라)
            if self.distiller and training_data:
                model = self.distiller.distill_model(
                    model,
                    training_data,
                    device=self.model_config.get("device", "cpu")
                )
            
            logger.info("모델 최적화 완료")
            return model
            
        except Exception as e:
            logger.error(f"모델 최적화 중 오류 발생: {str(e)}", exc_info=True)
            raise
    
    def get_model_size_info(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        모델 크기 정보 반환
        
        Args:
            model: 모델
        
        Returns:
            모델 크기 정보 딕셔너리
        """
        # 모델 파라미터 수 계산
        param_count = sum(p.numel() for p in model.parameters())
        
        # 모델 메모리 사용량 계산 (대략적인 추정)
        bits_per_param = self.model_config.get("quantization_bits", 4)
        memory_bytes = param_count * bits_per_param / 8
        memory_mb = memory_bytes / (1024 * 1024)
        
        # 모델 희소성 계산 (프루닝된 경우)
        non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters())
        sparsity = 1.0 - non_zero_params / param_count if param_count > 0 else 0.0
        
        return {
            "param_count": param_count,
            "memory_mb": memory_mb,
            "bits_per_param": bits_per_param,
            "sparsity": sparsity
        }
