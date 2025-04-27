"""
한국어 특화 경량화 AI 모델 로더 모듈
- Polyglot-ko 410M 모델을 기반으로 경량화 및 최적화 적용
- INT4 양자화, 모델 구조 최적화 등 적용
"""

import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KoreanLightweightModel:
    """한국어 특화 경량화 AI 모델 클래스"""
    
    def __init__(self, model_name="EleutherAI/polyglot-ko-1.3b", use_4bit=True, device="cpu"):
        """
        모델 초기화
        
        Args:
            model_name: 기본 모델 이름 (Hugging Face 모델 ID)
            use_4bit: 4비트 양자화 사용 여부
            device: 실행 장치 (cpu 또는 cuda)
        """
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # 메모리 사용량 모니터링을 위한 초기 메모리 사용량 기록
        self.initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        logger.info(f"모델 초기화: {model_name}, 4비트 양자화: {use_4bit}, 장치: {device}")
    
    def load_model(self, use_sharding=True):
        """
        모델 로드 및 최적화
        
        Args:
            use_sharding: 모델 샤딩 사용 여부
        
        Returns:
            로드된 모델과 토크나이저
        """
        logger.info(f"모델 로드 시작: {self.model_name}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # 4비트 양자화 설정
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("4비트 양자화 설정 적용")
            
            # 4비트 양자화된 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
        # 모델 샤딩 사용 (메모리 제약이 심한 환경용)
        elif use_sharding:
            logger.info("모델 샤딩 적용")
            # 빈 가중치로 모델 초기화
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # 체크포인트 로드 및 디스패치 (메모리 효율적 로딩)
            self.model = load_checkpoint_and_dispatch(
                self.model, 
                self.model_name, 
                device_map="auto" if torch.cuda.is_available() else None,
                no_split_module_classes=["TransformerBlock"]
            )
        
        # 일반 모델 로드
        else:
            logger.info("일반 모델 로드")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
        
        # 메모리 사용량 로깅
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            memory_used = current_memory - self.initial_memory
            logger.info(f"모델 로드 후 메모리 사용량: {memory_used / 1024**2:.2f} MB")
        
        logger.info("모델 로드 완료")
        return self.model, self.tokenizer
    
    def optimize_for_inference(self):
        """
        추론을 위한 모델 최적화
        - KV 캐시 최적화
        - 메모리 효율적 설정
        """
        if self.model is None:
            logger.error("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
            return
        
        logger.info("추론을 위한 모델 최적화 적용")
        
        # 추론 모드 설정
        self.model.eval()
        
        # 그래디언트 계산 비활성화
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 메모리 최적화를 위한 설정
        if hasattr(self.model.config, "use_cache"):
            # KV 캐시 활성화
            self.model.config.use_cache = True
        
        # 토크나이저 패딩 설정
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("추론 최적화 완료")
        return self.model
    
    def generate_text(self, prompt, max_length=128, temperature=0.7, top_p=0.9):
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            max_length: 최대 생성 길이
            temperature: 생성 온도 (높을수록 다양한 결과)
            top_p: 상위 확률 샘플링 파라미터
        
        Returns:
            생성된 텍스트
        """
        if self.model is None or self.tokenizer is None:
            logger.error("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
            return ""
        
        logger.info(f"텍스트 생성 시작: 프롬프트 길이 {len(prompt)}")
        
        # 입력 인코딩
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        
        # 메모리 효율적인 생성 설정
        generation_config = {
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            # 메모리 효율을 위한 설정
            "use_cache": True,
        }
        
        # 텍스트 생성
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **generation_config)
        
        # 결과 디코딩
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        logger.info(f"텍스트 생성 완료: 결과 길이 {len(generated_text)}")
        return generated_text
    
    def get_memory_usage(self):
        """현재 메모리 사용량 반환"""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            memory_used = current_memory - self.initial_memory
            return memory_used / 1024**2  # MB 단위
        return 0
