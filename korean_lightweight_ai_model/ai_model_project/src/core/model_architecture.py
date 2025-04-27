"""
한국어 특화 초경량 AI 모델 - 모델 구조 정의 모듈

이 모듈은 한국어 특화 초경량 AI 모델의 아키텍처를 정의합니다.
- 모델 설정 클래스
- 토크나이저 설정 클래스
- 추론 설정 클래스
- 대화 설정 클래스
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelConfig:
    """모델 설정 클래스"""
    
    def __init__(self, 
                 model_name: str = "polyglot-ko-410m",
                 quantization_bits: int = 2,
                 pruning_ratio: float = 0.5,
                 max_memory_gb: float = 2.5,
                 use_knowledge_distillation: bool = True,
                 teacher_model_name: Optional[str] = "polyglot-ko-1.3b",
                 device: str = "cpu"):
        """
        모델 설정 초기화
        
        Args:
            model_name: 기본 모델 이름 또는 경로
            quantization_bits: 양자화 비트 수 (2, 3, 4, 8)
            pruning_ratio: 프루닝 비율 (0.0 ~ 0.9)
            max_memory_gb: 최대 허용 메모리 (GB)
            use_knowledge_distillation: 지식 증류 사용 여부
            teacher_model_name: 교사 모델 이름 (지식 증류 사용 시)
            device: 실행 장치 (cpu 또는 cuda)
        """
        self.model_name = model_name
        self.quantization_bits = quantization_bits
        self.pruning_ratio = pruning_ratio
        self.max_memory_gb = max_memory_gb
        self.use_knowledge_distillation = use_knowledge_distillation
        self.teacher_model_name = teacher_model_name
        self.device = device
        
        logger.info(f"모델 설정 초기화: {model_name}, {quantization_bits}비트 양자화, {pruning_ratio:.1f} 프루닝 비율")
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "model_name": self.model_name,
            "quantization_bits": self.quantization_bits,
            "pruning_ratio": self.pruning_ratio,
            "max_memory_gb": self.max_memory_gb,
            "use_knowledge_distillation": self.use_knowledge_distillation,
            "teacher_model_name": self.teacher_model_name,
            "device": self.device
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """딕셔너리에서 설정 생성"""
        return cls(
            model_name=config_dict.get("model_name", "polyglot-ko-410m"),
            quantization_bits=config_dict.get("quantization_bits", 2),
            pruning_ratio=config_dict.get("pruning_ratio", 0.5),
            max_memory_gb=config_dict.get("max_memory_gb", 2.5),
            use_knowledge_distillation=config_dict.get("use_knowledge_distillation", True),
            teacher_model_name=config_dict.get("teacher_model_name", "polyglot-ko-1.3b"),
            device=config_dict.get("device", "cpu")
        )

class TokenizerConfig:
    """토크나이저 설정 클래스"""
    
    def __init__(self,
                 use_korean_optimizer: bool = True,
                 jamo_separation_minimization: bool = True,
                 use_morpheme_analyzer: bool = True,
                 special_token_handling: bool = True):
        """
        토크나이저 설정 초기화
        
        Args:
            use_korean_optimizer: 한국어 최적화 사용 여부
            jamo_separation_minimization: 자모 분리 최소화 여부
            use_morpheme_analyzer: 형태소 분석기 사용 여부
            special_token_handling: 특수 토큰 처리 여부
        """
        self.use_korean_optimizer = use_korean_optimizer
        self.jamo_separation_minimization = jamo_separation_minimization
        self.use_morpheme_analyzer = use_morpheme_analyzer
        self.special_token_handling = special_token_handling
        
        logger.info(f"토크나이저 설정 초기화: 한국어 최적화={use_korean_optimizer}, 자모 분리 최소화={jamo_separation_minimization}")
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "use_korean_optimizer": self.use_korean_optimizer,
            "jamo_separation_minimization": self.jamo_separation_minimization,
            "use_morpheme_analyzer": self.use_morpheme_analyzer,
            "special_token_handling": self.special_token_handling
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TokenizerConfig':
        """딕셔너리에서 설정 생성"""
        return cls(
            use_korean_optimizer=config_dict.get("use_korean_optimizer", True),
            jamo_separation_minimization=config_dict.get("jamo_separation_minimization", True),
            use_morpheme_analyzer=config_dict.get("use_morpheme_analyzer", True),
            special_token_handling=config_dict.get("special_token_handling", True)
        )

class InferenceConfig:
    """추론 설정 클래스"""
    
    def __init__(self,
                 kv_cache_size_mb: int = 128,
                 use_memory_mapping: bool = True,
                 enable_inplace_operations: bool = True,
                 batch_size: int = 1,
                 max_new_tokens: int = 512):
        """
        추론 설정 초기화
        
        Args:
            kv_cache_size_mb: KV 캐시 크기 (MB)
            use_memory_mapping: 메모리 매핑 사용 여부
            enable_inplace_operations: 인플레이스 연산 활성화 여부
            batch_size: 배치 크기
            max_new_tokens: 최대 생성 토큰 수
        """
        self.kv_cache_size_mb = kv_cache_size_mb
        self.use_memory_mapping = use_memory_mapping
        self.enable_inplace_operations = enable_inplace_operations
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        
        logger.info(f"추론 설정 초기화: KV 캐시={kv_cache_size_mb}MB, 메모리 매핑={use_memory_mapping}")
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "kv_cache_size_mb": self.kv_cache_size_mb,
            "use_memory_mapping": self.use_memory_mapping,
            "enable_inplace_operations": self.enable_inplace_operations,
            "batch_size": self.batch_size,
            "max_new_tokens": self.max_new_tokens
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'InferenceConfig':
        """딕셔너리에서 설정 생성"""
        return cls(
            kv_cache_size_mb=config_dict.get("kv_cache_size_mb", 128),
            use_memory_mapping=config_dict.get("use_memory_mapping", True),
            enable_inplace_operations=config_dict.get("enable_inplace_operations", True),
            batch_size=config_dict.get("batch_size", 1),
            max_new_tokens=config_dict.get("max_new_tokens", 512)
        )

class ConversationConfig:
    """대화 설정 클래스"""
    
    def __init__(self,
                 context_length: int = 5,
                 response_quality_enhancement: bool = True,
                 default_style: str = "casual",
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.1):
        """
        대화 설정 초기화
        
        Args:
            context_length: 대화 컨텍스트 길이
            response_quality_enhancement: 응답 품질 향상 여부
            default_style: 기본 대화 스타일 ("formal", "casual", "professional")
            temperature: 생성 온도
            top_p: 누적 확률 임계값
            repetition_penalty: 반복 패널티
        """
        self.context_length = context_length
        self.response_quality_enhancement = response_quality_enhancement
        self.default_style = default_style
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        
        logger.info(f"대화 설정 초기화: 컨텍스트 길이={context_length}, 스타일={default_style}")
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "context_length": self.context_length,
            "response_quality_enhancement": self.response_quality_enhancement,
            "default_style": self.default_style,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConversationConfig':
        """딕셔너리에서 설정 생성"""
        return cls(
            context_length=config_dict.get("context_length", 5),
            response_quality_enhancement=config_dict.get("response_quality_enhancement", True),
            default_style=config_dict.get("default_style", "casual"),
            temperature=config_dict.get("temperature", 0.7),
            top_p=config_dict.get("top_p", 0.9),
            repetition_penalty=config_dict.get("repetition_penalty", 1.1)
        )

class ModelArchitecture:
    """모델 아키텍처 클래스"""
    
    def __init__(self,
                 model_config: ModelConfig,
                 tokenizer_config: TokenizerConfig,
                 inference_config: InferenceConfig,
                 conversation_config: ConversationConfig):
        """
        모델 아키텍처 초기화
        
        Args:
            model_config: 모델 설정
            tokenizer_config: 토크나이저 설정
            inference_config: 추론 설정
            conversation_config: 대화 설정
        """
        self.model_config = model_config
        self.tokenizer_config = tokenizer_config
        self.inference_config = inference_config
        self.conversation_config = conversation_config
        
        logger.info("모델 아키텍처 초기화 완료")
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """아키텍처를 딕셔너리로 변환"""
        return {
            "model_config": self.model_config.to_dict(),
            "tokenizer_config": self.tokenizer_config.to_dict(),
            "inference_config": self.inference_config.to_dict(),
            "conversation_config": self.conversation_config.to_dict()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Dict[str, Any]]) -> 'ModelArchitecture':
        """딕셔너리에서 아키텍처 생성"""
        return cls(
            model_config=ModelConfig.from_dict(config_dict.get("model_config", {})),
            tokenizer_config=TokenizerConfig.from_dict(config_dict.get("tokenizer_config", {})),
            inference_config=InferenceConfig.from_dict(config_dict.get("inference_config", {})),
            conversation_config=ConversationConfig.from_dict(config_dict.get("conversation_config", {}))
        )
    
    def save_config(self, output_dir: str) -> str:
        """
        설정 저장
        
        Args:
            output_dir: 출력 디렉토리
        
        Returns:
            저장된 파일 경로
        """
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, "model_config.json")
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"모델 설정 저장 완료: {config_path}")
        return config_path
    
    @classmethod
    def load_config(cls, config_path: str) -> 'ModelArchitecture':
        """
        설정 로드
        
        Args:
            config_path: 설정 파일 경로
        
        Returns:
            모델 아키텍처
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        logger.info(f"모델 설정 로드 완료: {config_path}")
        return cls.from_dict(config_dict)

def create_extreme_lightweight_architecture() -> ModelArchitecture:
    """
    극한의 경량화 아키텍처 생성
    
    Returns:
        모델 아키텍처
    """
    # 모델 설정
    model_config = ModelConfig(
        model_name="polyglot-ko-410m",
        quantization_bits=2,
        pruning_ratio=0.5,
        max_memory_gb=2.5,
        use_knowledge_distillation=True,
        teacher_model_name="polyglot-ko-1.3b"
    )
    
    # 토크나이저 설정
    tokenizer_config = TokenizerConfig(
        use_korean_optimizer=True,
        jamo_separation_minimization=True,
        use_morpheme_analyzer=True,
        special_token_handling=True
    )
    
    # 추론 설정
    inference_config = InferenceConfig(
        kv_cache_size_mb=128,
        use_memory_mapping=True,
        enable_inplace_operations=True,
        batch_size=1,
        max_new_tokens=512
    )
    
    # 대화 설정
    conversation_config = ConversationConfig(
        context_length=5,
        response_quality_enhancement=True,
        default_style="casual",
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    # 아키텍처 생성
    architecture = ModelArchitecture(
        model_config=model_config,
        tokenizer_config=tokenizer_config,
        inference_config=inference_config,
        conversation_config=conversation_config
    )
    
    logger.info("극한의 경량화 아키텍처 생성 완료")
    return architecture

def create_chatgpt_level_architecture() -> ModelArchitecture:
    """
    ChatGPT 수준의 아키텍처 생성
    
    Returns:
        모델 아키텍처
    """
    # 모델 설정
    model_config = ModelConfig(
        model_name="polyglot-ko-1.3b",  # 더 큰 모델 사용
        quantization_bits=3,  # 3비트로 약간 높임 (품질 향상)
        pruning_ratio=0.3,    # 프루닝 비율 낮춤 (품질 향상)
        max_memory_gb=2.8,    # 메모리 제한 약간 높임
        use_knowledge_distillation=True,
        teacher_model_name="polyglot-ko-3.8b"  # 더 큰 교사 모델
    )
    
    # 토크나이저 설정
    tokenizer_config = TokenizerConfig(
        use_korean_optimizer=True,
        jamo_separation_minimization=True,
        use_morpheme_analyzer=True,
        special_token_handling=True
    )
    
    # 추론 설정
    inference_config = InferenceConfig(
        kv_cache_size_mb=256,  # KV 캐시 크기 증가
        use_memory_mapping=True,
        enable_inplace_operations=True,
        batch_size=1,
        max_new_tokens=1024  # 더 긴 응답 생성 가능
    )
    
    # 대화 설정
    conversation_config = ConversationConfig(
        context_length=10,  # 더 긴 컨텍스트 기억
        response_quality_enhancement=True,
        default_style="casual",
        temperature=0.8,  # 약간 더 창의적인 응답
        top_p=0.92,
        repetition_penalty=1.15
    )
    
    # 아키텍처 생성
    architecture = ModelArchitecture(
        model_config=model_config,
        tokenizer_config=tokenizer_config,
        inference_config=inference_config,
        conversation_config=conversation_config
    )
    
    logger.info("ChatGPT 수준의 아키텍처 생성 완료")
    return architecture
