"""
한국어 특화 초경량 AI 모델 구조 정의
- 모듈화된 구조 설계
- 명확한 의존성 관리
- 확장 가능한 아키텍처
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Callable

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """모델 설정 클래스"""
    model_name: str = "EleutherAI/polyglot-ko-1.3b"
    model_type: str = "causal_lm"
    quantization_bits: int = 4  # 2, 3, 4, 8 중 선택
    use_kv_cache: bool = True
    use_flash_attention: bool = True
    use_group_query_attention: bool = True
    use_pruning: bool = False
    pruning_ratio: float = 0.3
    use_knowledge_distillation: bool = False
    teacher_model_name: Optional[str] = None
    max_memory_gb: float = 2.5
    device: str = "cpu"
    cache_dir: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "quantization_bits": self.quantization_bits,
            "use_kv_cache": self.use_kv_cache,
            "use_flash_attention": self.use_flash_attention,
            "use_group_query_attention": self.use_group_query_attention,
            "use_pruning": self.use_pruning,
            "pruning_ratio": self.pruning_ratio,
            "use_knowledge_distillation": self.use_knowledge_distillation,
            "teacher_model_name": self.teacher_model_name,
            "max_memory_gb": self.max_memory_gb,
            "device": self.device,
            "cache_dir": self.cache_dir
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """딕셔너리에서 설정 생성"""
        return cls(**config_dict)
    
    def save(self, file_path: str) -> None:
        """설정을 JSON 파일로 저장"""
        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"모델 설정이 {file_path}에 저장되었습니다.")
    
    @classmethod
    def load(cls, file_path: str) -> 'ModelConfig':
        """JSON 파일에서 설정 로드"""
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        logger.info(f"모델 설정이 {file_path}에서 로드되었습니다.")
        return cls.from_dict(config_dict)

@dataclass
class TokenizerConfig:
    """토크나이저 설정 클래스"""
    tokenizer_name: str = "EleutherAI/polyglot-ko-1.3b"
    use_korean_optimization: bool = True
    padding_side: str = "left"
    truncation_side: str = "right"
    max_length: int = 2048
    add_special_tokens: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "tokenizer_name": self.tokenizer_name,
            "use_korean_optimization": self.use_korean_optimization,
            "padding_side": self.padding_side,
            "truncation_side": self.truncation_side,
            "max_length": self.max_length,
            "add_special_tokens": self.add_special_tokens
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TokenizerConfig':
        """딕셔너리에서 설정 생성"""
        return cls(**config_dict)

@dataclass
class InferenceConfig:
    """추론 설정 클래스"""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_new_tokens: int = 256
    use_memory_optimization: bool = True
    max_kv_cache_size: int = 512
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "use_memory_optimization": self.use_memory_optimization,
            "max_kv_cache_size": self.max_kv_cache_size
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'InferenceConfig':
        """딕셔너리에서 설정 생성"""
        return cls(**config_dict)

@dataclass
class ConversationConfig:
    """대화 설정 클래스"""
    max_history_tokens: int = 768
    default_style: str = "casual"
    enhance_korean_response: bool = True
    maintain_context: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "max_history_tokens": self.max_history_tokens,
            "default_style": self.default_style,
            "enhance_korean_response": self.enhance_korean_response,
            "maintain_context": self.maintain_context
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConversationConfig':
        """딕셔너리에서 설정 생성"""
        return cls(**config_dict)

class ModuleRegistry:
    """모듈 레지스트리 클래스"""
    
    def __init__(self):
        """모듈 레지스트리 초기화"""
        self.modules = {}
        logger.info("모듈 레지스트리 초기화 완료")
    
    def register(self, name: str, module: Any) -> None:
        """모듈 등록"""
        self.modules[name] = module
        logger.info(f"모듈 등록: {name}")
    
    def get(self, name: str) -> Any:
        """모듈 가져오기"""
        if name not in self.modules:
            logger.error(f"등록되지 않은 모듈: {name}")
            return None
        return self.modules[name]
    
    def list_modules(self) -> List[str]:
        """등록된 모듈 목록 반환"""
        return list(self.modules.keys())

class ModelArchitecture:
    """모델 아키텍처 클래스"""
    
    def __init__(self, model_config: ModelConfig, tokenizer_config: TokenizerConfig,
                 inference_config: InferenceConfig, conversation_config: ConversationConfig):
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
        
        self.registry = ModuleRegistry()
        
        logger.info("모델 아키텍처 초기화 완료")
    
    def initialize(self) -> bool:
        """모델 아키텍처 초기화"""
        try:
            # 모듈 초기화 및 등록
            self._initialize_tokenizer()
            self._initialize_model()
            self._initialize_inference_optimizer()
            self._initialize_conversation_optimizer()
            
            logger.info("모델 아키텍처 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"모델 아키텍처 초기화 중 오류 발생: {str(e)}", exc_info=True)
            return False
    
    def _initialize_tokenizer(self) -> None:
        """토크나이저 초기화"""
        from tokenizer_optimizer import KoreanTokenizerOptimizer
        
        tokenizer_optimizer = KoreanTokenizerOptimizer(self.tokenizer_config.tokenizer_name)
        tokenizer = tokenizer_optimizer.load_tokenizer()
        
        # 토크나이저 설정 적용
        tokenizer.padding_side = self.tokenizer_config.padding_side
        tokenizer.truncation_side = self.tokenizer_config.truncation_side
        
        self.registry.register("tokenizer_optimizer", tokenizer_optimizer)
        self.registry.register("tokenizer", tokenizer)
        
        logger.info("토크나이저 초기화 완료")
    
    def _initialize_model(self) -> None:
        """모델 초기화"""
        from model_loader import KoreanLightweightModel
        
        model_loader = KoreanLightweightModel(
            model_name=self.model_config.model_name,
            use_4bit=self.model_config.quantization_bits == 4,
            device=self.model_config.device
        )
        
        # 모델 로드 및 최적화
        model, _ = model_loader.load_model(use_sharding=True)
        model_loader.optimize_for_inference()
        
        self.registry.register("model_loader", model_loader)
        self.registry.register("model", model)
        
        logger.info("모델 초기화 완료")
    
    def _initialize_inference_optimizer(self) -> None:
        """추론 최적화 초기화"""
        from inference_optimizer import InferenceOptimizer
        
        inference_optimizer = InferenceOptimizer(
            max_memory_gb=self.model_config.max_memory_gb,
            max_cache_size=self.inference_config.max_kv_cache_size
        )
        
        self.registry.register("inference_optimizer", inference_optimizer)
        
        logger.info("추론 최적화 초기화 완료")
    
    def _initialize_conversation_optimizer(self) -> None:
        """대화 최적화 초기화"""
        from conversation_optimizer import ConversationOptimizer
        
        model_loader = self.registry.get("model_loader")
        tokenizer_optimizer = self.registry.get("tokenizer_optimizer")
        
        conversation_optimizer = ConversationOptimizer(
            model_loader,
            tokenizer_optimizer
        )
        
        # 대화 설정 적용
        conversation_optimizer.max_history_tokens = self.conversation_config.max_history_tokens
        conversation_optimizer.set_conversation_style(self.conversation_config.default_style)
        
        self.registry.register("conversation_optimizer", conversation_optimizer)
        
        logger.info("대화 최적화 초기화 완료")
    
    def get_module(self, name: str) -> Any:
        """모듈 가져오기"""
        return self.registry.get(name)
    
    def save_config(self, directory: str) -> None:
        """설정 저장"""
        os.makedirs(directory, exist_ok=True)
        
        self.model_config.save(os.path.join(directory, "model_config.json"))
        
        # 다른 설정도 저장
        import json
        with open(os.path.join(directory, "tokenizer_config.json"), 'w', encoding='utf-8') as f:
            json.dump(self.tokenizer_config.to_dict(), f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(directory, "inference_config.json"), 'w', encoding='utf-8') as f:
            json.dump(self.inference_config.to_dict(), f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(directory, "conversation_config.json"), 'w', encoding='utf-8') as f:
            json.dump(self.conversation_config.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"모델 설정이 {directory}에 저장되었습니다.")
    
    @classmethod
    def load_config(cls, directory: str) -> 'ModelArchitecture':
        """설정 로드"""
        model_config = ModelConfig.load(os.path.join(directory, "model_config.json"))
        
        # 다른 설정도 로드
        import json
        with open(os.path.join(directory, "tokenizer_config.json"), 'r', encoding='utf-8') as f:
            tokenizer_config = TokenizerConfig.from_dict(json.load(f))
        
        with open(os.path.join(directory, "inference_config.json"), 'r', encoding='utf-8') as f:
            inference_config = InferenceConfig.from_dict(json.load(f))
        
        with open(os.path.join(directory, "conversation_config.json"), 'r', encoding='utf-8') as f:
            conversation_config = ConversationConfig.from_dict(json.load(f))
        
        logger.info(f"모델 설정이 {directory}에서 로드되었습니다.")
        return cls(model_config, tokenizer_config, inference_config, conversation_config)

def create_default_architecture() -> ModelArchitecture:
    """기본 모델 아키텍처 생성"""
    model_config = ModelConfig()
    tokenizer_config = TokenizerConfig()
    inference_config = InferenceConfig()
    conversation_config = ConversationConfig()
    
    return ModelArchitecture(model_config, tokenizer_config, inference_config, conversation_config)

def create_extreme_lightweight_architecture() -> ModelArchitecture:
    """초경량 모델 아키텍처 생성"""
    model_config = ModelConfig(
        model_name="EleutherAI/polyglot-ko-1.3b",
        quantization_bits=2,  # 2비트 양자화
        use_kv_cache=True,
        use_flash_attention=True,
        use_group_query_attention=True,
        use_pruning=True,
        pruning_ratio=0.5,  # 50% 프루닝
        max_memory_gb=1.5  # 메모리 제한 더 낮게 설정
    )
    
    tokenizer_config = TokenizerConfig(
        use_korean_optimization=True
    )
    
    inference_config = InferenceConfig(
        use_memory_optimization=True,
        max_kv_cache_size=256  # KV 캐시 크기 축소
    )
    
    conversation_config = ConversationConfig(
        max_history_tokens=512  # 이력 토큰 수 축소
    )
    
    return ModelArchitecture(model_config, tokenizer_config, inference_config, conversation_config)
