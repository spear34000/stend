"""
한국어 특화 초경량 AI 모델 - 유틸리티 모듈

이 모듈은 다양한 유틸리티 함수를 제공합니다:
- 메모리 모니터링
- 로깅
- 설정 관리
- 성능 측정
"""

import os
import time
import json
import psutil
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    def get_memory_usage_stats(self) -> Dict[str, float]:
        """
        메모리 사용량 통계 반환
        
        Returns:
            메모리 사용량 통계 딕셔너리
        """
        # 프로세스 메모리 사용량 확인
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # 시스템 메모리 사용량 확인
        system_memory = psutil.virtual_memory()
        
        return {
            "process_rss_mb": memory_info.rss / (1024 * 1024),
            "process_vms_mb": memory_info.vms / (1024 * 1024),
            "peak_memory_mb": self.peak_memory / (1024 * 1024),
            "system_total_mb": system_memory.total / (1024 * 1024),
            "system_available_mb": system_memory.available / (1024 * 1024),
            "system_used_percent": system_memory.percent
        }

class ConfigManager:
    """설정 관리 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        설정 관리 초기화
        
        Args:
            config_path: 설정 파일 경로 (None인 경우 기본 설정 사용)
        """
        self.config_path = config_path
        self.config = {}
        
        # 설정 로드
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.config = self.get_default_config()
        
        logger.info(f"설정 관리 초기화: {config_path if config_path else '기본 설정'}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        기본 설정 반환
        
        Returns:
            기본 설정 딕셔너리
        """
        return {
            "model_config": {
                "model_name": "polyglot-ko-410m",
                "quantization_bits": 2,
                "pruning_ratio": 0.5,
                "max_memory_gb": 2.5,
                "use_knowledge_distillation": True,
                "teacher_model_name": "polyglot-ko-1.3b",
                "device": "cpu"
            },
            "tokenizer_config": {
                "use_korean_optimizer": True,
                "jamo_separation_minimization": True,
                "use_morpheme_analyzer": True,
                "special_token_handling": True
            },
            "inference_config": {
                "kv_cache_size_mb": 128,
                "use_memory_mapping": True,
                "enable_inplace_operations": True,
                "batch_size": 1,
                "max_new_tokens": 512
            },
            "conversation_config": {
                "context_length": 5,
                "response_quality_enhancement": True,
                "default_style": "casual",
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
        }
    
    def load_config(self, config_path: str) -> None:
        """
        설정 로드
        
        Args:
            config_path: 설정 파일 경로
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            logger.info(f"설정 로드 완료: {config_path}")
        except Exception as e:
            logger.error(f"설정 로드 중 오류 발생: {str(e)}", exc_info=True)
            self.config = self.get_default_config()
    
    def save_config(self, output_path: str) -> None:
        """
        설정 저장
        
        Args:
            output_path: 출력 파일 경로
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"설정 저장 완료: {output_path}")
        except Exception as e:
            logger.error(f"설정 저장 중 오류 발생: {str(e)}", exc_info=True)
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        설정 반환
        
        Args:
            section: 설정 섹션 (None인 경우 전체 설정 반환)
        
        Returns:
            설정 딕셔너리
        """
        if section:
            return self.config.get(section, {})
        return self.config
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """
        설정 업데이트
        
        Args:
            section: 설정 섹션
            key: 설정 키
            value: 설정 값
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
        logger.info(f"설정 업데이트: {section}.{key} = {value}")

class PerformanceTracker:
    """성능 추적 클래스"""
    
    def __init__(self):
        """성능 추적 초기화"""
        self.start_time = 0
        self.end_time = 0
        self.metrics = {}
        
        logger.info("성능 추적 초기화")
    
    def start_tracking(self) -> None:
        """성능 추적 시작"""
        self.start_time = time.time()
        self.metrics = {}
        
        logger.info("성능 추적 시작")
    
    def stop_tracking(self) -> None:
        """성능 추적 종료"""
        self.end_time = time.time()
        self.metrics["elapsed_time"] = self.end_time - self.start_time
        
        logger.info(f"성능 추적 종료: 소요 시간={self.metrics['elapsed_time']:.2f}초")
    
    def add_metric(self, name: str, value: Any) -> None:
        """
        메트릭 추가
        
        Args:
            name: 메트릭 이름
            value: 메트릭 값
        """
        self.metrics[name] = value
        logger.info(f"메트릭 추가: {name} = {value}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        메트릭 반환
        
        Returns:
            메트릭 딕셔너리
        """
        return self.metrics
    
    def log_metrics(self) -> None:
        """메트릭 로깅"""
        logger.info("성능 메트릭:")
        for name, value in self.metrics.items():
            logger.info(f"  {name}: {value}")

class FileUtils:
    """파일 유틸리티 클래스"""
    
    @staticmethod
    def ensure_dir(directory: str) -> None:
        """
        디렉토리 생성
        
        Args:
            directory: 디렉토리 경로
        """
        os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def list_files(directory: str, extension: Optional[str] = None) -> List[str]:
        """
        파일 목록 반환
        
        Args:
            directory: 디렉토리 경로
            extension: 파일 확장자 (None인 경우 모든 파일 반환)
        
        Returns:
            파일 경로 리스트
        """
        if not os.path.exists(directory):
            return []
        
        files = []
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                if extension is None or file.endswith(extension):
                    files.append(file_path)
        
        return files
    
    @staticmethod
    def read_text_file(file_path: str) -> str:
        """
        텍스트 파일 읽기
        
        Args:
            file_path: 파일 경로
        
        Returns:
            파일 내용
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"파일 읽기 중 오류 발생: {str(e)}", exc_info=True)
            return ""
    
    @staticmethod
    def write_text_file(file_path: str, content: str) -> bool:
        """
        텍스트 파일 쓰기
        
        Args:
            file_path: 파일 경로
            content: 파일 내용
        
        Returns:
            성공 여부
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
        except Exception as e:
            logger.error(f"파일 쓰기 중 오류 발생: {str(e)}", exc_info=True)
            return False

class TorchUtils:
    """PyTorch 유틸리티 클래스"""
    
    @staticmethod
    def get_available_device() -> str:
        """
        사용 가능한 디바이스 반환
        
        Returns:
            디바이스 문자열 ("cuda" 또는 "cpu")
        """
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @staticmethod
    def get_gpu_memory_usage() -> Dict[str, float]:
        """
        GPU 메모리 사용량 반환
        
        Returns:
            GPU 메모리 사용량 딕셔너리
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        memory_stats = {}
        
        for i in range(torch.cuda.device_count()):
            memory_stats[f"gpu_{i}_total_mb"] = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
            memory_stats[f"gpu_{i}_reserved_mb"] = torch.cuda.memory_reserved(i) / (1024 * 1024)
            memory_stats[f"gpu_{i}_allocated_mb"] = torch.cuda.memory_allocated(i) / (1024 * 1024)
        
        return memory_stats
    
    @staticmethod
    def clear_gpu_memory() -> None:
        """GPU 메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_model_size(model: torch.nn.Module) -> Dict[str, Any]:
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
        param_size_bytes = 0
        buffer_size_bytes = 0
        
        for param in model.parameters():
            param_size_bytes += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size_bytes += buffer.nelement() * buffer.element_size()
        
        total_size_bytes = param_size_bytes + buffer_size_bytes
        
        return {
            "param_count": param_count,
            "param_size_mb": param_size_bytes / (1024 * 1024),
            "buffer_size_mb": buffer_size_bytes / (1024 * 1024),
            "total_size_mb": total_size_bytes / (1024 * 1024)
        }

class KoreanUtils:
    """한국어 유틸리티 클래스"""
    
    # 한국어 자모 관련 유틸리티
    CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    JONGSUNG = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    
    @staticmethod
    def split_syllable(char: str) -> Tuple[str, str, str]:
        """
        한글 음절을 자모로 분리
        
        Args:
            char: 한글 음절
        
        Returns:
            (초성, 중성, 종성) 튜플
        """
        if not '가' <= char <= '힣':
            return (char, '', '')
        
        char_code = ord(char) - 0xAC00
        jong_idx = char_code % 28
        jung_idx = ((char_code - jong_idx) // 28) % 21
        cho_idx = ((char_code - jong_idx) // 28) // 21
        
        return (
            KoreanUtils.CHOSUNG[cho_idx],
            KoreanUtils.JUNGSUNG[jung_idx],
            KoreanUtils.JONGSUNG[jong_idx]
        )
    
    @staticmethod
    def combine_jamo(cho: str, jung: str, jong: str = ' ') -> str:
        """
        자모를 한글 음절로 결합
        
        Args:
            cho: 초성
            jung: 중성
            jong: 종성 (기본값은 공백)
        
        Returns:
            한글 음절
        """
        try:
            cho_idx = KoreanUtils.CHOSUNG.index(cho)
            jung_idx = KoreanUtils.JUNGSUNG.index(jung)
            jong_idx = KoreanUtils.JONGSUNG.index(jong)
            
            char_code = 0xAC00 + (cho_idx * 21 + jung_idx) * 28 + jong_idx
            return chr(char_code)
        except ValueError:
            return cho + jung + jong
    
    @staticmethod
    def has_batchim(char: str) -> bool:
        """
        한글 음절의 받침 여부 확인
        
        Args:
            char: 한글 음절
        
        Returns:
            받침 여부
        """
        if not '가' <= char <= '힣':
            return False
        
        char_code = ord(char) - 0xAC00
        jong_idx = char_code % 28
        
        return jong_idx > 0
    
    @staticmethod
    def fix_josa(text: str) -> str:
        """
        조사 교정
        
        Args:
            text: 입력 텍스트
        
        Returns:
            교정된 텍스트
        """
        # 조사 교정 패턴
        patterns = [
            (r'([가-힣])이(는|가)', KoreanUtils._fix_josa_helper),
            (r'([가-힣])을(를)', KoreanUtils._fix_josa_helper),
            (r'([가-힣])과(와)', KoreanUtils._fix_josa_helper),
            (r'([가-힣])은(는)', KoreanUtils._fix_josa_helper),
            (r'([가-힣])아(야)', KoreanUtils._fix_josa_helper)
        ]
        
        # 패턴 적용
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    @staticmethod
    def _fix_josa_helper(match) -> str:
        """
        조사 교정 헬퍼 함수
        
        Args:
            match: 정규식 매치 객체
        
        Returns:
            교정된 조사
        """
        word = match.group(1)
        josa_pair = match.group(2)
        
        # 받침 여부 확인
        has_batchim = KoreanUtils.has_batchim(word[-1])
        
        # 조사 교정
        if josa_pair == '는':
            return word + ('은' if has_batchim else '는')
        elif josa_pair == '가':
            return word + ('이' if has_batchim else '가')
        elif josa_pair == '를':
            return word + ('을' if has_batchim else '를')
        elif josa_pair == '와':
            return word + ('과' if has_batchim else '와')
        elif josa_pair == '야':
            return word + ('아' if has_batchim else '야')
        
        return match.group(0)
