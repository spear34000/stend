"""
한국어 특화 초경량 AI 모델 - 토크나이저 최적화 모듈

이 모듈은 한국어에 특화된 토크나이저 최적화 기법을 구현합니다:
- 한국어 형태소 분석 기반 토크나이저
- 자모 분리 최소화 로직
- 특수 토큰 처리
"""

import os
import re
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 한국어 자모 관련 유틸리티
CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

class KoreanTokenizerOptimizer:
    """한국어 토크나이저 최적화 클래스"""
    
    def __init__(self, model_name_or_path: str):
        """
        한국어 토크나이저 최적화 초기화
        
        Args:
            model_name_or_path: 모델 이름 또는 경로
        """
        self.model_name_or_path = model_name_or_path
        self.tokenizer = None
        self.special_tokens = {
            "pad_token": "[PAD]",
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "[UNK]",
            "mask_token": "[MASK]"
        }
        
        # 한국어 형태소 분석기 초기화 (필요한 경우)
        self.morpheme_analyzer = None
        
        logger.info(f"한국어 토크나이저 최적화 초기화: {model_name_or_path}")
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        토크나이저 로드 및 최적화
        
        Returns:
            최적화된 토크나이저
        """
        logger.info(f"토크나이저 로드 시작: {self.model_name_or_path}")
        
        try:
            # 기본 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            
            # 특수 토큰 설정
            self._setup_special_tokens()
            
            # 한국어 최적화 적용
            self._apply_korean_optimizations()
            
            logger.info("토크나이저 로드 및 최적화 완료")
            return self.tokenizer
            
        except Exception as e:
            logger.error(f"토크나이저 로드 중 오류 발생: {str(e)}", exc_info=True)
            raise
    
    def _setup_special_tokens(self) -> None:
        """특수 토큰 설정"""
        # 필요한 특수 토큰 추가
        for token_name, token_value in self.special_tokens.items():
            if getattr(self.tokenizer, token_name) is None:
                setattr(self.tokenizer, token_name, token_value)
        
        # 패딩 토큰이 없는 경우 EOS 토큰으로 대체
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _apply_korean_optimizations(self) -> None:
        """한국어 최적화 적용"""
        # 자모 분리 최소화 처리
        self._apply_jamo_separation_minimization()
        
        # 형태소 분석기 초기화 (필요한 경우)
        self._initialize_morpheme_analyzer()
    
    def _apply_jamo_separation_minimization(self) -> None:
        """자모 분리 최소화 적용"""
        # 원래 토크나이저의 토큰화 메서드 저장
        original_tokenize = self.tokenizer.tokenize
        
        # 자모 분리 최소화 로직이 적용된 토큰화 메서드 정의
        def optimized_tokenize(text, *args, **kwargs):
            # 자모 분리 패턴 감지 및 처리
            processed_text = self._minimize_jamo_separation(text)
            # 원래 토큰화 메서드 호출
            return original_tokenize(processed_text, *args, **kwargs)
        
        # 토크나이저의 토큰화 메서드 교체
        self.tokenizer.tokenize = optimized_tokenize
    
    def _minimize_jamo_separation(self, text: str) -> str:
        """
        자모 분리 최소화 처리
        
        Args:
            text: 입력 텍스트
        
        Returns:
            처리된 텍스트
        """
        # 자모 분리 패턴 감지
        jamo_pattern = re.compile(r'([ㄱ-ㅎ][ㅏ-ㅣ])')
        
        # 자모 분리된 패턴을 결합
        def replace_jamo(match):
            jamo = match.group(1)
            # 초성과 중성을 결합하여 한글 문자로 변환
            if len(jamo) >= 2:
                cho_idx = CHOSUNG.index(jamo[0]) if jamo[0] in CHOSUNG else 0
                jung_idx = JUNGSUNG.index(jamo[1]) if jamo[1] in JUNGSUNG else 0
                # 유니코드 한글 시작 코드: 0xAC00
                # 한글 = 0xAC00 + (초성 * 21 + 중성) * 28 + 종성
                unicode_idx = 0xAC00 + (cho_idx * 21 + jung_idx) * 28
                return chr(unicode_idx)
            return jamo
        
        # 자모 분리 패턴 처리
        processed_text = jamo_pattern.sub(replace_jamo, text)
        return processed_text
    
    def _initialize_morpheme_analyzer(self) -> None:
        """형태소 분석기 초기화"""
        try:
            # 형태소 분석기 라이브러리 임포트 시도
            from konlpy.tag import Mecab
            self.morpheme_analyzer = Mecab()
            logger.info("형태소 분석기(Mecab) 초기화 완료")
        except ImportError:
            logger.warning("형태소 분석기(Mecab) 라이브러리를 찾을 수 없습니다. 형태소 분석 기능이 비활성화됩니다.")
        except Exception as e:
            logger.warning(f"형태소 분석기 초기화 중 오류 발생: {str(e)}. 형태소 분석 기능이 비활성화됩니다.")
    
    def analyze_morphemes(self, text: str) -> List[str]:
        """
        형태소 분석 수행
        
        Args:
            text: 입력 텍스트
        
        Returns:
            형태소 리스트
        """
        if self.morpheme_analyzer is None:
            # 형태소 분석기가 없는 경우 공백으로 분리
            return text.split()
        
        try:
            # 형태소 분석 수행
            morphemes = self.morpheme_analyzer.morphs(text)
            return morphemes
        except Exception as e:
            logger.warning(f"형태소 분석 중 오류 발생: {str(e)}. 공백으로 분리합니다.")
            return text.split()
    
    def tokenize_with_morpheme_analysis(self, text: str) -> List[str]:
        """
        형태소 분석 기반 토큰화
        
        Args:
            text: 입력 텍스트
        
        Returns:
            토큰 리스트
        """
        # 형태소 분석
        morphemes = self.analyze_morphemes(text)
        
        # 각 형태소를 토큰화
        tokens = []
        for morpheme in morphemes:
            morpheme_tokens = self.tokenizer.tokenize(morpheme)
            tokens.extend(morpheme_tokens)
        
        return tokens
    
    def optimize_tokenizer_vocabulary(self) -> None:
        """토크나이저 어휘 최적화"""
        # 이 메서드는 토크나이저의 어휘를 한국어에 맞게 최적화하는 로직을 구현할 수 있음
        # 현재는 구현되지 않음
        pass
    
    def save_optimized_tokenizer(self, output_dir: str) -> str:
        """
        최적화된 토크나이저 저장
        
        Args:
            output_dir: 출력 디렉토리
        
        Returns:
            저장된 경로
        """
        if self.tokenizer is None:
            logger.error("토크나이저가 초기화되지 않았습니다.")
            return ""
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"최적화된 토크나이저 저장 완료: {output_dir}")
            return output_dir
        except Exception as e:
            logger.error(f"토크나이저 저장 중 오류 발생: {str(e)}", exc_info=True)
            return ""

class ChatGPTLevelTokenizerOptimizer(KoreanTokenizerOptimizer):
    """ChatGPT 수준의 한국어 토크나이저 최적화 클래스"""
    
    def __init__(self, model_name_or_path: str):
        """
        ChatGPT 수준의 한국어 토크나이저 최적화 초기화
        
        Args:
            model_name_or_path: 모델 이름 또는 경로
        """
        super().__init__(model_name_or_path)
        
        # 추가 특수 토큰 설정
        self.special_tokens.update({
            "sep_token": "[SEP]",
            "cls_token": "[CLS]",
            "system_token": "<|system|>",
            "user_token": "<|user|>",
            "assistant_token": "<|assistant|>"
        })
        
        logger.info(f"ChatGPT 수준의 한국어 토크나이저 최적화 초기화: {model_name_or_path}")
    
    def _apply_korean_optimizations(self) -> None:
        """한국어 최적화 적용 (확장)"""
        # 기본 최적화 적용
        super()._apply_korean_optimizations()
        
        # 추가 최적화 적용
        self._apply_advanced_korean_optimizations()
    
    def _apply_advanced_korean_optimizations(self) -> None:
        """고급 한국어 최적화 적용"""
        # 대화 형식 최적화
        self._optimize_for_conversation()
        
        # 한국어 특화 토큰화 최적화
        self._optimize_korean_tokenization()
    
    def _optimize_for_conversation(self) -> None:
        """대화 형식 최적화"""
        # 대화 형식 토큰화 최적화 로직
        # 예: 시스템/사용자/어시스턴트 구분 토큰 처리
        pass
    
    def _optimize_korean_tokenization(self) -> None:
        """한국어 특화 토큰화 최적화"""
        # 한국어 특화 토큰화 최적화 로직
        # 예: 한국어 고유 표현, 존댓말/반말 처리 등
        pass
    
    def tokenize_conversation(self, messages: List[Dict[str, str]]) -> List[str]:
        """
        대화 형식 토큰화
        
        Args:
            messages: 대화 메시지 리스트 (역할과 내용 포함)
        
        Returns:
            토큰 리스트
        """
        if self.tokenizer is None:
            logger.error("토크나이저가 초기화되지 않았습니다.")
            return []
        
        # 역할별 토큰 매핑
        role_tokens = {
            "system": self.special_tokens.get("system_token", "<|system|>"),
            "user": self.special_tokens.get("user_token", "<|user|>"),
            "assistant": self.special_tokens.get("assistant_token", "<|assistant|>")
        }
        
        # 대화 형식 구성
        conversation_text = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            role_token = role_tokens.get(role, role_tokens["user"])
            conversation_text += f"{role_token}\n{content}\n"
        
        # 토큰화
        return self.tokenizer.tokenize(conversation_text)
