"""
한국어 특화 경량화 AI 모델 토크나이저 최적화 모듈
- 한국어 형태소 분석 기반 토크나이저 최적화
- 토큰화 효율성 향상
- 한글 자모 분리 최소화
"""

import os
import re
import logging
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoTokenizer, PreTrainedTokenizer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KoreanTokenizerOptimizer:
    """한국어 토크나이저 최적화 클래스"""
    
    def __init__(self, base_tokenizer_name="EleutherAI/polyglot-ko-1.3b"):
        """
        한국어 토크나이저 최적화 초기화
        
        Args:
            base_tokenizer_name: 기본 토크나이저 이름 (Hugging Face 모델 ID)
        """
        self.base_tokenizer_name = base_tokenizer_name
        self.tokenizer = None
        self.common_korean_phrases = {}  # 자주 사용되는 한국어 표현 사전
        
        logger.info(f"한국어 토크나이저 최적화 초기화: {base_tokenizer_name}")
    
    def load_tokenizer(self):
        """
        토크나이저 로드
        
        Returns:
            최적화된 토크나이저
        """
        logger.info(f"토크나이저 로드: {self.base_tokenizer_name}")
        
        # 기본 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_tokenizer_name)
        
        # 토크나이저 설정 최적화
        self._optimize_tokenizer_settings()
        
        logger.info("토크나이저 로드 완료")
        return self.tokenizer
    
    def _optimize_tokenizer_settings(self):
        """토크나이저 설정 최적화"""
        if self.tokenizer is None:
            logger.error("토크나이저가 로드되지 않았습니다.")
            return
        
        # 패딩 설정
        self.tokenizer.padding_side = "left"
        
        # PAD 토큰이 없으면 EOS 토큰으로 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 특수 토큰 처리 설정
        self.tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>"
        })
        
        logger.info("토크나이저 설정 최적화 완료")
    
    def analyze_tokenization_efficiency(self, text):
        """
        토큰화 효율성 분석
        
        Args:
            text: 분석할 텍스트
        
        Returns:
            토큰화 효율성 분석 결과 딕셔너리
        """
        if self.tokenizer is None:
            logger.error("토크나이저가 로드되지 않았습니다.")
            return {}
        
        # 토큰화
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text)
        
        # 한글 문자 수
        korean_char_count = len(re.findall(r'[가-힣]', text))
        
        # 토큰 당 한글 문자 비율
        korean_char_per_token = korean_char_count / len(tokens) if tokens else 0
        
        # 자모 분리 토큰 수
        jamo_tokens = [t for t in tokens if re.search(r'[ㄱ-ㅎㅏ-ㅣ]', t)]
        jamo_token_ratio = len(jamo_tokens) / len(tokens) if tokens else 0
        
        result = {
            "text_length": len(text),
            "token_count": len(tokens),
            "korean_char_count": korean_char_count,
            "korean_char_per_token": korean_char_per_token,
            "jamo_token_count": len(jamo_tokens),
            "jamo_token_ratio": jamo_token_ratio,
            "tokens": tokens
        }
        
        logger.info(f"토큰화 효율성 분석: {len(text)}자 -> {len(tokens)}토큰 (한글 문자/토큰: {korean_char_per_token:.2f})")
        return result
    
    def build_common_phrases_vocab(self, texts, min_freq=5):
        """
        자주 사용되는 한국어 표현 어휘 구축
        
        Args:
            texts: 학습 텍스트 리스트
            min_freq: 최소 출현 빈도
        """
        if self.tokenizer is None:
            logger.error("토크나이저가 로드되지 않았습니다.")
            return
        
        logger.info(f"자주 사용되는 한국어 표현 어휘 구축 시작: {len(texts)} 텍스트")
        
        # 한국어 표현 빈도 계산
        phrase_freq = {}
        for text in texts:
            # 2~4 글자 한국어 표현 추출
            for n in range(2, 5):
                for i in range(len(text) - n + 1):
                    phrase = text[i:i+n]
                    # 한글로만 구성된 표현만 포함
                    if re.match(r'^[가-힣]+$', phrase):
                        phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1
        
        # 최소 빈도 이상인 표현만 선택
        self.common_korean_phrases = {
            phrase: freq for phrase, freq in phrase_freq.items() 
            if freq >= min_freq
        }
        
        logger.info(f"자주 사용되는 한국어 표현 {len(self.common_korean_phrases)}개 구축 완료")
    
    def optimize_tokenization(self, text):
        """
        토큰화 최적화
        
        Args:
            text: 최적화할 텍스트
        
        Returns:
            최적화된 토큰 ID 리스트
        """
        if self.tokenizer is None:
            logger.error("토크나이저가 로드되지 않았습니다.")
            return []
        
        # 자모 분리 최소화를 위한 전처리
        processed_text = self._preprocess_for_jamo_minimization(text)
        
        # 토큰화
        token_ids = self.tokenizer.encode(processed_text)
        
        logger.info(f"토큰화 최적화: {len(text)}자 -> {len(token_ids)}토큰")
        return token_ids
    
    def _preprocess_for_jamo_minimization(self, text):
        """
        자모 분리 최소화를 위한 전처리
        
        Args:
            text: 전처리할 텍스트
        
        Returns:
            전처리된 텍스트
        """
        # 자주 사용되는 한국어 표현 처리
        if self.common_korean_phrases:
            for phrase in sorted(self.common_korean_phrases.keys(), key=len, reverse=True):
                # 공백 추가로 토큰화 유도
                if phrase in text:
                    text = text.replace(phrase, f" {phrase} ")
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def save_optimized_tokenizer(self, save_dir):
        """
        최적화된 토크나이저 저장
        
        Args:
            save_dir: 저장 디렉토리 경로
        """
        if self.tokenizer is None:
            logger.error("토크나이저가 로드되지 않았습니다.")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        self.tokenizer.save_pretrained(save_dir)
        
        # 자주 사용되는 한국어 표현 사전 저장
        if self.common_korean_phrases:
            import json
            with open(os.path.join(save_dir, "common_korean_phrases.json"), "w", encoding="utf-8") as f:
                json.dump(self.common_korean_phrases, f, ensure_ascii=False, indent=2)
        
        logger.info(f"최적화된 토크나이저 저장 완료: {save_dir}")
