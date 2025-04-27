"""
한국어 특화 경량화 AI 모델 대화 최적화 모듈
- 자연스러운 한국어 대화 생성 최적화
- 대화 맥락 유지 및 일관성 향상
- 한국어 문법 및 어휘 정확성 개선
"""

import os
import re
import json
import logging
from typing import List, Dict, Tuple, Optional, Union

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationOptimizer:
    """한국어 대화 최적화 클래스"""
    
    def __init__(self, model_loader, tokenizer_optimizer):
        """
        대화 최적화 초기화
        
        Args:
            model_loader: 모델 로더 객체
            tokenizer_optimizer: 토크나이저 최적화 객체
        """
        self.model_loader = model_loader
        self.tokenizer_optimizer = tokenizer_optimizer
        self.conversation_history = []
        self.max_history_tokens = 512  # 대화 이력 최대 토큰 수
        
        # 한국어 대화 스타일 설정
        self.conversation_styles = {
            "formal": {"prefix": "정중하게 답변해주세요: ", "suffix": ""},
            "casual": {"prefix": "친근하게 답변해주세요: ", "suffix": ""},
            "professional": {"prefix": "전문가처럼 답변해주세요: ", "suffix": ""}
        }
        
        # 기본 대화 스타일
        self.current_style = "casual"
        
        logger.info("대화 최적화 초기화 완료")
    
    def set_conversation_style(self, style):
        """
        대화 스타일 설정
        
        Args:
            style: 대화 스타일 ("formal", "casual", "professional")
        """
        if style in self.conversation_styles:
            self.current_style = style
            logger.info(f"대화 스타일 설정: {style}")
        else:
            logger.warning(f"지원하지 않는 대화 스타일: {style}, 기본값 'casual' 사용")
            self.current_style = "casual"
    
    def add_to_history(self, role, text):
        """
        대화 이력에 추가
        
        Args:
            role: 발화자 역할 ("user" 또는 "assistant")
            text: 발화 텍스트
        """
        self.conversation_history.append({"role": role, "content": text})
        
        # 대화 이력 토큰 수 제한
        self._trim_conversation_history()
    
    def _trim_conversation_history(self):
        """대화 이력 토큰 수 제한"""
        if not self.conversation_history:
            return
        
        # 현재 대화 이력의 토큰 수 계산
        history_text = " ".join([item["content"] for item in self.conversation_history])
        tokens = self.tokenizer_optimizer.tokenizer.tokenize(history_text)
        
        # 토큰 수가 제한을 초과하면 가장 오래된 대화부터 제거
        while len(tokens) > self.max_history_tokens and len(self.conversation_history) > 1:
            self.conversation_history.pop(0)
            history_text = " ".join([item["content"] for item in self.conversation_history])
            tokens = self.tokenizer_optimizer.tokenizer.tokenize(history_text)
    
    def format_prompt_with_history(self, user_input):
        """
        대화 이력을 포함한 프롬프트 형식화
        
        Args:
            user_input: 사용자 입력 텍스트
        
        Returns:
            형식화된 프롬프트
        """
        # 사용자 입력을 대화 이력에 추가
        self.add_to_history("user", user_input)
        
        # 대화 이력 형식화
        formatted_history = ""
        for item in self.conversation_history:
            role_prefix = "사용자: " if item["role"] == "user" else "AI: "
            formatted_history += f"{role_prefix}{item['content']}\n"
        
        # 현재 스타일에 맞는 프롬프트 생성
        style_prefix = self.conversation_styles[self.current_style]["prefix"]
        style_suffix = self.conversation_styles[self.current_style]["suffix"]
        
        prompt = f"{formatted_history}AI: "
        
        # 스타일 프리픽스가 있으면 전체 프롬프트 앞에 추가
        if style_prefix:
            prompt = f"{style_prefix}\n{prompt}"
        
        logger.info(f"형식화된 프롬프트 생성 완료 (길이: {len(prompt)})")
        return prompt
    
    def enhance_korean_response(self, response):
        """
        한국어 응답 품질 향상
        
        Args:
            response: 원본 응답 텍스트
        
        Returns:
            향상된 응답 텍스트
        """
        # 응답에서 "AI:" 프리픽스 제거
        response = re.sub(r'^AI:\s*', '', response)
        
        # 불필요한 공백 정리
        response = re.sub(r'\s+', ' ', response).strip()
        
        # 한국어 문장 부호 교정
        response = self._fix_korean_punctuation(response)
        
        # 반복 문장 제거
        response = self._remove_repetitions(response)
        
        return response
    
    def _fix_korean_punctuation(self, text):
        """
        한국어 문장 부호 교정
        
        Args:
            text: 원본 텍스트
        
        Returns:
            교정된 텍스트
        """
        # 영어 따옴표를 한글 따옴표로 변환
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")
        
        # 마침표 뒤에 공백 추가
        text = re.sub(r'([.!?])', r'\1 ', text)
        
        # 중복된 문장 부호 정리
        text = re.sub(r'([.!?])\s*\1+', r'\1', text)
        
        return text
    
    def _remove_repetitions(self, text):
        """
        반복 문장 제거
        
        Args:
            text: 원본 텍스트
        
        Returns:
            반복이 제거된 텍스트
        """
        # 문장 분리
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # 중복 문장 제거
        unique_sentences = []
        for sentence in sentences:
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        # 문장 재결합
        return ' '.join(unique_sentences)
    
    def generate_response(self, user_input, temperature=0.7, max_length=256):
        """
        사용자 입력에 대한 응답 생성
        
        Args:
            user_input: 사용자 입력 텍스트
            temperature: 생성 온도 (높을수록 다양한 결과)
            max_length: 최대 생성 길이
        
        Returns:
            생성된 응답 텍스트
        """
        # 대화 이력을 포함한 프롬프트 형식화
        prompt = self.format_prompt_with_history(user_input)
        
        # 응답 생성
        generated_text = self.model_loader.generate_text(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9
        )
        
        # 프롬프트 이후의 텍스트만 추출
        response = generated_text[len(prompt):]
        
        # 응답 품질 향상
        enhanced_response = self.enhance_korean_response(response)
        
        # 응답을 대화 이력에 추가
        self.add_to_history("assistant", enhanced_response)
        
        logger.info(f"응답 생성 완료 (길이: {len(enhanced_response)})")
        return enhanced_response
    
    def save_conversation_history(self, file_path):
        """
        대화 이력 저장
        
        Args:
            file_path: 저장할 파일 경로
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        
        logger.info(f"대화 이력 저장 완료: {file_path}")
    
    def load_conversation_history(self, file_path):
        """
        대화 이력 로드
        
        Args:
            file_path: 로드할 파일 경로
        """
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            
            logger.info(f"대화 이력 로드 완료: {file_path}")
        else:
            logger.warning(f"대화 이력 파일이 존재하지 않습니다: {file_path}")
    
    def clear_conversation_history(self):
        """대화 이력 초기화"""
        self.conversation_history = []
        logger.info("대화 이력 초기화 완료")
