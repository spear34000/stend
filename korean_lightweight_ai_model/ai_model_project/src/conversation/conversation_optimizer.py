"""
한국어 특화 초경량 AI 모델 - 대화 최적화 모듈

이 모듈은 자연스러운 한국어 대화를 위한 최적화 기법을 구현합니다:
- 대화 맥락 유지
- 한국어 응답 품질 향상
- 다양한 대화 스타일 지원
"""

import os
import re
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationOptimizer:
    """대화 최적화 클래스"""
    
    def __init__(self, 
                 context_length: int = 5,
                 response_quality_enhancement: bool = True,
                 default_style: str = "casual"):
        """
        대화 최적화 초기화
        
        Args:
            context_length: 대화 컨텍스트 길이
            response_quality_enhancement: 응답 품질 향상 여부
            default_style: 기본 대화 스타일 ("formal", "casual", "professional")
        """
        self.context_length = context_length
        self.response_quality_enhancement = response_quality_enhancement
        self.default_style = default_style
        self.conversation_history = []
        
        # 대화 스타일 매핑
        self.style_mapping = {
            "formal": self._apply_formal_style,
            "casual": self._apply_casual_style,
            "professional": self._apply_professional_style
        }
        
        logger.info(f"대화 최적화 초기화: 컨텍스트 길이={context_length}, 스타일={default_style}")
    
    def process_conversation(self, user_input: str, model_response: str, 
                            style: Optional[str] = None) -> str:
        """
        대화 처리
        
        Args:
            user_input: 사용자 입력
            model_response: 모델 응답
            style: 대화 스타일 (None인 경우 기본 스타일 사용)
        
        Returns:
            처리된 응답
        """
        logger.info("대화 처리 시작")
        
        try:
            # 대화 이력 업데이트
            self._update_conversation_history(user_input, model_response)
            
            # 응답 품질 향상 (설정에 따라)
            if self.response_quality_enhancement:
                processed_response = self._enhance_response_quality(model_response)
            else:
                processed_response = model_response
            
            # 대화 스타일 적용
            current_style = style if style else self.default_style
            if current_style in self.style_mapping:
                processed_response = self.style_mapping[current_style](processed_response)
            
            logger.info("대화 처리 완료")
            return processed_response
            
        except Exception as e:
            logger.error(f"대화 처리 중 오류 발생: {str(e)}", exc_info=True)
            return model_response
    
    def _update_conversation_history(self, user_input: str, model_response: str) -> None:
        """
        대화 이력 업데이트
        
        Args:
            user_input: 사용자 입력
            model_response: 모델 응답
        """
        # 대화 이력에 추가
        self.conversation_history.append({
            "user": user_input,
            "assistant": model_response
        })
        
        # 대화 이력 길이 제한
        if len(self.conversation_history) > self.context_length:
            self.conversation_history = self.conversation_history[-self.context_length:]
    
    def _enhance_response_quality(self, response: str) -> str:
        """
        응답 품질 향상
        
        Args:
            response: 원본 응답
        
        Returns:
            품질이 향상된 응답
        """
        # 1. 문장 부호 교정
        response = self._fix_punctuation(response)
        
        # 2. 반복 제거
        response = self._remove_repetition(response)
        
        # 3. 한국어 문법 교정
        response = self._fix_korean_grammar(response)
        
        return response
    
    def _fix_punctuation(self, text: str) -> str:
        """
        문장 부호 교정
        
        Args:
            text: 입력 텍스트
        
        Returns:
            교정된 텍스트
        """
        # 문장 부호 교정 로직
        # 예: 연속된 마침표 교정, 띄어쓰기 교정 등
        
        # 연속된 마침표 교정
        text = re.sub(r'\.{2,}', '...', text)
        
        # 마침표 뒤에 공백 추가
        text = re.sub(r'\.([가-힣A-Za-z0-9])', '. \\1', text)
        
        # 쉼표 뒤에 공백 추가
        text = re.sub(r',([가-힣A-Za-z0-9])', ', \\1', text)
        
        return text
    
    def _remove_repetition(self, text: str) -> str:
        """
        반복 제거
        
        Args:
            text: 입력 텍스트
        
        Returns:
            반복이 제거된 텍스트
        """
        # 반복 제거 로직
        # 예: 동일한 문장이나 구문 반복 제거
        
        # 문장 분리
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # 중복 문장 제거
        unique_sentences = []
        for sentence in sentences:
            if sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        # 문장 결합
        return ' '.join(unique_sentences)
    
    def _fix_korean_grammar(self, text: str) -> str:
        """
        한국어 문법 교정
        
        Args:
            text: 입력 텍스트
        
        Returns:
            교정된 텍스트
        """
        # 한국어 문법 교정 로직
        # 예: 조사 사용 교정, 어미 교정 등
        
        # 조사 교정 (예시)
        text = re.sub(r'([가-힣])이(는|가)', self._fix_josa, text)
        
        return text
    
    def _fix_josa(self, match) -> str:
        """
        조사 교정 헬퍼 함수
        
        Args:
            match: 정규식 매치 객체
        
        Returns:
            교정된 조사
        """
        word = match.group(1)
        josa = match.group(2)
        
        # 받침 여부 확인
        has_batchim = (ord(word[-1]) - 0xAC00) % 28 > 0
        
        # 조사 교정
        if josa == '는':
            return word + ('은' if has_batchim else '는')
        elif josa == '가':
            return word + ('이' if has_batchim else '가')
        
        return match.group(0)
    
    def _apply_formal_style(self, text: str) -> str:
        """
        정중한 스타일 적용
        
        Args:
            text: 입력 텍스트
        
        Returns:
            정중한 스타일이 적용된 텍스트
        """
        # 정중한 스타일 적용 로직
        # 예: 존댓말 변환, 격식체 사용 등
        
        # 반말을 존댓말로 변환 (간단한 예시)
        text = re.sub(r'([가-힣])해\b', '\\1합니다', text)
        text = re.sub(r'([가-힣])야\b', '\\1입니다', text)
        text = re.sub(r'([가-힣])다\b', '\\1습니다', text)
        
        return text
    
    def _apply_casual_style(self, text: str) -> str:
        """
        친근한 스타일 적용
        
        Args:
            text: 입력 텍스트
        
        Returns:
            친근한 스타일이 적용된 텍스트
        """
        # 친근한 스타일 적용 로직
        # 예: 반말 사용, 구어체 표현 등
        
        # 존댓말을 반말로 변환 (간단한 예시)
        text = re.sub(r'([가-힣])합니다\b', '\\1해', text)
        text = re.sub(r'([가-힣])입니다\b', '\\1야', text)
        text = re.sub(r'([가-힣])습니다\b', '\\1다', text)
        
        return text
    
    def _apply_professional_style(self, text: str) -> str:
        """
        전문가 스타일 적용
        
        Args:
            text: 입력 텍스트
        
        Returns:
            전문가 스타일이 적용된 텍스트
        """
        # 전문가 스타일 적용 로직
        # 예: 전문 용어 사용, 객관적 표현 등
        
        # 정중한 스타일 적용
        text = self._apply_formal_style(text)
        
        # 추가적인 전문가 스타일 적용 (예시)
        text = re.sub(r'생각합니다', '판단됩니다', text)
        text = re.sub(r'좋은', '효과적인', text)
        
        return text
    
    def get_conversation_context(self) -> List[Dict[str, str]]:
        """
        대화 컨텍스트 반환
        
        Returns:
            대화 컨텍스트
        """
        return self.conversation_history
    
    def clear_conversation_history(self) -> None:
        """대화 이력 초기화"""
        self.conversation_history = []
        logger.info("대화 이력 초기화 완료")

class ChatGPTLevelConversationOptimizer(ConversationOptimizer):
    """ChatGPT 수준의 대화 최적화 클래스"""
    
    def __init__(self, 
                 context_length: int = 10,
                 response_quality_enhancement: bool = True,
                 default_style: str = "casual",
                 personality: str = "helpful",
                 enable_context_awareness: bool = True):
        """
        ChatGPT 수준의 대화 최적화 초기화
        
        Args:
            context_length: 대화 컨텍스트 길이
            response_quality_enhancement: 응답 품질 향상 여부
            default_style: 기본 대화 스타일 ("formal", "casual", "professional")
            personality: 대화 성격 ("helpful", "creative", "concise")
            enable_context_awareness: 컨텍스트 인식 활성화 여부
        """
        super().__init__(
            context_length=context_length,
            response_quality_enhancement=response_quality_enhancement,
            default_style=default_style
        )
        
        self.personality = personality
        self.enable_context_awareness = enable_context_awareness
        
        # 대화 성격 매핑
        self.personality_mapping = {
            "helpful": self._apply_helpful_personality,
            "creative": self._apply_creative_personality,
            "concise": self._apply_concise_personality
        }
        
        # 추가 대화 스타일 매핑
        self.style_mapping.update({
            "academic": self._apply_academic_style,
            "poetic": self._apply_poetic_style,
            "humorous": self._apply_humorous_style
        })
        
        logger.info(f"ChatGPT 수준의 대화 최적화 초기화: 성격={personality}, 컨텍스트 인식={enable_context_awareness}")
    
    def process_conversation(self, user_input: str, model_response: str, 
                            style: Optional[str] = None,
                            personality: Optional[str] = None) -> str:
        """
        대화 처리 (확장)
        
        Args:
            user_input: 사용자 입력
            model_response: 모델 응답
            style: 대화 스타일 (None인 경우 기본 스타일 사용)
            personality: 대화 성격 (None인 경우 기본 성격 사용)
        
        Returns:
            처리된 응답
        """
        logger.info("ChatGPT 수준의 대화 처리 시작")
        
        try:
            # 대화 이력 업데이트
            self._update_conversation_history(user_input, model_response)
            
            # 컨텍스트 인식 적용 (설정에 따라)
            if self.enable_context_awareness:
                processed_response = self._apply_context_awareness(model_response)
            else:
                processed_response = model_response
            
            # 응답 품질 향상 (설정에 따라)
            if self.response_quality_enhancement:
                processed_response = self._enhance_response_quality(processed_response)
            
            # 대화 성격 적용
            current_personality = personality if personality else self.personality
            if current_personality in self.personality_mapping:
                processed_response = self.personality_mapping[current_personality](processed_response)
            
            # 대화 스타일 적용
            current_style = style if style else self.default_style
            if current_style in self.style_mapping:
                processed_response = self.style_mapping[current_style](processed_response)
            
            logger.info("ChatGPT 수준의 대화 처리 완료")
            return processed_response
            
        except Exception as e:
            logger.error(f"ChatGPT 수준의 대화 처리 중 오류 발생: {str(e)}", exc_info=True)
            return model_response
    
    def _apply_context_awareness(self, response: str) -> str:
        """
        컨텍스트 인식 적용
        
        Args:
            response: 원본 응답
        
        Returns:
            컨텍스트 인식이 적용된 응답
        """
        # 대화 이력이 없는 경우
        if not self.conversation_history:
            return response
        
        # 이전 대화 참조
        prev_conversations = self.conversation_history[:-1]
        if not prev_conversations:
            return response
        
        # 이전 대화에서 언급된 주요 키워드 추출
        keywords = self._extract_keywords_from_history(prev_conversations)
        
        # 컨텍스트 인식 적용 (예시)
        for keyword in keywords:
            if keyword not in response and keyword in self.conversation_history[-1]["user"]:
                # 응답에 키워드 추가
                response = f"{keyword}에 대해 말씀드리자면, {response}"
                break
        
        return response
    
    def _extract_keywords_from_history(self, history: List[Dict[str, str]]) -> List[str]:
        """
        대화 이력에서 키워드 추출
        
        Args:
            history: 대화 이력
        
        Returns:
            추출된 키워드 리스트
        """
        # 간단한 키워드 추출 로직 (예시)
        keywords = []
        
        for conversation in history:
            # 사용자 입력에서 명사 추출 (간단한 예시)
            user_input = conversation["user"]
            words = user_input.split()
            
            for word in words:
                # 2글자 이상의 단어만 키워드로 간주 (간단한 예시)
                if len(word) >= 2 and word not in keywords:
                    keywords.append(word)
        
        return keywords
    
    def _enhance_response_quality(self, response: str) -> str:
        """
        응답 품질 향상 (확장)
        
        Args:
            response: 원본 응답
        
        Returns:
            품질이 향상된 응답
        """
        # 기본 품질 향상 적용
        response = super()._enhance_response_quality(response)
        
        # 추가 품질 향상 적용
        response = self._improve_coherence(response)
        response = self._improve_naturalness(response)
        
        return response
    
    def _improve_coherence(self, text: str) -> str:
        """
        일관성 향상
        
        Args:
            text: 입력 텍스트
        
        Returns:
            일관성이 향상된 텍스트
        """
        # 일관성 향상 로직 (예시)
        # 문장 간 연결 개선, 논리적 흐름 강화 등
        
        return text
    
    def _improve_naturalness(self, text: str) -> str:
        """
        자연스러움 향상
        
        Args:
            text: 입력 텍스트
        
        Returns:
            자연스러움이 향상된 텍스트
        """
        # 자연스러움 향상 로직 (예시)
        # 구어체 표현 사용, 자연스러운 어휘 선택 등
        
        return text
    
    def _apply_helpful_personality(self, text: str) -> str:
        """
        도움이 되는 성격 적용
        
        Args:
            text: 입력 텍스트
        
        Returns:
            도움이 되는 성격이 적용된 텍스트
        """
        # 도움이 되는 성격 적용 로직 (예시)
        # 추가 정보 제공, 구체적인 설명 등
        
        return text
    
    def _apply_creative_personality(self, text: str) -> str:
        """
        창의적인 성격 적용
        
        Args:
            text: 입력 텍스트
        
        Returns:
            창의적인 성격이 적용된 텍스트
        """
        # 창의적인 성격 적용 로직 (예시)
        # 비유적 표현, 다양한 관점 제시 등
        
        return text
    
    def _apply_concise_personality(self, text: str) -> str:
        """
        간결한 성격 적용
        
        Args:
            text: 입력 텍스트
        
        Returns:
            간결한 성격이 적용된 텍스트
        """
        # 간결한 성격 적용 로직 (예시)
        # 불필요한 내용 제거, 핵심만 간략히 표현 등
        
        # 문장 수 제한 (예시)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 3:
            return '. '.join(sentences[:3]) + '.'
        
        return text
    
    def _apply_academic_style(self, text: str) -> str:
        """
        학술적 스타일 적용
        
        Args:
            text: 입력 텍스트
        
        Returns:
            학술적 스타일이 적용된 텍스트
        """
        # 학술적 스타일 적용 로직 (예시)
        # 객관적 표현, 인용 형식 사용 등
        
        return text
    
    def _apply_poetic_style(self, text: str) -> str:
        """
        시적 스타일 적용
        
        Args:
            text: 입력 텍스트
        
        Returns:
            시적 스타일이 적용된 텍스트
        """
        # 시적 스타일 적용 로직 (예시)
        # 운율 사용, 비유적 표현 등
        
        return text
    
    def _apply_humorous_style(self, text: str) -> str:
        """
        유머러스한 스타일 적용
        
        Args:
            text: 입력 텍스트
        
        Returns:
            유머러스한 스타일이 적용된 텍스트
        """
        # 유머러스한 스타일 적용 로직 (예시)
        # 재치 있는 표현, 농담 추가 등
        
        return text
