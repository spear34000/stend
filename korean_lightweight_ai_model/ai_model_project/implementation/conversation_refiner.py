"""
한국어 특화 경량화 AI 모델 대화 품질 개선 모듈
- 테스트 결과 기반 대화 품질 개선
- 한국어 자연스러움 향상
- 대화 일관성 강화
"""

import os
import re
import json
import logging
from typing import List, Dict, Tuple, Optional, Union

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationRefiner:
    """대화 품질 개선 클래스"""
    
    def __init__(self, conversation_optimizer):
        """
        대화 품질 개선 초기화
        
        Args:
            conversation_optimizer: 대화 최적화 객체
        """
        self.conversation_optimizer = conversation_optimizer
        
        # 한국어 자연스러움 향상을 위한 패턴
        self.improvement_patterns = {
            # 반복 표현 제거
            r'(\w+)\s+\1': r'\1',
            
            # 영어 따옴표를 한글 따옴표로 변환
            r'"([^"]*)"': r'"\1"',
            r"'([^']*)'": r''\1'',
            
            # 한국어 조사 교정
            r'이(가|을|를|에|의|로|와|과|나|이나|든지) ': r'이\1 ',
            r'([가-힣])(을|를) ([가-힣])': r'\1\2 \3',
            
            # 존댓말 일관성
            r'([가-힣]+)해요\s+([가-힣]+)한다': r'\1해요 \2해요',
            r'([가-힣]+)합니다\s+([가-힣]+)해요': r'\1합니다 \2합니다',
            
            # 불필요한 공백 제거
            r'\s+([.,!?])': r'\1',
            r'([.,!?])\s+([가-힣])': r'\1 \2'
        }
        
        # 한국어 자연스러운 표현 사전
        self.natural_expressions = {
            "안녕하세요": ["안녕하세요", "반갑습니다", "만나서 반갑습니다"],
            "감사합니다": ["감사합니다", "고맙습니다", "감사의 말씀을 드립니다"],
            "죄송합니다": ["죄송합니다", "사과드립니다", "양해 부탁드립니다"],
            "알겠습니다": ["알겠습니다", "이해했습니다", "네, 그렇게 하겠습니다"]
        }
        
        logger.info("대화 품질 개선 초기화 완료")
    
    def analyze_test_results(self, test_results_path):
        """
        테스트 결과 분석
        
        Args:
            test_results_path: 테스트 결과 파일 경로
        
        Returns:
            분석 결과 딕셔너리
        """
        logger.info(f"테스트 결과 분석 시작: {test_results_path}")
        
        if not os.path.exists(test_results_path):
            logger.error(f"테스트 결과 파일이 존재하지 않습니다: {test_results_path}")
            return {}
        
        try:
            with open(test_results_path, 'r', encoding='utf-8') as f:
                test_results = json.load(f)
            
            analysis = {
                "general_conversation": self._analyze_general_conversation(test_results.get("general_conversation", [])),
                "context_maintenance": self._analyze_context_maintenance(test_results.get("context_maintenance", [])),
                "style_adaptation": self._analyze_style_adaptation(test_results.get("style_adaptation", [])),
                "korean_language_quality": self._analyze_korean_quality(test_results.get("korean_language_quality", []))
            }
            
            logger.info("테스트 결과 분석 완료")
            return analysis
            
        except Exception as e:
            logger.error(f"테스트 결과 분석 중 오류 발생: {str(e)}", exc_info=True)
            return {}
    
    def _analyze_general_conversation(self, results):
        """일반 대화 테스트 결과 분석"""
        if not results:
            return {}
        
        avg_generation_time = sum(r.get("generation_time", 0) for r in results) / len(results)
        
        # 응답 길이 분석
        response_lengths = [len(r.get("response", "")) for r in results]
        avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        
        return {
            "avg_generation_time": avg_generation_time,
            "avg_response_length": avg_response_length,
            "improvement_areas": self._identify_improvement_areas(results)
        }
    
    def _analyze_context_maintenance(self, results):
        """대화 맥락 유지 테스트 결과 분석"""
        if not results:
            return {}
        
        context_scores = []
        
        for flow in results:
            exchanges = flow.get("exchanges", [])
            if len(exchanges) < 2:
                continue
            
            # 맥락 유지 점수 계산 (간단한 휴리스틱)
            context_score = 0
            for i in range(1, len(exchanges)):
                prev_exchange = exchanges[i-1]
                curr_exchange = exchanges[i]
                
                # 이전 응답에 언급된 키워드가 현재 응답에도 포함되는지 확인
                prev_keywords = self._extract_keywords(prev_exchange.get("response", ""))
                curr_response = curr_exchange.get("response", "")
                
                keyword_matches = sum(1 for kw in prev_keywords if kw in curr_response)
                context_score += keyword_matches / len(prev_keywords) if prev_keywords else 0
            
            context_score = context_score / (len(exchanges) - 1) if len(exchanges) > 1 else 0
            context_scores.append(context_score)
        
        avg_context_score = sum(context_scores) / len(context_scores) if context_scores else 0
        
        return {
            "avg_context_score": avg_context_score,
            "improvement_areas": self._identify_context_issues(results)
        }
    
    def _analyze_style_adaptation(self, results):
        """대화 스타일 적응 테스트 결과 분석"""
        if not results:
            return {}
        
        style_consistency = {}
        
        for result in results:
            style = result.get("style", "")
            response = result.get("response", "")
            
            if style == "formal":
                # 존댓말 비율 계산
                formal_markers = len(re.findall(r'습니다|입니다|니다|세요', response))
                informal_markers = len(re.findall(r'해요|네요|어요|군요', response))
                total_markers = formal_markers + informal_markers
                
                formal_ratio = formal_markers / total_markers if total_markers > 0 else 0
                style_consistency[style] = formal_ratio
                
            elif style == "casual":
                # 반말 비율 계산
                casual_markers = len(re.findall(r'해요|네요|어요|군요', response))
                formal_markers = len(re.findall(r'습니다|입니다|니다|세요', response))
                total_markers = casual_markers + formal_markers
                
                casual_ratio = casual_markers / total_markers if total_markers > 0 else 0
                style_consistency[style] = casual_ratio
        
        return {
            "style_consistency": style_consistency,
            "improvement_areas": self._identify_style_issues(results)
        }
    
    def _analyze_korean_quality(self, results):
        """한국어 언어 품질 테스트 결과 분석"""
        if not results:
            return {}
        
        avg_korean_char_ratio = sum(r.get("korean_char_ratio", 0) for r in results) / len(results)
        avg_jamo_separation_ratio = sum(r.get("jamo_separation_ratio", 0) for r in results) / len(results)
        
        # 문법 오류 분석
        grammar_issues = []
        for result in results:
            response = result.get("response", "")
            issues = self._identify_grammar_issues(response)
            if issues:
                grammar_issues.append({
                    "response": response,
                    "issues": issues
                })
        
        return {
            "avg_korean_char_ratio": avg_korean_char_ratio,
            "avg_jamo_separation_ratio": avg_jamo_separation_ratio,
            "grammar_issues": grammar_issues
        }
    
    def _extract_keywords(self, text):
        """텍스트에서 키워드 추출"""
        # 간단한 키워드 추출 (명사 위주)
        words = re.findall(r'[가-힣]{2,}', text)
        return [w for w in words if len(w) >= 2]
    
    def _identify_improvement_areas(self, results):
        """개선 영역 식별"""
        improvement_areas = []
        
        for result in results:
            response = result.get("response", "")
            
            # 반복 문장 확인
            sentences = re.split(r'(?<=[.!?])\s+', response)
            unique_sentences = set(sentences)
            if len(sentences) - len(unique_sentences) > 0:
                improvement_areas.append("반복 문장 제거")
            
            # 불완전한 문장 확인
            if not response.strip().endswith(('.', '!', '?', '다', '요', '죠', '죠.')):
                improvement_areas.append("불완전한 문장 개선")
            
            # 일관성 없는 존댓말/반말 확인
            formal_markers = len(re.findall(r'습니다|입니다|니다|세요', response))
            informal_markers = len(re.findall(r'해요|네요|어요|군요', response))
            if formal_markers > 0 and informal_markers > 0:
                improvement_areas.append("존댓말/반말 일관성 개선")
        
        return list(set(improvement_areas))
    
    def _identify_context_issues(self, results):
        """맥락 유지 문제 식별"""
        context_issues = []
        
        for flow in results:
            exchanges = flow.get("exchanges", [])
            if len(exchanges) < 2:
                continue
            
            for i in range(1, len(exchanges)):
                prev_prompt = exchanges[i-1].get("prompt", "")
                curr_response = exchanges[i].get("response", "")
                
                # 이전 질문에 대한 참조가 없는 경우
                if not any(kw in curr_response for kw in self._extract_keywords(prev_prompt)):
                    context_issues.append("이전 대화 참조 부족")
                    break
        
        return list(set(context_issues))
    
    def _identify_style_issues(self, results):
        """스타일 적응 문제 식별"""
        style_issues = []
        
        for result in results:
            style = result.get("style", "")
            response = result.get("response", "")
            
            if style == "formal":
                # 존댓말이 아닌 표현 확인
                if re.search(r'해\b|네\b|어\b|군\b', response):
                    style_issues.append("정중한 스타일에 반말 표현 포함")
                    
            elif style == "casual":
                # 너무 격식있는 표현 확인
                if re.search(r'드리겠습니다|되겠습니다|하겠습니다', response):
                    style_issues.append("친근한 스타일에 과도한 격식 표현 포함")
        
        return list(set(style_issues))
    
    def _identify_grammar_issues(self, text):
        """문법 오류 식별"""
        issues = []
        
        # 조사 오류 확인
        if re.search(r'([^이])(가|을|를) ', text) or re.search(r'이(가|을|를)([^가-힣\s])', text):
            issues.append("조사 사용 오류")
        
        # 어미 오류 확인
        if re.search(r'([^하])ㅂ니다', text) or re.search(r'습니다([^가-힣\s\.])', text):
            issues.append("어미 사용 오류")
        
        # 자모 분리 확인
        if re.search(r'[ㄱ-ㅎㅏ-ㅣ]+', text):
            issues.append("자모 분리 발생")
        
        return issues
    
    def apply_improvements(self):
        """
        개선 사항 적용
        
        Returns:
            개선된 대화 최적화 객체
        """
        logger.info("대화 품질 개선 적용 시작")
        
        # 한국어 자연스러움 향상을 위한 패턴 적용
        original_enhance_fn = self.conversation_optimizer.enhance_korean_response
        
        def enhanced_response_function(response):
            # 기존 개선 함수 호출
            improved_response = original_enhance_fn(response)
            
            # 추가 개선 패턴 적용
            for pattern, replacement in self.improvement_patterns.items():
                improved_response = re.sub(pattern, replacement, improved_response)
            
            # 자연스러운 표현으로 대체
            for formal, alternatives in self.natural_expressions.items():
                if formal in improved_response:
                    # 무작위성 추가를 위해 항상 첫 번째 대안을 사용하지 않고
                    # 문맥에 따라 다른 표현 선택 (여기서는 간단히 문장 길이로 결정)
                    index = len(improved_response) % len(alternatives)
                    improved_response = improved_response.replace(formal, alternatives[index])
            
            return improved_response
        
        # 개선된 함수로 교체
        self.conversation_optimizer.enhance_korean_response = enhanced_response_function
        
        # 대화 맥락 유지 개선
        self.conversation_optimizer.max_history_tokens = 768  # 맥락 이력 확장
        
        logger.info("대화 품질 개선 적용 완료")
        return self.conversation_optimizer
    
    def generate_improvement_report(self, analysis):
        """
        개선 보고서 생성
        
        Args:
            analysis: 분석 결과 딕셔너리
        
        Returns:
            개선 보고서 문자열
        """
        report = "# 한국어 특화 경량화 AI 모델 대화 품질 개선 보고서\n\n"
        
        # 일반 대화 개선 사항
        report += "## 1. 일반 대화 개선 사항\n\n"
        general = analysis.get("general_conversation", {})
        report += f"- 평균 응답 생성 시간: {general.get('avg_generation_time', 0):.2f}초\n"
        report += f"- 평균 응답 길이: {general.get('avg_response_length', 0):.1f}자\n"
        
        improvement_areas = general.get("improvement_areas", [])
        if improvement_areas:
            report += "- 개선 영역:\n"
            for area in improvement_areas:
                report += f"  - {area}\n"
        
        # 맥락 유지 개선 사항
        report += "\n## 2. 대화 맥락 유지 개선 사항\n\n"
        context = analysis.get("context_maintenance", {})
        report += f"- 맥락 유지 점수: {context.get('avg_context_score', 0):.2f}/1.0\n"
        
        context_issues = context.get("improvement_areas", [])
        if context_issues:
            report += "- 개선 영역:\n"
            for issue in context_issues:
                report += f"  - {issue}\n"
        
        # 스타일 적응 개선 사항
        report += "\n## 3. 대화 스타일 적응 개선 사항\n\n"
        style = analysis.get("style_adaptation", {})
        style_consistency = style.get("style_consistency", {})
        
        for style_name, consistency in style_consistency.items():
            report += f"- {style_name} 스타일 일관성: {consistency:.2f}/1.0\n"
        
        style_issues = style.get("improvement_areas", [])
        if style_issues:
            report += "- 개선 영역:\n"
            for issue in style_issues:
                report += f"  - {issue}\n"
        
        # 한국어 품질 개선 사항
        report += "\n## 4. 한국어 언어 품질 개선 사항\n\n"
        korean = analysis.get("korean_language_quality", {})
        report += f"- 한글 문자 비율: {korean.get('avg_korean_char_ratio', 0):.2f}\n"
        report += f"- 자모 분리 비율: {korean.get('avg_jamo_separation_ratio', 0):.2f}\n"
        
        grammar_issues = korean.get("grammar_issues", [])
        if grammar_issues:
            report += "- 문법 오류 사례:\n"
            for i, issue in enumerate(grammar_issues[:3]):  # 처음 3개만 표시
                report += f"  - 사례 {i+1}: {issue.get('issues', [])}\n"
        
        # 적용된 개선 사항
        report += "\n## 5. 적용된 개선 사항\n\n"
        report += "- 한국어 자연스러움 향상을 위한 패턴 적용\n"
        report += "- 대화 맥락 유지를 위한 이력 토큰 확장 (512 → 768)\n"
        report += "- 자연스러운 한국어 표현 다양화\n"
        report += "- 존댓말/반말 일관성 개선\n"
        report += "- 조사 및 어미 사용 교정\n"
        
        return report
