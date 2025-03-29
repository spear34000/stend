"""
Gemini 2.0 Flash API를 사용한 AI 에이전트의 유틸리티 함수
"""
import os
import json
from typing import Dict, List, Any, Optional

def save_conversation(conversation_history: List[Dict[str, str]], file_path: str):
    """
    대화 기록을 JSON 파일로 저장
    
    Args:
        conversation_history: 저장할 대화 기록
        file_path: 저장할 파일 경로
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_history, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"대화 저장 중 오류 발생: {str(e)}")
        return False

def load_conversation(file_path: str) -> Optional[List[Dict[str, str]]]:
    """
    JSON 파일에서 대화 기록 로드
    
    Args:
        file_path: 로드할 파일 경로
        
    Returns:
        로드된 대화 기록 또는 오류 시 None
    """
    try:
        if not os.path.exists(file_path):
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"대화 로드 중 오류 발생: {str(e)}")
        return None

def format_conversation_for_display(conversation_history: List[Dict[str, str]]) -> str:
    """
    대화 기록을 표시용으로 포맷팅
    
    Args:
        conversation_history: 포맷팅할 대화 기록
        
    Returns:
        포맷팅된 대화 텍스트
    """
    formatted_text = ""
    
    for message in conversation_history:
        role = message.get("role", "")
        content = message.get("content", "")
        
        # 시스템 메시지는 표시하지 않음
        if role == "system":
            continue
            
        # 역할에 따라 표시 이름 설정
        display_name = "사용자" if role == "user" else "AI"
        
        # 메시지 추가
        formatted_text += f"{display_name}: {content}\n\n"
    
    return formatted_text.strip()

def validate_api_key(api_key: str) -> bool:
    """
    API 키 형식 검증 (간단한 검증)
    
    Args:
        api_key: 검증할 API 키
        
    Returns:
        유효한 API 키인지 여부
    """
    # API 키가 비어있지 않고 일정 길이 이상인지 확인
    return bool(api_key and len(api_key) >= 10)
