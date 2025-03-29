"""
Gemini 2.0 Flash API를 사용한 AI 에이전트 설정 파일
"""
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# API 키 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 모델 설정
MODEL_NAME = "gemini-2.0-flash"

# 기본 설정
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 40
