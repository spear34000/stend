"""
한국어 특화 초경량 AI 모델 - API 모듈

이 모듈은 한국어 특화 초경량 AI 모델의 API 인터페이스를 제공합니다.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Optional, Union, Any, Tuple
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 프로젝트 루트 디렉토리 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 모듈 임포트
from src.main import UltraLightKoreanAI
from src.utils.utils import ConfigManager, MemoryMonitor

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="한국어 특화 초경량 AI 모델 API",
    description="RAM 3GB, SSD 64GB 환경에서도 ChatGPT 수준의 성능을 제공하는 한국어 특화 초경량 AI 모델 API",
    version="1.0.0"
)

# 모델 인스턴스
model = None
memory_monitor = MemoryMonitor()

# 요청 모델 정의
class ChatRequest(BaseModel):
    message: str
    style: Optional[str] = "casual"
    personality: Optional[str] = "helpful"

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

# 응답 모델 정의
class MemoryUsage(BaseModel):
    process_rss_mb: float
    peak_memory_mb: float

class ChatResponse(BaseModel):
    response: str
    memory_usage: MemoryUsage

class GenerateResponse(BaseModel):
    generated_text: str
    memory_usage: MemoryUsage

class ModelSize(BaseModel):
    param_count: int
    total_size_mb: float

class ModelInfoResponse(BaseModel):
    model_name: str
    quantization_bits: int
    memory_usage: MemoryUsage
    model_size: ModelSize

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 이벤트"""
    global model, memory_monitor
    
    try:
        # 메모리 모니터링 시작
        memory_monitor.start_monitoring()
        
        # 모델 초기화
        logger.info("모델 초기화 중...")
        model = UltraLightKoreanAI()
        model.load()
        
        logger.info("모델 초기화 완료")
    except Exception as e:
        logger.error(f"모델 초기화 중 오류 발생: {str(e)}", exc_info=True)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    채팅 API 엔드포인트
    
    Args:
        request: 채팅 요청
    
    Returns:
        채팅 응답
    """
    global model, memory_monitor
    
    try:
        # 모델이 초기화되지 않은 경우 초기화
        if model is None:
            logger.info("모델 초기화 중...")
            model = UltraLightKoreanAI()
            model.load()
        
        # 응답 생성
        response = model.chat(
            user_input=request.message,
            style=request.style,
            personality=request.personality
        )
        
        # 메모리 사용량 통계
        memory_stats = memory_monitor.get_memory_usage_stats()
        
        # 응답 반환
        return ChatResponse(
            response=response,
            memory_usage=MemoryUsage(
                process_rss_mb=memory_stats["process_rss_mb"],
                peak_memory_mb=memory_stats["peak_memory_mb"]
            )
        )
        
    except Exception as e:
        logger.error(f"채팅 API 처리 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    텍스트 생성 API 엔드포인트
    
    Args:
        request: 텍스트 생성 요청
    
    Returns:
        텍스트 생성 응답
    """
    global model, memory_monitor
    
    try:
        # 모델이 초기화되지 않은 경우 초기화
        if model is None:
            logger.info("모델 초기화 중...")
            model = UltraLightKoreanAI()
            model.load()
        
        # 텍스트 생성
        generated_text = model.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # 메모리 사용량 통계
        memory_stats = memory_monitor.get_memory_usage_stats()
        
        # 응답 반환
        return GenerateResponse(
            generated_text=generated_text,
            memory_usage=MemoryUsage(
                process_rss_mb=memory_stats["process_rss_mb"],
                peak_memory_mb=memory_stats["peak_memory_mb"]
            )
        )
        
    except Exception as e:
        logger.error(f"텍스트 생성 API 처리 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/info", response_model=ModelInfoResponse)
async def model_info():
    """
    모델 정보 API 엔드포인트
    
    Returns:
        모델 정보 응답
    """
    global model, memory_monitor
    
    try:
        # 모델이 초기화되지 않은 경우 초기화
        if model is None:
            logger.info("모델 초기화 중...")
            model = UltraLightKoreanAI()
            model.load()
        
        # 모델 설정 가져오기
        model_config = model.model_config
        
        # 메모리 사용량 통계
        memory_stats = memory_monitor.get_memory_usage_stats()
        
        # 모델 크기 정보
        from src.utils.utils import TorchUtils
        model_size_info = TorchUtils.get_model_size(model.model)
        
        # 응답 반환
        return ModelInfoResponse(
            model_name=model_config.get("model_name", "polyglot-ko-410m"),
            quantization_bits=model_config.get("quantization_bits", 2),
            memory_usage=MemoryUsage(
                process_rss_mb=memory_stats["process_rss_mb"],
                peak_memory_mb=memory_stats["peak_memory_mb"]
            ),
            model_size=ModelSize(
                param_count=model_size_info["param_count"],
                total_size_mb=model_size_info["total_size_mb"]
            )
        )
        
    except Exception as e:
        logger.error(f"모델 정보 API 처리 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 처리기"""
    logger.error(f"전역 예외 발생: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="한국어 특화 초경량 AI 모델 API 서버")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="호스트 주소")
    parser.add_argument("--port", type=int, default=8000, help="포트 번호")
    args = parser.parse_args()
    
    # API 서버 실행
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
