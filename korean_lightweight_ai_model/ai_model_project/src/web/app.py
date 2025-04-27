"""
한국어 특화 초경량 AI 모델 - 웹 인터페이스 모듈

이 모듈은 한국어 특화 초경량 AI 모델의 웹 인터페이스를 제공합니다.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Optional, Union, Any, Tuple
from flask import Flask, request, jsonify, render_template

# 프로젝트 루트 디렉토리 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 모듈 임포트
from src.main import UltraLightKoreanAI
from src.utils.utils import ConfigManager, MemoryMonitor

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask 앱 초기화
app = Flask(__name__, 
            static_folder=os.path.join(project_root, 'src', 'web', 'static'),
            template_folder=os.path.join(project_root, 'src', 'web', 'templates'))

# 모델 인스턴스
model = None
memory_monitor = MemoryMonitor()

@app.route('/')
def index():
    """인덱스 페이지"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    채팅 API 엔드포인트
    
    요청 형식:
    {
        "message": "사용자 메시지",
        "style": "casual", // optional
        "personality": "helpful" // optional
    }
    
    응답 형식:
    {
        "response": "AI 응답",
        "memory_usage": {
            "process_rss_mb": 123.45,
            "peak_memory_mb": 234.56
        }
    }
    """
    global model, memory_monitor
    
    try:
        # 요청 데이터 파싱
        data = request.json
        user_message = data.get('message', '')
        style = data.get('style', 'casual')
        personality = data.get('personality', 'helpful')
        
        if not user_message:
            return jsonify({"error": "메시지가 비어 있습니다."}), 400
        
        # 메모리 모니터링 시작
        memory_monitor.start_monitoring()
        
        # 모델이 초기화되지 않은 경우 초기화
        if model is None:
            logger.info("모델 초기화 중...")
            model = UltraLightKoreanAI()
            model.load()
        
        # 응답 생성
        response = model.chat(
            user_input=user_message,
            style=style,
            personality=personality
        )
        
        # 메모리 사용량 통계
        memory_stats = memory_monitor.get_memory_usage_stats()
        
        # 응답 반환
        return jsonify({
            "response": response,
            "memory_usage": {
                "process_rss_mb": memory_stats["process_rss_mb"],
                "peak_memory_mb": memory_stats["peak_memory_mb"]
            }
        })
        
    except Exception as e:
        logger.error(f"채팅 API 처리 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    """
    텍스트 생성 API 엔드포인트
    
    요청 형식:
    {
        "prompt": "생성 프롬프트",
        "max_new_tokens": 512, // optional
        "temperature": 0.7, // optional
        "top_p": 0.9 // optional
    }
    
    응답 형식:
    {
        "generated_text": "생성된 텍스트",
        "memory_usage": {
            "process_rss_mb": 123.45,
            "peak_memory_mb": 234.56
        }
    }
    """
    global model, memory_monitor
    
    try:
        # 요청 데이터 파싱
        data = request.json
        prompt = data.get('prompt', '')
        max_new_tokens = data.get('max_new_tokens', 512)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)
        
        if not prompt:
            return jsonify({"error": "프롬프트가 비어 있습니다."}), 400
        
        # 메모리 모니터링 시작
        memory_monitor.start_monitoring()
        
        # 모델이 초기화되지 않은 경우 초기화
        if model is None:
            logger.info("모델 초기화 중...")
            model = UltraLightKoreanAI()
            model.load()
        
        # 텍스트 생성
        generated_text = model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # 메모리 사용량 통계
        memory_stats = memory_monitor.get_memory_usage_stats()
        
        # 응답 반환
        return jsonify({
            "generated_text": generated_text,
            "memory_usage": {
                "process_rss_mb": memory_stats["process_rss_mb"],
                "peak_memory_mb": memory_stats["peak_memory_mb"]
            }
        })
        
    except Exception as e:
        logger.error(f"텍스트 생성 API 처리 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """
    모델 정보 API 엔드포인트
    
    응답 형식:
    {
        "model_name": "polyglot-ko-410m",
        "quantization_bits": 2,
        "memory_usage": {
            "process_rss_mb": 123.45,
            "peak_memory_mb": 234.56
        },
        "model_size": {
            "param_count": 410000000,
            "total_size_mb": 123.45
        }
    }
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
        return jsonify({
            "model_name": model_config.get("model_name", "polyglot-ko-410m"),
            "quantization_bits": model_config.get("quantization_bits", 2),
            "memory_usage": {
                "process_rss_mb": memory_stats["process_rss_mb"],
                "peak_memory_mb": memory_stats["peak_memory_mb"]
            },
            "model_size": {
                "param_count": model_size_info["param_count"],
                "total_size_mb": model_size_info["total_size_mb"]
            }
        })
        
    except Exception as e:
        logger.error(f"모델 정보 API 처리 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="한국어 특화 초경량 AI 모델 웹 인터페이스")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="호스트 주소")
    parser.add_argument("--port", type=int, default=5000, help="포트 번호")
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    args = parser.parse_args()
    
    # 웹 서버 실행
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
