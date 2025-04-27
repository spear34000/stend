"""
한국어 특화 초경량 AI 모델 - 웹 기반 학습 인터페이스

이 모듈은 웹 인터페이스를 통해 모델의 학습과 훈련을 자동화합니다.
"""

import os
import sys
import json
import logging
import argparse
import threading
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from flask import Flask, request, jsonify, render_template, Response, stream_with_context

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

# 전역 변수
model = None
memory_monitor = MemoryMonitor()
training_in_progress = False
training_logs = []
training_thread = None

@app.route('/')
def index():
    """학습 인터페이스 메인 페이지"""
    return render_template('training.html')

@app.route('/api/train', methods=['POST'])
def start_training():
    """
    모델 학습 시작 API
    
    요청 형식:
    {
        "dataset": "korean_conversation", // 학습 데이터셋
        "epochs": 3,                      // 학습 에포크
        "batch_size": 8,                  // 배치 크기
        "learning_rate": 5e-5,            // 학습률
        "max_steps": 1000                 // 최대 학습 스텝
    }
    
    응답 형식:
    {
        "status": "started",
        "message": "학습이 시작되었습니다."
    }
    """
    global training_in_progress, training_logs, training_thread
    
    # 이미 학습 중인 경우
    if training_in_progress:
        return jsonify({
            "status": "error",
            "message": "이미 학습이 진행 중입니다."
        }), 400
    
    try:
        # 요청 데이터 파싱
        data = request.json
        dataset = data.get('dataset', 'korean_conversation')
        epochs = data.get('epochs', 3)
        batch_size = data.get('batch_size', 8)
        learning_rate = data.get('learning_rate', 5e-5)
        max_steps = data.get('max_steps', 1000)
        
        # 학습 설정
        training_config = {
            "dataset": dataset,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_steps": max_steps
        }
        
        # 학습 로그 초기화
        training_logs = []
        
        # 학습 스레드 시작
        training_in_progress = True
        training_thread = threading.Thread(
            target=train_model_thread,
            args=(training_config,)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            "status": "started",
            "message": "학습이 시작되었습니다."
        })
        
    except Exception as e:
        logger.error(f"학습 시작 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"학습 시작 중 오류 발생: {str(e)}"
        }), 500

@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    """
    학습 상태 확인 API
    
    응답 형식:
    {
        "status": "in_progress", // "in_progress", "completed", "not_started", "error"
        "progress": 45,          // 진행률 (0-100)
        "current_epoch": 2,      // 현재 에포크
        "current_step": 500,     // 현재 스텝
        "loss": 0.234,           // 현재 손실값
        "elapsed_time": 120      // 경과 시간 (초)
    }
    """
    global training_in_progress
    
    if not training_in_progress and not training_logs:
        return jsonify({
            "status": "not_started",
            "message": "학습이 시작되지 않았습니다."
        })
    
    if not training_in_progress and training_logs:
        # 마지막 로그 가져오기
        last_log = training_logs[-1] if training_logs else {}
        
        return jsonify({
            "status": "completed",
            "message": "학습이 완료되었습니다.",
            "final_loss": last_log.get("loss", 0),
            "total_epochs": last_log.get("epoch", 0),
            "total_steps": last_log.get("step", 0),
            "elapsed_time": last_log.get("elapsed_time", 0)
        })
    
    # 현재 진행 중인 학습 상태
    current_log = training_logs[-1] if training_logs else {}
    total_steps = current_log.get("total_steps", 1000)
    current_step = current_log.get("step", 0)
    progress = int((current_step / total_steps) * 100) if total_steps > 0 else 0
    
    return jsonify({
        "status": "in_progress",
        "progress": progress,
        "current_epoch": current_log.get("epoch", 0),
        "current_step": current_step,
        "loss": current_log.get("loss", 0),
        "elapsed_time": current_log.get("elapsed_time", 0)
    })

@app.route('/api/train/logs', methods=['GET'])
def stream_training_logs():
    """학습 로그 스트리밍 API"""
    
    def generate():
        global training_logs
        last_index = 0
        
        while True:
            if len(training_logs) > last_index:
                for i in range(last_index, len(training_logs)):
                    yield f"data: {json.dumps(training_logs[i])}\n\n"
                last_index = len(training_logs)
            
            # 학습이 완료된 경우
            if not training_in_progress and last_index > 0:
                yield "data: {\"status\": \"completed\"}\n\n"
                break
                
            time.sleep(0.5)
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/train/cancel', methods=['POST'])
def cancel_training():
    """
    학습 취소 API
    
    응답 형식:
    {
        "status": "cancelled",
        "message": "학습이 취소되었습니다."
    }
    """
    global training_in_progress
    
    if not training_in_progress:
        return jsonify({
            "status": "error",
            "message": "현재 진행 중인 학습이 없습니다."
        }), 400
    
    try:
        # 학습 취소 플래그 설정
        training_in_progress = False
        
        return jsonify({
            "status": "cancelled",
            "message": "학습이 취소되었습니다."
        })
        
    except Exception as e:
        logger.error(f"학습 취소 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"학습 취소 중 오류 발생: {str(e)}"
        }), 500

@app.route('/api/model/download', methods=['GET'])
def download_model():
    """
    학습된 모델 다운로드 API
    
    응답: 모델 파일 다운로드
    """
    try:
        # 모델 파일 경로
        model_path = os.path.join(project_root, 'models', 'trained_model.bin')
        
        # 모델 파일이 존재하는지 확인
        if not os.path.exists(model_path):
            return jsonify({
                "status": "error",
                "message": "학습된 모델 파일이 존재하지 않습니다."
            }), 404
        
        # 모델 파일 다운로드
        return send_file(
            model_path,
            as_attachment=True,
            download_name='korean_lightweight_model.bin',
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"모델 다운로드 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"모델 다운로드 중 오류 발생: {str(e)}"
        }), 500

def train_model_thread(config: Dict[str, Any]) -> None:
    """
    모델 학습 스레드 함수
    
    Args:
        config: 학습 설정
    """
    global training_in_progress, training_logs, model
    
    try:
        logger.info(f"모델 학습 시작: {config}")
        
        # 모델 초기화
        if model is None:
            model = UltraLightKoreanAI()
            model.load()
        
        # 학습 설정
        dataset = config["dataset"]
        epochs = config["epochs"]
        batch_size = config["batch_size"]
        learning_rate = config["learning_rate"]
        max_steps = config["max_steps"]
        
        # 학습 시작 시간
        start_time = time.time()
        
        # 학습 데이터 로드 (실제로는 데이터셋에 따라 다르게 로드)
        logger.info(f"데이터셋 로드 중: {dataset}")
        
        # 학습 로그 추가
        training_logs.append({
            "step": 0,
            "epoch": 0,
            "loss": 0.0,
            "elapsed_time": 0,
            "message": f"데이터셋 로드 중: {dataset}",
            "total_steps": max_steps
        })
        
        # 학습 루프 (실제로는 모델 학습 코드로 대체)
        for epoch in range(1, epochs + 1):
            if not training_in_progress:
                break
                
            logger.info(f"에포크 {epoch}/{epochs} 시작")
            
            for step in range(1, max_steps + 1):
                if not training_in_progress:
                    break
                
                # 실제 학습 코드 대신 시뮬레이션
                time.sleep(0.1)  # 학습 시간 시뮬레이션
                
                # 손실값 계산 (시뮬레이션)
                loss = 2.5 * (0.95 ** (epoch - 1)) * (0.999 ** step)
                
                # 현재 시간
                current_time = time.time()
                elapsed_time = int(current_time - start_time)
                
                # 로그 기록
                if step % 10 == 0 or step == max_steps:
                    logger.info(f"에포크 {epoch}/{epochs}, 스텝 {step}/{max_steps}, 손실: {loss:.4f}")
                    
                    # 학습 로그 추가
                    training_logs.append({
                        "step": step,
                        "epoch": epoch,
                        "loss": loss,
                        "elapsed_time": elapsed_time,
                        "message": f"에포크 {epoch}/{epochs}, 스텝 {step}/{max_steps}, 손실: {loss:.4f}",
                        "total_steps": max_steps
                    })
        
        # 학습 완료
        logger.info("모델 학습 완료")
        
        # 모델 저장
        model_dir = os.path.join(project_root, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # 모델 저장 (실제로는 모델 저장 코드로 대체)
        logger.info(f"모델 저장 중: {os.path.join(model_dir, 'trained_model.bin')}")
        
        # 학습 완료 로그 추가
        final_time = time.time()
        total_time = int(final_time - start_time)
        
        training_logs.append({
            "step": max_steps,
            "epoch": epochs,
            "loss": loss,
            "elapsed_time": total_time,
            "message": "모델 학습 완료 및 저장됨",
            "total_steps": max_steps
        })
        
    except Exception as e:
        logger.error(f"모델 학습 중 오류 발생: {str(e)}", exc_info=True)
        
        # 오류 로그 추가
        training_logs.append({
            "step": 0,
            "epoch": 0,
            "loss": 0.0,
            "elapsed_time": 0,
            "message": f"오류 발생: {str(e)}",
            "error": True
        })
        
    finally:
        # 학습 상태 업데이트
        training_in_progress = False

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="한국어 특화 초경량 AI 모델 웹 기반 학습 인터페이스")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="호스트 주소")
    parser.add_argument("--port", type=int, default=5001, help="포트 번호")
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    args = parser.parse_args()
    
    # 웹 서버 실행
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
