#!/bin/bash

# 한국어 특화 경량화 AI 모델 테스트 실행 스크립트
# 3GB RAM 환경에서 모델 성능 평가

echo "한국어 특화 경량화 AI 모델 테스트 시작"

# 작업 디렉토리 설정
WORK_DIR="/home/ubuntu/ai_model_project"
IMPL_DIR="$WORK_DIR/implementation"
EVAL_DIR="$WORK_DIR/evaluation_results"

# 평가 결과 디렉토리 생성
mkdir -p $EVAL_DIR

# 가상 환경 활성화
cd $IMPL_DIR
source venv/bin/activate

echo "가상 환경 활성화 완료"

# 메모리 제한 설정 (3GB 환경 시뮬레이션)
# 실제 환경에서는 주석 처리
# ulimit -v 3145728  # 3GB in KB

echo "메모리 사용량 모니터링 시작"
echo "현재 메모리 사용량: $(free -m | grep Mem | awk '{print $3}') MB"

# 모델 평가 실행
echo "모델 평가 실행 중..."
python model_evaluator.py \
  --model_name "EleutherAI/polyglot-ko-1.3b" \
  --use_4bit \
  --use_sharding \
  --max_memory_gb 2.5 \
  --output_dir "$EVAL_DIR"

# 평가 결과 확인
echo "평가 완료. 결과 확인 중..."
if [ -f "$EVAL_DIR/evaluation_results.json" ]; then
  echo "평가 결과 파일이 생성되었습니다."
  echo "결과 요약:"
  grep -A 10 "average" "$EVAL_DIR/evaluation_results.json" | head -10
else
  echo "평가 결과 파일이 생성되지 않았습니다."
fi

echo "메모리 사용량: $(free -m | grep Mem | awk '{print $3}') MB"

# 가상 환경 비활성화
deactivate

echo "한국어 특화 경량화 AI 모델 테스트 완료"
