#!/bin/bash

# 한국어 특화 경량화 AI 모델 대화 테스트 실행 스크립트
# 자연스러운 한국어 대화 기능 테스트

echo "한국어 특화 경량화 AI 모델 대화 테스트 시작"

# 작업 디렉토리 설정
WORK_DIR="/home/ubuntu/ai_model_project"
IMPL_DIR="$WORK_DIR/implementation"
TEST_DIR="$WORK_DIR/conversation_test_results"

# 테스트 결과 디렉토리 생성
mkdir -p $TEST_DIR

# 가상 환경 활성화
cd $IMPL_DIR
source venv/bin/activate

echo "가상 환경 활성화 완료"

# 메모리 사용량 모니터링 시작
echo "현재 메모리 사용량: $(free -m | grep Mem | awk '{print $3}') MB"

# 대화 테스트 실행
echo "대화 테스트 실행 중..."
python conversation_tester.py \
  --model_name "EleutherAI/polyglot-ko-1.3b" \
  --use_4bit \
  --output_dir "$TEST_DIR"

# 테스트 결과 확인
echo "테스트 완료. 결과 확인 중..."
if [ -f "$TEST_DIR/conversation_examples.txt" ]; then
  echo "대화 테스트 결과 파일이 생성되었습니다."
  echo "대화 예시 샘플:"
  head -n 20 "$TEST_DIR/conversation_examples.txt"
else
  echo "대화 테스트 결과 파일이 생성되지 않았습니다."
fi

echo "메모리 사용량: $(free -m | grep Mem | awk '{print $3}') MB"

# 대화 인터페이스 실행 (선택적)
read -p "대화 인터페이스를 실행하시겠습니까? (y/n): " run_interface
if [ "$run_interface" = "y" ]; then
  echo "대화 인터페이스 실행 중..."
  python conversation_interface.py \
    --model_name "EleutherAI/polyglot-ko-1.3b" \
    --use_4bit \
    --max_memory_gb 2.5
fi

# 가상 환경 비활성화
deactivate

echo "한국어 특화 경량화 AI 모델 대화 테스트 완료"
