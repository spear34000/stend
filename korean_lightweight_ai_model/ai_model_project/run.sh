#!/bin/bash

# 한국어 특화 초경량 AI 모델 실행 스크립트

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}한국어 특화 초경량 AI 모델 실행 스크립트${NC}"
echo -e "${BLUE}RAM 3GB, SSD 64GB 환경에서 ChatGPT 수준의 성능을 제공하는 한국어 특화 모델${NC}"
echo "--------------------------------------------------------------"

# 프로젝트 루트 디렉토리 설정
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 가상 환경 활성화
if [ -d "venv" ]; then
    echo -e "${YELLOW}가상 환경 활성화 중...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}가상 환경 생성 및 활성화 중...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    
    echo -e "${YELLOW}필수 패키지 설치 중...${NC}"
    pip install -q -r requirements.txt
fi

# 실행 모드 선택
echo "--------------------------------------------------------------"
echo "실행 모드를 선택하세요:"
echo "1. 대화 모드 (CLI)"
echo "2. 웹 인터페이스"
echo "3. API 서버"
echo "4. 벤치마크 실행"
echo "5. 종료"
echo "--------------------------------------------------------------"

read -p "선택 (1-5): " choice

case $choice in
    1)
        echo -e "${GREEN}대화 모드 시작...${NC}"
        python3 src/main.py
        ;;
    2)
        echo -e "${GREEN}웹 인터페이스 시작...${NC}"
        python3 src/web/app.py
        ;;
    3)
        echo -e "${GREEN}API 서버 시작...${NC}"
        python3 src/api/api.py
        ;;
    4)
        echo -e "${GREEN}벤치마크 실행...${NC}"
        python3 src/benchmarks/benchmarker.py --all
        ;;
    5)
        echo -e "${GREEN}종료합니다.${NC}"
        exit 0
        ;;
    *)
        echo -e "${YELLOW}잘못된 선택입니다. 종료합니다.${NC}"
        exit 1
        ;;
esac
