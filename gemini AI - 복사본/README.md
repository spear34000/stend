# Gemini 2.0 Flash API를 사용한 AI 에이전트

이 프로젝트는 Google의 Gemini 2.0 Flash API를 사용하여 사용자가 말한 대로 만들어주는 AI 에이전트를 구현한 것입니다. 이 에이전트는 사용자의 요청을 이해하고 그에 맞는 응답을 생성합니다.

## 주요 기능

- Gemini 2.0 Flash API를 사용한 텍스트 응답 생성
- 대화 컨텍스트 관리 및 기억
- 사용자 입력 처리 및 명령어 인식
- 오류 처리 및 재시도 메커니즘
- 응답 캐싱을 통한 성능 최적화

## 프로젝트 구조

```
ai_agent/
├── src/                      # 소스 코드 디렉토리
│   ├── __init__.py           # 패키지 초기화 파일
│   ├── agent.py              # 기본 에이전트 클래스
│   ├── api_client.py         # Gemini API 클라이언트
│   ├── config.py             # 설정 파일
│   ├── conversation.py       # 대화 관리 클래스
│   ├── error_handler.py      # 오류 처리 클래스
│   ├── input_processor.py    # 입력 처리 클래스
│   ├── request_handler.py    # 요청 처리 클래스
│   └── response_generator.py # 응답 생성 클래스
├── tests/                    # 테스트 디렉토리
│   ├── test_agent.py         # 기본 기능 테스트
│   └── test_error_scenarios.py # 오류 시나리오 테스트
├── .env                      # 환경 변수 파일 (API 키 포함)
├── .env.example              # 환경 변수 예제 파일
├── API_KEY_SETUP.md          # API 키 설정 안내
├── main.py                   # 메인 실행 파일
├── README.md                 # 프로젝트 설명
└── todo.md                   # 개발 계획 및 진행 상황
```

## 설치 방법

1. 저장소 클론:
   ```
   git clone <repository-url>
   cd ai_agent
   ```

2. 필요한 패키지 설치:
   ```
   pip install -q -U google-genai python-dotenv
   ```

3. API 키 설정:
   - `.env.example` 파일을 `.env`로 복사
   - `.env` 파일에 Gemini API 키 입력 (API 키 획득 방법은 `API_KEY_SETUP.md` 참조)

## 사용 방법

### 기본 실행

```
python main.py
```

### 명령행 옵션

```
python main.py --api_key YOUR_API_KEY --temperature 0.7 --max_tokens 2048
```

- `--api_key`: Gemini API 키 (설정되지 않은 경우 .env 파일에서 로드)
- `--temperature`: 응답 생성 온도 (0.0 ~ 1.0, 기본값: 0.7)
- `--max_tokens`: 최대 출력 토큰 수 (기본값: 2048)

### 대화 중 명령어

- 종료: 'exit', 'quit', '종료', '나가기'
- 초기화: 'reset', 'clear', '초기화', '리셋'
- 도움말: 'help', '도움말', '도움'
- 매개변수 설정: '온도:0.8', '최대토큰:100' 등

## 테스트 실행

기본 기능 테스트:
```
python -m unittest tests/test_agent.py
```

오류 시나리오 테스트:
```
python -m unittest tests/test_error_scenarios.py
```

## 주의사항

- Gemini API는 비율 제한이 있습니다. 자세한 내용은 [Gemini API 문서](https://ai.google.dev/gemini-api/docs/rate-limits)를 참조하세요.
- API 키는 비밀 정보이므로 공개 저장소에 커밋하지 마세요.
- 무료 등급 사용자는 분당 요청 수(RPM)와 일일 요청 수(RPD) 제한이 있습니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
