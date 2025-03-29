"""
API 키 설정 방법 안내 문서
"""

# Gemini API 키 설정 방법

Gemini 2.0 Flash API를 사용하기 위해서는 API 키가 필요합니다. 아래 단계를 따라 API 키를 설정하세요.

## API 키 획득하기

1. [Google AI Studio](https://ai.google.dev/)에 접속합니다.
2. Google 계정으로 로그인합니다.
3. 우측 상단의 프로필 아이콘을 클릭하고 "API 키 관리"를 선택합니다.
4. "API 키 생성" 버튼을 클릭하여 새 API 키를 생성합니다.
5. 생성된 API 키를 안전하게 복사합니다.

## API 키 설정하기

1. 프로젝트 루트 디렉토리에 있는 `.env` 파일을 엽니다.
2. `GEMINI_API_KEY="YOUR_API_KEY"` 부분에서 `YOUR_API_KEY`를 실제 API 키로 교체합니다.
3. 파일을 저장합니다.

## 주의사항

- API 키는 비밀 정보이므로 공개 저장소에 커밋하지 마세요.
- `.env` 파일은 `.gitignore`에 추가하여 실수로 커밋되지 않도록 하세요.
- 무료 등급 사용자는 분당 요청 수(RPM)와 일일 요청 수(RPD) 제한이 있습니다.
