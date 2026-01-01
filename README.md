# Stend: 통합 봇 플랫폼

Stend는 봇 서비스를 개발하고 관리하기 위한 통합 플랫폼입니다. Iris 프로젝트를 참고하여 개발되었으며, 빠르고 안정적인 기능을 제공합니다.

## ✨ 주요 기능

- **통합 CLI (`stend.py`)**: 인프라 기동, 빌드, 배포, 서버 실행을 하나의 명령어로 관리합니다.
- **Stend API Portal**: API 문서 확인 및 테스트를 위한 웹 대시보드를 제공합니다.
- **고속 액션 엔진**:
    - **직접 전송 (Fast Send)**: 안드로이드 인텐트를 직접 사용하여 더욱 빠른 메시지 전송이 가능합니다.
    - **백그라운드 읽음 처리**: 앱 화면 전환 없이 읽음 처리를 수행합니다.
    - **인증 정보 추출**: 카카오톡의 `aot` 및 `d_id` 정보를 추출합니다.
- **확장 도구**:
    - **웹훅 브릿지**: 실시간 이벤트를 외부 URL로 전송합니다.
    - **공유 상태 API**: 여러 인스턴스 간 데이터 공유를 지원합니다.
- **Node.js SDK**: TypeScript/Node.js 환경을 위한 전용 SDK(`stend-node-sdk`)를 제공합니다.

## 🚀 시작하기

### 사전 준비
- Python 3.8+
- Docker & Docker-Compose
- ADB (Android Debug Bridge)
- JDK 11+ (빌드용)

### 실행 방법
```bash
# 플랫폼 통합 실행 (Docker, 빌드, 배포, 서버 통합)
python stend.py start

# 대시보드 접속
# http://localhost:5001
```

## 🛠️ CLI 주요 명령어

| 명령어 | 설명 |
| :--- | :--- |
| `start` | 전체 시스템 통합 기동 |
| `stop` | 시스템 종료 |
| `build` | 안드로이드 소스 빌드 |
| `deploy` | 빌드된 APK 배포 및 실행 |
| `admin` | 관리자 명단 관리 |

## 📦 SDK 사용 예시 (Node.js)
```typescript
import { StendBot, Command } from './stend-node-sdk';

class MyBot extends StendBot {
    @Command({ trigger: 'ping' })
    async onPing(ctx, args) {
        await ctx.reply('pong!');
    }
}

const bot = new MyBot({ name: 'MyBot', endpoint: 'http://localhost:5001' });
bot.start();
```

## 📜 Acknowledgements
본 프로젝트는 **Iris** 프로젝트의 아키텍처와 로직을 참고하여 제작되었습니다.

## ⚖️ License
**MIT License**. 자세한 내용은 [LICENSE](LICENSE) 파일을 확인하세요.
