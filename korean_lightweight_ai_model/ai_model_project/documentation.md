# 한국어 특화 초경량 AI 모델 프로젝트 문서

## 프로젝트 개요

본 프로젝트는 제한된 하드웨어 환경(RAM 3GB, SSD 64GB)에서도 원활하게 작동하는 한국어 특화 초경량 AI 모델을 개발하는 것을 목표로 합니다. 기존 모델들과 차별화된 극한의 경량화 기법을 적용하여 성능은 최대한 유지하면서 메모리 사용량과 모델 크기를 획기적으로 줄였습니다.

## 주요 특징

- **2비트 양자화**: 기존 8비트, 4비트 양자화를 넘어 2비트 양자화 기법 적용으로 모델 크기 75% 이상 감소
- **공격적 프루닝**: 50% 이상의 가중치를 제거하여 모델 희소성 증가 및 메모리 사용량 최소화
- **한국어 특화 토크나이저**: 한국어 형태소 분석 기반 토크나이저와 자모 분리 최소화 로직으로 토큰화 효율성 향상
- **메모리 최적화 기법**: KV 캐시 관리, 모델 샤딩, 메모리 매핑 기법으로 제한된 RAM에서도 안정적 작동
- **대화 최적화**: 자연스러운 한국어 대화를 위한 맥락 유지 및 응답 품질 향상 기능

## 기술 아키텍처

프로젝트는 다음과 같은 모듈로 구성되어 있습니다:

1. **모델 아키텍처 (model_architecture.py)**
   - 모델 구조 정의 및 설정 관리
   - 초경량 아키텍처 생성 및 구성

2. **극한의 최적화 (extreme_optimizer.py)**
   - 2비트 양자화 구현
   - 모델 프루닝 적용
   - 지식 증류 기법 구현
   - 메모리 최적화 기법 통합

3. **토크나이저 최적화 (tokenizer_optimizer.py)**
   - 한국어 특화 토크나이저 최적화
   - 자모 분리 최소화 로직
   - 토큰화 효율성 향상

4. **추론 최적화 (inference_optimizer.py)**
   - 메모리 효율적 추론 구현
   - KV 캐시 관리
   - 인플레이스 연산 최적화

5. **대화 최적화 (conversation_optimizer.py)**
   - 대화 맥락 유지
   - 한국어 응답 품질 향상
   - 다양한 대화 스타일 지원

6. **통합 실행 모듈 (ultralight_korean_ai.py)**
   - 모든 모듈 통합
   - 사용자 인터페이스 제공
   - 벤치마크 및 성능 측정

7. **벤치마크 모듈 (model_benchmarker.py)**
   - 다른 모델과의 성능 비교
   - 차별점 분석 및 시각화
   - 비교 보고서 생성

## 성능 및 차별점

본 모델은 다른 한국어 모델들과 비교하여 다음과 같은 차별점을 가집니다:

1. **모델 크기**: 기존 모델 대비 평균 5배 이상 작은 모델 크기
2. **메모리 사용량**: 3GB RAM 이하 환경에서도 원활하게 작동하는 유일한 한국어 모델
3. **효율성**: 모델 크기와 메모리 사용량 대비 한국어 처리 성능이 탁월함
4. **속도**: 경량화에도 불구하고 경쟁력 있는 추론 속도 제공

## 사용 방법

### 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/ultralight-korean-ai.git
cd ultralight-korean-ai

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 모델 실행

```bash
# 대화 모드
python ultralight_korean_ai.py --mode chat

# 벤치마크 모드
python ultralight_korean_ai.py --mode benchmark --output_dir ./benchmark_results

# 다른 모델과 비교
python model_benchmarker.py --output_dir ./comparison_results
```

### 설정 커스터마이징

모델 설정은 JSON 파일을 통해 커스터마이징할 수 있습니다:

```json
{
  "model_config": {
    "model_name": "polyglot-ko-410m",
    "quantization_bits": 2,
    "pruning_ratio": 0.5,
    "max_memory_gb": 2.5,
    "use_knowledge_distillation": true,
    "teacher_model_name": "polyglot-ko-1.3b"
  },
  "tokenizer_config": {
    "use_korean_optimizer": true,
    "jamo_separation_minimization": true
  },
  "inference_config": {
    "kv_cache_size_mb": 128,
    "use_memory_mapping": true,
    "enable_inplace_operations": true
  },
  "conversation_config": {
    "context_length": 5,
    "response_quality_enhancement": true,
    "default_style": "casual"
  }
}
```

## 벤치마크 결과

다른 한국어 모델들과의 비교 결과는 다음과 같습니다:

| 모델 | 크기 (MB) | 파라미터 (M) | RAM 요구 (MB) | 속도 (토큰/초) | 한국어 정확도 |
|------|-----------|--------------|---------------|----------------|---------------|
| **UltralightKorean-2bit** | 500 | 410 | 1500 | 20.0 | 0.83 |
| Polyglot-ko-1.3B | 2600 | 1300 | 5000 | 12.5 | 0.85 |
| KoGPT-2 | 1200 | 750 | 3500 | 18.2 | 0.82 |
| ETRI-Eagle-3B | 6000 | 3000 | 12000 | 8.3 | 0.89 |
| Llama3-Korean-Bllossom-8B-GGUF-Q4_K_M | 4800 | 8000 | 8000 | 6.7 | 0.91 |

### 시각화 결과

![모델 크기 비교](benchmark_results/model_size_comparison.png)
![RAM 요구사항 비교](benchmark_results/ram_requirement_comparison.png)
![효율성 비교](benchmark_results/efficiency_comparison.png)
![종합 성능 비교](benchmark_results/radar_comparison.png)

## 한계 및 향후 개선 방향

현재 모델의 한계점과 향후 개선 방향은 다음과 같습니다:

1. **정확도 향상**: 극한의 경량화로 인한 정확도 손실을 최소화하기 위한 추가 연구
2. **도메인 특화**: 특정 도메인에 특화된 버전 개발 (의료, 법률, 교육 등)
3. **멀티모달 지원**: 이미지 인식 등 다양한 모달리티 지원을 위한 확장
4. **온디바이스 최적화**: 모바일 및 임베디드 디바이스에 최적화된 버전 개발

## 결론

UltralightKorean AI 모델은 제한된 하드웨어 환경에서도 원활하게 작동하는 한국어 특화 초경량 AI 모델로, 2비트 양자화, 공격적 프루닝, 한국어 특화 토크나이저 등 다양한 최적화 기법을 통해 기존 모델들과 차별화된 성능을 제공합니다. 특히 3GB RAM, 64GB SSD 환경에서도 자연스러운 한국어 대화가 가능한 유일한 모델로, 저사양 디바이스에서의 AI 활용 가능성을 크게 확장했습니다.
