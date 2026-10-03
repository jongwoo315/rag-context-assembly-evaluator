# RAG Context Assembly Evaluator

FastAPI + Vite/React로 구성한 RAG context-assembly 평가 워크벤치입니다.

## 주요 기능

- 여러 context-assembly 전략(top-k, MMR, parent-child, rerank cutoff, token budget, citation group)을 한 번에 평가합니다.
- golden QA 셋을 기준으로 precision/recall/f1 및 context_precision을 산출합니다.
- 실패 유형을 miss/noise/weak_grounding/over_assembly/under_assembly로 분류하여 권고를 제공합니다.
- React UI에서 샘플 corpus와 question set으로 즉시 실험해 볼 수 있습니다.

## 스택

- Backend: Python, FastAPI, scikit-learn(TFIDF 기반 검색)
- Frontend: React, Vite

## 실행 방법

```bash
# backend
cd backend
uv sync --extra test
uv run uvicorn app.main:app --reload --port 8000

# frontend
cd frontend
npm install
npm run dev
```

## 포트폴리오 신호

이 프로젝트는 RAG 시스템의 검색 계층과 생성 계층 사이에 위치한 context-assembly 계층을 독립적으로 평가합니다. 단순한 retrieval 성능 추적을 넘어, 어떻게 청크를 모아 주어질 문답을 만드는지에 대한 시스템적인 시야를 제공합니다. 이는 데이터서비스 AI 역에서 RAG 운영 및 허드를 제어하는 능력을 보여줍니다.

## 관련 용어

- 수집(ingestion): 샘플 corpus를 메모리에 로드하는 단계
- 검색(retrieval): TF-IDF cosine similarity로 청크 우선순위를 계산하는 단계
- 재색인(reindex): corpus를 다시 분석하여 chunk matrix를 갱신하는 것
- 관측성(observability): 전략별 지연, failure-mode 수, 최종 추천을 로깅하여 검토할 수 있도록 노출하는 소프트웨어 특성
