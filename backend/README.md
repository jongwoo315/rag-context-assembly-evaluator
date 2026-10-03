# RAG Context Assembly Evaluator — Backend

FastAPI 기반 RAG context-assembly 평가 백엔드입니다.

## 시작하기

```bash
cd backend
uv sync --extra test
uv run uvicorn app.main:app --reload --port 8000
```

## 테스트

```bash
uv run pytest
```

## 포트폴리오 신호

이 백엔드는 다음을 보여줍니다.

- context-assembly 당 정권 평가 (precision/recall/f1)
- 뱥지별 token 사용량을 고려한 context_precision
- miss/noise/weak_grounding/over_assembly/under_assembly failure-mode 분류
- 전략별 추천 정책 생성
