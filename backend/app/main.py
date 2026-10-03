from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.data import SAMPLE_CORPUS, SAMPLE_QUESTIONS
from app.engine import ContextAssemblyEngine
from app.models import AssemblyStrategy, CorpusUpload, EvalResult, PolicyRun, QuestionSetUpload

ENV_PATH = Path(__file__).resolve().parents[3] / "secrets" / "secrets.env"
load_dotenv(ENV_PATH, override=True)

runs: dict[str, PolicyRun] = {}
corpora: dict[str, list[dict[str, Any]]] = {}
question_sets: dict[str, list[dict[str, Any]]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # seed default data
    default_corpus_id = str(uuid.uuid4())[:8]
    default_questions_id = str(uuid.uuid4())[:8]
    corpora[default_corpus_id] = SAMPLE_CORPUS
    question_sets[default_questions_id] = SAMPLE_QUESTIONS
    app.state.default_corpus_id = default_corpus_id
    app.state.default_questions_id = default_questions_id
    yield


app = FastAPI(title="RAG Context Assembly Evaluator")

# seed defaults synchronously so TestClient can read them without lifespan
if not hasattr(app.state, "default_corpus_id"):
    _default_corpus_id = str(uuid.uuid4())[:8]
    _default_questions_id = str(uuid.uuid4())[:8]
    corpora[_default_corpus_id] = SAMPLE_CORPUS
    question_sets[_default_questions_id] = SAMPLE_QUESTIONS
    app.state.default_corpus_id = _default_corpus_id
    app.state.default_questions_id = _default_questions_id
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config/defaults")
def defaults():
    return {
        "corpus_id": app.state.default_corpus_id,
        "questions_id": app.state.default_questions_id,
    }


@app.post("/api/corpus")
def upload_corpus(payload: CorpusUpload) -> dict[str, str]:
    corpus_id = str(uuid.uuid4())[:8]
    corpora[corpus_id] = payload.documents
    return {"corpus_id": corpus_id, "count": str(len(payload.documents))}


@app.post("/api/questions")
def upload_questions(payload: QuestionSetUpload) -> dict[str, str]:
    questions_id = str(uuid.uuid4())[:8]
    question_sets[questions_id] = payload.questions
    return {"questions_id": questions_id, "count": str(len(payload.questions))}


@app.get("/api/corpus/{corpus_id}")
def get_corpus(corpus_id: str):
    docs = corpora.get(corpus_id, [])
    return {"documents": docs, "count": len(docs)}


@app.get("/api/questions/{questions_id}")
def get_questions(questions_id: str):
    questions = question_sets.get(questions_id, [])
    return {"questions": questions, "count": len(questions)}


@app.post("/api/eval/{corpus_id}/{questions_id}")
def run_eval(corpus_id: str, questions_id: str, config: dict[str, Any] | None = None) -> PolicyRun:
    config = config or {}
    docs = corpora.get(corpus_id, [])
    questions = question_sets.get(questions_id, [])
    if not docs or not questions:
        return PolicyRun(run_id=str(uuid.uuid4())[:8], status="missing_data")

    run_id = str(uuid.uuid4())[:8]
    engine = ContextAssemblyEngine(docs)

    results: list[EvalResult] = []
    for strategy in AssemblyStrategy:
        result = engine.evaluate_strategy(strategy, questions, config)
        results.append(result)

    best = max(results, key=lambda r: r.f1)
    run = PolicyRun(
        run_id=run_id,
        status="ok",
        results=results,
        best_strategy=best.strategy,
        best_f1=best.f1,
    )
    runs[run_id] = run
    return run


@app.get("/api/eval/{run_id}")
def get_eval(run_id: str):
    return runs.get(run_id, PolicyRun(run_id=run_id, status="not_found"))


@app.get("/api/strategies")
def list_strategies():
    return [{"value": s.value, "label": s.value.replace("_", " ").title()} for s in AssemblyStrategy]
