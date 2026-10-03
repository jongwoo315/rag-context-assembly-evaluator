from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AssemblyStrategy(str, Enum):
    TOP_K = "top_k"
    MMR = "mmr"
    PARENT_CHILD = "parent_child"
    RERANK_CUTOFF = "rerank_cutoff"
    TOKEN_BUDGET = "token_budget"
    CITATION_GROUP = "citation_group"


class EvalResult(BaseModel):
    strategy: AssemblyStrategy
    precision: float
    recall: float
    f1: float
    context_precision: float
    token_count: int
    chunk_count: int
    latency_ms: int
    failure_modes: dict[str, int] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)


class PolicyRun(BaseModel):
    run_id: str
    status: str
    results: list[EvalResult] = Field(default_factory=list)
    best_strategy: AssemblyStrategy | None = None
    best_f1: float = 0.0


class CorpusUpload(BaseModel):
    documents: list[dict[str, Any]]


class QuestionSetUpload(BaseModel):
    questions: list[dict[str, Any]]
