from __future__ import annotations

import random
import time
from collections import Counter
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import AssemblyStrategy, EvalResult

random.seed(42)
np.random.seed(42)


class ContextAssemblyEngine:
    def __init__(self, documents: list[dict[str, Any]]):
        self.docs = documents
        self.chunks: list[dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.chunk_matrix = None
        self._build_chunks()

    def _build_chunks(self):
        for doc_idx, doc in enumerate(self.docs):
            text = doc.get("text", "")
            # simple paragraph chunking
            parts = [p.strip() for p in text.split("\n\n") if p.strip()]
            if not parts:
                parts = [text]
            for chunk_idx, chunk_text in enumerate(parts):
                self.chunks.append(
                    {
                        "doc_id": doc.get("id", f"doc_{doc_idx}"),
                        "chunk_id": f"{doc.get('id', f'doc_{doc_idx}')}_c{chunk_idx}",
                        "text": chunk_text,
                        "parent_id": doc.get("id", f"doc_{doc_idx}"),
                        "citation": doc.get("source", f"source_{doc_idx}"),
                        "metadata": doc.get("metadata", {}),
                    }
                )
        texts = [c["text"] for c in self.chunks]
        if len(texts) >= 2:
            self.chunk_matrix = self.vectorizer.fit_transform(texts)
        else:
            self.chunk_matrix = None

    def _query_vec(self, query: str) -> Any:
        if self.chunk_matrix is None:
            return None
        return self.vectorizer.transform([query])

    def _retrieve_scores(self, query: str) -> np.ndarray:
        if self.chunk_matrix is None:
            return np.zeros(len(self.chunks))
        q = self._query_vec(query)
        scores = cosine_similarity(q, self.chunk_matrix).flatten()
        return scores

    def _expected_chunks(self, question: dict[str, Any]) -> set[str]:
        expected = question.get("expected_chunk_ids", [])
        if expected:
            return set(expected)
        expected_doc = question.get("expected_doc_id")
        if expected_doc:
            return {
                c["chunk_id"]
                for c in self.chunks
                if c["doc_id"] == expected_doc or c["parent_id"] == expected_doc
            }
        return set()

    def evaluate_strategy(
        self, strategy: AssemblyStrategy, questions: list[dict[str, Any]], config: dict[str, Any] | None = None
    ) -> EvalResult:
        config = config or {}
        k = config.get("k", 5)
        mmr_lambda = config.get("mmr_lambda", 0.5)
        token_budget = config.get("token_budget", 512)
        rerank_cutoff = config.get("rerank_cutoff", 0.3)
        min_f1 = config.get("min_f1", 0.5)

        start = time.perf_counter()
        per_question: list[dict[str, Any]] = []

        for q in questions:
            query = q["query"]
            selected = self._assemble(query, strategy, k, mmr_lambda, token_budget, rerank_cutoff)
            expected = self._expected_chunks(q)
            retrieved_ids = {c["chunk_id"] for c in selected}
            per_question.append(
                {
                    "selected": selected,
                    "expected": expected,
                    "retrieved_ids": retrieved_ids,
                    "precision": self._precision(retrieved_ids, expected),
                    "recall": self._recall(retrieved_ids, expected),
                    "f1": self._f1(retrieved_ids, expected),
                    "token_count": sum(len(c["text"].split()) for c in selected),
                    "chunk_count": len(selected),
                }
            )

        latency_ms = int((time.perf_counter() - start) * 1000)

        precisions = [p["precision"] for p in per_question]
        recalls = [p["recall"] for p in per_question]
        f1s = [p["f1"] for p in per_question]
        context_precisions = [
            p["precision"] * (1 - abs(p["chunk_count"] - len(p["expected"])) / max(p["chunk_count"] + len(p["expected"]), 1))
            for p in per_question
        ]
        token_counts = [p["token_count"] for p in per_question]
        chunk_counts = [p["chunk_count"] for p in per_question]

        failure_modes = Counter()
        for p in per_question:
            if p["recall"] == 0:
                failure_modes["miss"] += 1
            elif p["precision"] < 0.5:
                failure_modes["noise"] += 1
            elif p["f1"] < min_f1:
                failure_modes["weak_grounding"] += 1
            elif p["chunk_count"] > k:
                failure_modes["over_assembly"] += 1
            elif p["chunk_count"] < len(p["expected"]):
                failure_modes["under_assembly"] += 1
            else:
                failure_modes["ok"] += 1

        recommendations = self._recommend(strategy, failure_modes, config)

        return EvalResult(
            strategy=strategy,
            precision=round(float(np.mean(precisions)), 3),
            recall=round(float(np.mean(recalls)), 3),
            f1=round(float(np.mean(f1s)), 3),
            context_precision=round(float(np.mean(context_precisions)), 3),
            token_count=int(np.mean(token_counts)),
            chunk_count=int(np.mean(chunk_counts)),
            latency_ms=latency_ms,
            failure_modes=dict(failure_modes),
            recommendations=recommendations,
        )

    def _assemble(
        self,
        query: str,
        strategy: AssemblyStrategy,
        k: int,
        mmr_lambda: float,
        token_budget: int,
        rerank_cutoff: float,
    ) -> list[dict[str, Any]]:
        scores = self._retrieve_scores(query)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        if strategy == AssemblyStrategy.TOP_K:
            top = indexed[:k]
            return [self.chunks[i] for i, _ in top]

        if strategy == AssemblyStrategy.MMR:
            return self._mmr(query, k, mmr_lambda)

        if strategy == AssemblyStrategy.PARENT_CHILD:
            top_ids = {i for i, _ in indexed[:k]}
            selected: list[dict[str, Any]] = []
            seen_parents = set()
            for i, _ in indexed:
                if len(selected) >= k * 2:
                    break
                chunk = self.chunks[i]
                if chunk["parent_id"] not in seen_parents:
                    seen_parents.add(chunk["parent_id"])
                    # add siblings
                    siblings = [c for c in self.chunks if c["parent_id"] == chunk["parent_id"]]
                    selected.extend(siblings[:3])
            return selected[: k * 2]

        if strategy == AssemblyStrategy.RERANK_CUTOFF:
            reranked = [i for i, s in indexed if s >= rerank_cutoff]
            return [self.chunks[i] for i in reranked[:k]]

        if strategy == AssemblyStrategy.TOKEN_BUDGET:
            selected = []
            tokens = 0
            for i, _ in indexed:
                chunk = self.chunks[i]
                ntok = len(chunk["text"].split())
                if tokens + ntok > token_budget:
                    break
                selected.append(chunk)
                tokens += ntok
                if len(selected) >= k * 2:
                    break
            return selected

        if strategy == AssemblyStrategy.CITATION_GROUP:
            top = indexed[: k * 2]
            by_citation: dict[str, list[dict[str, Any]]] = {}
            for i, _ in top:
                c = self.chunks[i]
                by_citation.setdefault(c["citation"], []).append(c)
            selected = []
            for citation in sorted(by_citation, key=lambda c: len(by_citation[c]), reverse=True):
                selected.extend(by_citation[citation][:2])
                if len(selected) >= k:
                    break
            return selected[:k]

        return []

    def _mmr(self, query: str, k: int, lambda_param: float) -> list[dict[str, Any]]:
        if self.chunk_matrix is None or not self.chunks:
            return []
        q_vec = self._query_vec(query)
        doc_sims = cosine_similarity(q_vec, self.chunk_matrix).flatten()
        selected: list[int] = []
        remaining = set(range(len(self.chunks)))
        while len(selected) < k and remaining:
            best_score = -1.0
            best_idx = -1
            for idx in remaining:
                relevance = doc_sims[idx]
                if selected:
                    sim_to_selected = cosine_similarity(
                        self.chunk_matrix[idx], self.chunk_matrix[selected]
                    ).max()
                else:
                    sim_to_selected = 0.0
                score = lambda_param * relevance - (1 - lambda_param) * sim_to_selected
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
        return [self.chunks[i] for i in selected]

    @staticmethod
    def _precision(retrieved: set[str], expected: set[str]) -> float:
        if not retrieved:
            return 0.0
        if not expected:
            return 1.0
        return len(retrieved & expected) / len(retrieved)

    @staticmethod
    def _recall(retrieved: set[str], expected: set[str]) -> float:
        if not expected:
            return 1.0
        return len(retrieved & expected) / len(expected)

    @staticmethod
    def _f1(retrieved: set[str], expected: set[str]) -> float:
        p = ContextAssemblyEngine._precision(retrieved, expected)
        r = ContextAssemblyEngine._recall(retrieved, expected)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @staticmethod
    def _recommend(strategy: AssemblyStrategy, failure_modes: Counter, config: dict[str, Any]) -> list[str]:
        recs = []
        if failure_modes["miss"] > 0:
            recs.append(f"{strategy.value}: 검색 단계에서 기대 청크를 놓치는 경우가 많습니다. top_k를 늘리거나 hybrid search를 고려하세요.")
        if failure_modes["noise"] > 0:
            recs.append(f"{strategy.value}: 관련 없는 청크가 assembly에 섞입니다. reranker threshold를 높이거나 MMR을 적용하세요.")
        if failure_modes["weak_grounding"] > 0:
            recs.append(f"{strategy.value}: 선택된 청크가 질문에 충분히 근거를 제공하지 못합니다. parent-child 확장이나 citation grouping을 시도하세요.")
        if failure_modes["over_assembly"] > 0:
            recs.append(f"{strategy.value}: 불필요하게 많은 청크가 포함됩니다. token_budget을 낮추거나 rerank cutoff를 강화하세요.")
        if failure_modes["under_assembly"] > 0:
            recs.append(f"{strategy.value}: 기대하는 답변을 완성하기에 청크가 부족합니다. parent-child 확장을 늘리세요.")
        if not recs:
            recs.append(f"{strategy.value}: 현재 설정에서 큰 문제는 보이지 않습니다. 실제 LLM 답변과의 일치도를 추가 평가하세요.")
        return recs
