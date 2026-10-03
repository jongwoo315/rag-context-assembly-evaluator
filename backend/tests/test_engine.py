from app.data import SAMPLE_CORPUS, SAMPLE_QUESTIONS
from app.engine import ContextAssemblyEngine
from app.models import AssemblyStrategy


def test_engine_builds_chunks():
    engine = ContextAssemblyEngine(SAMPLE_CORPUS)
    assert len(engine.chunks) >= len(SAMPLE_CORPUS)


def test_top_k_returns_expected_number():
    engine = ContextAssemblyEngine(SAMPLE_CORPUS)
    result = engine.evaluate_strategy(AssemblyStrategy.TOP_K, SAMPLE_QUESTIONS, {"k": 3})
    assert result.strategy == AssemblyStrategy.TOP_K
    assert result.chunk_count <= 3
    assert 0 <= result.f1 <= 1


def test_mmr_returns_reasonable_scores():
    engine = ContextAssemblyEngine(SAMPLE_CORPUS)
    result = engine.evaluate_strategy(AssemblyStrategy.MMR, SAMPLE_QUESTIONS, {"k": 3})
    assert result.strategy == AssemblyStrategy.MMR
    assert 0 <= result.precision <= 1
    assert 0 <= result.recall <= 1


def test_failure_modes_present():
    engine = ContextAssemblyEngine(SAMPLE_CORPUS)
    result = engine.evaluate_strategy(AssemblyStrategy.TOP_K, SAMPLE_QUESTIONS)
    assert len(result.failure_modes) > 0
