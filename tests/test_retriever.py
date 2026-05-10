"""Tests for BM25 and hybrid retrieval."""

from src.retriever import (
    build_bm25_index,
    build_semantic_index,
    expand_query,
    search_with_bm25,
    search_with_hybrid,
    search_with_semantic,
)

CHUNKS = [
    {
        'text': 'vLLM uses PagedAttention for efficient KV cache management.',
        'file_path': 'vllm/core.py',
        'first_character_index': 0,
        'last_character_index': 58,
    },
    {
        'text': 'The scheduler assigns requests to GPU workers.',
        'file_path': 'vllm/scheduler.py',
        'first_character_index': 0,
        'last_character_index': 46,
    },
    {
        'text': 'Tokenization converts text into integer token ids.',
        'file_path': 'vllm/tokenizer.py',
        'first_character_index': 0,
        'last_character_index': 50,
    },
]


def test_bm25_returns_correct_number_of_results() -> None:
    """search_with_bm25 should return exactly k results when k <= chunks."""
    index = build_bm25_index(CHUNKS)
    results = search_with_bm25('PagedAttention cache', index, CHUNKS, k=2)
    assert len(results) == 2


def test_bm25_ranks_relevant_chunk_first() -> None:
    """The chunk most relevant to the query should appear at position 0."""
    index = build_bm25_index(CHUNKS)
    results = search_with_bm25('PagedAttention cache', index, CHUNKS, k=3)
    assert results[0]['file_path'] == 'vllm/core.py'


def test_semantic_returns_correct_number_of_results() -> None:
    """search_with_semantic should return exactly k results."""
    model, embeddings = build_semantic_index(CHUNKS)
    results = search_with_semantic(
        'scheduler GPU', model, embeddings, CHUNKS, k=2
    )
    assert len(results) == 2


def test_semantic_ranks_relevant_chunk_first() -> None:
    """The chunk most semantically similar to the query ranks first."""
    model, embeddings = build_semantic_index(CHUNKS)
    results = search_with_semantic(
        'scheduler GPU workers', model, embeddings, CHUNKS, k=3
    )
    assert results[0]['file_path'] == 'vllm/scheduler.py'


def test_hybrid_combines_both_scores() -> None:
    """Hybrid results should rank the most relevant chunk first."""
    bm25_index = build_bm25_index(CHUNKS)
    model, embeddings = build_semantic_index(CHUNKS)
    results = search_with_hybrid(
        'PagedAttention cache', bm25_index, model, embeddings, CHUNKS, k=3
    )
    assert len(results) == 3
    assert results[0]['file_path'] == 'vllm/core.py'


def test_query_expansion_splits_camel_case() -> None:
    """expand_query should split 'KVCacheManager' into separate tokens."""
    expanded = expand_query('KVCacheManager')
    assert 'Cache' in expanded
    assert 'Manager' in expanded


def test_search_returns_minimal_source_compatible_dicts() -> None:
    """Each result dict should have the three MinimalSource fields."""
    index = build_bm25_index(CHUNKS)
    results = search_with_bm25('tokenization', index, CHUNKS, k=1)
    assert 'file_path' in results[0]
    assert 'first_character_index' in results[0]
    assert 'last_character_index' in results[0]
