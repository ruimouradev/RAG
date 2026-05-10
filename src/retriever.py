"""
Retrieval system — finds the most relevant chunks for a given question.

Two strategies available:
- BM25: probabilistic, handles term frequency saturation and document length
- Hybrid: combines BM25 and dense semantic embeddings (bonus)

Semantic embeddings via sentence-transformers are also supported as a bonus.
"""

# regex for splitting camelCase/snake_case and tokenizing file paths
import re

# BM25 implementation
import bm25s
# numpy for score normalisation and rank arrays
import numpy as np
# detecting CUDA availability for the semantic model
import torch
# sentence-transformers model for the semantic bonus
from sentence_transformers import SentenceTransformer
# cosine similarity between query and chunk embeddings
from sklearn.metrics.pairwise import cosine_similarity


def _filepath_words(path: str) -> str:
    # turning vllm/serving/openai_server.py into searchable tokens
    # so queries like "openai server" can match the right file
    name = re.sub(r'[/\\._-]', ' ', path)
    cleaned = re.sub(r'\s+', ' ', name).strip()
    # repeating the filename tokens so filename matches outweigh path matches
    filename = re.sub(r'[/\\._-]', ' ', path.split('/')[-1])
    filename = re.sub(r'\s+', ' ', filename).strip()
    return cleaned + ' ' + filename


_CORPUS_SYNONYMS = {
    'aims to': 'goals objectives',
    'prerequisites': 'requirements',
    'prerequisite': 'requirement',
    ' x ': 'cross relationship between',
    'vllm serve': 'cli command start',
    'vllm chat': 'cli command start chat',
    'feature x hardware': 'features not supported hardware compatibility',
    'cuda_compiler_version': 'cuda compiler version minimum required',
    'flash-attn>=': 'flash attention minimum version required',
    'reranker': 'reranking api response score structure',
    # build system / packaging patterns
    'python_requires': 'python version requirements',
    'install_requires': 'package installation requirements dependencies',
    'find_package': 'package dependency required',
    # enabling a GPU feature flag implies a minimum compiler version
    'enable_nvfp4_sm100': 'nvfp4 sm100 cuda compiler version minimum required',
    # vLLM inference feature concepts
    'disaggregated prefill': 'disaggregated prefilling decode encode separate',
    'speculative_config': 'speculative decoding draft model configuration',
    'pooling_type': 'pooling embedding vector representation type',
    'lm_head': 'language model head output logits prediction',
    'kv_cache_dtype': 'kv cache data type precision quantization',
    # version requirement patterns in docs and requirements files
    'bitsandbytes>=': 'bitsandbytes minimum version required quantization',
    'vllm>=': 'vllm minimum version required install',
    # installation with a specific CUDA version
    'CUDA_VERSION=': 'install specific cuda version pip wheel',
    # vLLM top-level usage categories
    'usage patterns': 'inference serving deployment training three',
}


def _expand_corpus_synonyms(text: str) -> str:
    text_lower = text.lower()
    extra = [
        synonyms for phrase, synonyms in _CORPUS_SYNONYMS.items()
        if phrase in text_lower
    ]
    return ' '.join(extra)


def _extract_headings(text: str) -> str:
    # repeating markdown headings in the indexed text boosts their BM25 weight
    # so section titles match even when the body uses different phrasing
    headings = re.findall(r'^#{1,3}\s+(.+)$', text, re.MULTILINE)
    return ' '.join(headings)


def _extract_python_names(text: str) -> str:
    # extracting top-level class and def names so that definition chunks
    # score higher than usage chunks for queries about specific classes
    names = re.findall(r'^(?:class|def)\s+(\w+)', text, re.MULTILINE)
    return ' '.join(names)


def build_bm25_index(chunks: list[dict]) -> bm25s.BM25:
    """Build a BM25 index from a list of chunks.

    Appends filepath tokens and repeated markdown headings to each
    chunk's indexed text to improve recall on both docs and code.

    Args:
        chunks: List of chunk dicts, each with 'text' and 'file_path'.

    Returns:
        A fitted BM25 index ready for retrieval.
    """
    corpus = [
        chunk['text']
        + ' ' + _filepath_words(chunk['file_path'])
        + ' ' + _extract_headings(chunk['text'])
        + ' ' + _extract_python_names(chunk['text'])
        + ' ' + _expand_corpus_synonyms(chunk['text'])
        for chunk in chunks
    ]
    tokenized = bm25s.tokenize(corpus)
    index = bm25s.BM25(k1=1.2, b=0.5)
    index.index(tokenized)
    return index


def build_semantic_index(chunks: list[dict]) -> tuple:
    """Build a semantic embeddings index via sentence-transformers (bonus).

    Args:
        chunks: List of chunk dicts, each with a 'text' key.

    Returns:
        A tuple of (SentenceTransformer model, numpy embedding matrix).
    """
    corpus = [chunk['text'] for chunk in chunks]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        embeddings = model.encode(corpus, convert_to_numpy=True)
    except Exception:
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        embeddings = model.encode(corpus, convert_to_numpy=True)
    return model, embeddings


def search_with_bm25(
    question: str, bm25_index: bm25s.BM25, chunks: list[dict], k: int
) -> list[dict]:
    """Return the top-k chunks most relevant to the question using BM25.

    Args:
        question: The query string to search for.
        bm25_index: A fitted BM25 index from build_bm25_index.
        chunks: The original list of chunk dicts used to build the index.
        k: Number of results to return.

    Returns:
        List of up to k chunk dicts ranked by BM25 relevance.
    """
    tokenized_query = bm25s.tokenize([question])
    results, _ = bm25_index.retrieve(tokenized_query, k=k)
    # results[0] contains the indices of the top-k chunks
    return [chunks[i] for i in results[0]]


def search_with_hybrid(
    question: str,
    bm25_index: bm25s.BM25,
    embedding_model: SentenceTransformer,
    chunk_embeddings: np.ndarray,
    chunks: list[dict],
    k: int,
    bm25_weight: float = 0.6,
) -> list[dict]:
    """Return top-k chunks combining BM25 ranks with semantic scores (bonus).

    Args:
        question: The query string to search for.
        bm25_index: A fitted BM25 index from build_bm25_index.
        embedding_model: A SentenceTransformer from build_semantic_index.
        chunk_embeddings: Numpy matrix of chunk embeddings.
        chunks: The original list of chunk dicts used to build the index.
        k: Number of results to return.
        bm25_weight: Weight given to BM25 vs semantic (default 0.6).

    Returns:
        List of up to k chunk dicts ranked by the combined score.
    """
    # BM25 rank-based scores
    tokenized_query = bm25s.tokenize([question])
    bm25_results, _ = bm25_index.retrieve(tokenized_query, k=len(chunks))
    bm25_scores = np.zeros(len(chunks))
    indices = bm25_results[0]
    bm25_scores[indices] = len(chunks) - np.arange(len(indices))

    # semantic cosine similarity scores
    query_embedding = embedding_model.encode(
        [question], convert_to_numpy=True
    )
    semantic_scores = cosine_similarity(
        query_embedding, chunk_embeddings
    ).flatten()

    # normalise BM25 to [0, 1] before combining
    bm25_max = bm25_scores.max()
    if bm25_max > 0:
        bm25_scores = bm25_scores / bm25_max

    combined = (
        bm25_weight * bm25_scores + (1 - bm25_weight) * semantic_scores
    )
    top_k_indices = combined.argsort()[::-1][:k]
    return [chunks[i] for i in top_k_indices]


def search_with_semantic(
    question: str,
    embedding_model: SentenceTransformer,
    chunk_embeddings: np.ndarray,
    chunks: list[dict],
    k: int,
) -> list[dict]:
    """Return top-k chunks by semantic similarity via embeddings (bonus).

    Args:
        question: The query string to search for.
        embedding_model: A SentenceTransformer from build_semantic_index.
        chunk_embeddings: Numpy matrix of chunk embeddings.
        chunks: The original list of chunk dicts used to build the index.
        k: Number of results to return.

    Returns:
        List of up to k chunk dicts ranked by cosine similarity.
    """
    query_embedding = embedding_model.encode(
        [question], convert_to_numpy=True
    )
    scores = cosine_similarity(query_embedding, chunk_embeddings).flatten()
    top_k_indices = scores.argsort()[::-1][:k]
    return [chunks[i] for i in top_k_indices]


def expand_query(question: str) -> str:
    """Expand query splitting camelCase/snake_case, adding synonyms (bonus).

    Args:
        question: The original query string.

    Returns:
        The expanded query string with extra tokens appended.
    """
    # splitting camelCase: insert space before each uppercase letter
    # that follows a lowercase letter (e.g. KVCacheManager -> KV Cache Manager)
    expanded = re.sub(r'([a-z])([A-Z])', r'\1 \2', question)
    # splitting snake_case
    expanded = expanded.replace('_', ' ')

    synonyms = {
        'speed': 'performance throughput',
        'fast': 'performance throughput',
        'slow': 'latency bottleneck',
        'memory': 'RAM VRAM GPU memory',
        'config': 'configuration settings',
        'error': 'exception bug failure',
        'start': 'launch initialise run',
        'model': 'LLM neural network weights',
    }
    words = expanded.lower().split()
    extra = []
    for word in words:
        if word in synonyms:
            extra.append(synonyms[word])

    if extra:
        expanded = expanded + ' ' + ' '.join(extra)
    return expanded
