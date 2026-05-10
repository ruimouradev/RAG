"""
Caching layer that stores and retrieves pre-computed search results.

Avoids re-running expensive retrieval or LLM calls for questions that have
already been processed. The cache is stored as JSON files on disk, keyed
by a hash of the question and search parameters.

This is what allows warm retrieval to process 1000 questions
in under 90 seconds.
"""

# md5 to build the cache key
import hashlib
# JSON read/write for the cache files
import json
# checking if a cache file exists on disk
import os


def compute_cache_key(
    question: str, k: int, method: str = "bm25"
) -> str:
    """
    Compute a unique cache key for a question and its search parameters.

    Uses an MD5 hash of the question text, k, and retrieval method so that
    the same question with different k values or methods produces different
    cache entries.

    Args:
        question: The search query or question text.
        k: The number of results requested.
        method: The retrieval method used (default 'bm25').

    Returns:
        A hex string that uniquely identifies this question + parameters.
    """
    raw = question + str(k) + method
    return hashlib.md5(raw.encode()).hexdigest()


def save_result_to_cache(
    cache_directory: str, cache_key: str, result: dict
) -> None:
    """
    Write a result dict to a JSON file in the cache directory.

    Args:
        cache_directory: Path to the folder where cache files are stored.
        cache_key: The key returned by compute_cache_key.
        result: The data to cache (must be JSON-serialisable).
    """
    cache_path = os.path.join(cache_directory, cache_key + '.json')
    with open(cache_path, 'w', encoding='utf-8') as file:
        json.dump(result, file)


def load_result_from_cache(
    cache_directory: str, cache_key: str
) -> dict | None:
    """
    Load a previously cached result from disk.

    Args:
        cache_directory: Path to the folder where cache files are stored.
        cache_key: The key returned by compute_cache_key.

    Returns:
        The cached dict if it exists, or None if there is no entry.
    """
    cache_path = os.path.join(cache_directory, cache_key + '.json')
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, encoding='utf-8') as file:
            result: dict = json.load(file)
            return result
    except (OSError, ValueError):
        return None
