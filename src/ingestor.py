"""
Ingestion pipeline that reads the vLLM repository and builds
the searchable knowledge base.

Walks through all .py, .md, and .rst files in the repository,
chunks each file, and saves the resulting index to disk so the
retriever can load it at search time.

Running this module should produce a persistent index
that survives between runs (no need to re-ingest on every search).
"""

# walking the repo tree and joining file paths
import os
# saving the index to disk as JSON
import json
# progress bar for the ingest loop
from tqdm import tqdm
from src.chunker import chunk_file

DEFAULT_MAX_CHUNK_SIZE = 2000


def find_all_indexable_files(repository_path: str) -> list[str]:
    """
    Recursively walk a directory and return the paths of all
    files worth indexing.

    Only .py, .md, .rst, and .txt files are included. Hidden directories,
    __pycache__, and top-level noise directories (tests, benchmarks,
    examples) are skipped.

    Args:
        repository_path: Root path of the repository to walk.

    Returns:
        A list of absolute file paths.
    """
    top_level_skip = {'tests', 'benchmarks', 'examples'}
    file_paths = []
    for root, dirs, files in os.walk(repository_path):
        is_repo_root = (os.path.normpath(root) == os.path.normpath(
            repository_path
        ))
        # skipping hidden dirs and __pycache__ always;
        # skipping noise dirs only at the repo root level
        dirs[:] = [
            d for d in dirs
            if not d.startswith('.') and d != '__pycache__'
            and not (is_repo_root and d in top_level_skip)
        ]
        for file in files:
            if file.endswith(('.py', '.md', '.rst', '.txt')):
                file_paths.append(os.path.join(root, file))
    return file_paths


def read_file_content(file_path: str) -> str:
    """
    Read the text content of a file, handling common encoding
    issues gracefully.

    Args:
        file_path: Absolute path to the file.

    Returns:
        The full text content of the file as a string.
    """
    try:
        # errors='ignore' skips any byte that isn't valid UTF-8 instead
        # of crashing — handy for the few weird files in the vLLM repo
        with open(file_path, encoding='utf-8', errors='ignore') as file:
            return file.read()
    except OSError as error:
        print(f"Warning: could not read {file_path}: {error}")
        return ''


def build_knowledge_base(
    repository_path: str,
    index_output_path: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE
) -> None:
    """
    Ingest all files in the repository and save the resulting
    index to disk.

    This is the main entry point for the ingest CLI command. It:
    1. Finds all indexable files
    2. Reads and chunks each file (with a progress bar)
    3. Saves the full list of chunks as a JSON index

    Args:
        repository_path: Root path of the vLLM repository.
        index_output_path: Path where the index file will be saved.
        max_chunk_size: Maximum characters per chunk (default 2000).
    """
    if not os.path.isdir(repository_path):
        print(f"Error: repository path '{repository_path}' does not exist.")
        return
    file_paths = find_all_indexable_files(repository_path)
    if not file_paths:
        print(f"Error: no indexable files found in '{repository_path}'.")
        return
    chunks = []
    for file_path in tqdm(file_paths, desc="Ingesting files"):
        content = read_file_content(file_path)
        for chunk in chunk_file(file_path, content, max_chunk_size):
            chunk['file_path'] = file_path
            chunks.append(chunk)
    with open(index_output_path, 'w', encoding='utf-8') as file:
        json.dump(chunks, file)


def load_knowledge_base(index_path: str) -> list[dict]:
    """
    Load a previously built index from disk.

    Args:
        index_path: Path to the saved index JSON file.

    Returns:
        A list of chunk dicts with 'text', 'file_path',
        'first_character_index', 'last_character_index'.
    """
    try:
        with open(index_path, encoding='utf-8') as file:
            result: list[dict] = json.load(file)
            return result
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Failed to load index from {index_path}"
        ) from error
