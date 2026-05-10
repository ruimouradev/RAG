"""Tests for Python and Markdown chunking strategies."""

from src.chunker import chunk_file, split_python_file_into_chunks

PYTHON_SOURCE = """\
def foo():
    return 1


def bar():
    return 2


def baz():
    x = 'a' * 3000
    return x
"""

MARKDOWN_SOURCE = """\
# Introduction

This is the intro section.

# Usage

This is the usage section.

# Advanced

This is a longer advanced section with more content.
"""


def test_python_chunks_respect_max_size() -> None:
    """No Python chunk should exceed max_chunk_size characters."""
    chunks = split_python_file_into_chunks(PYTHON_SOURCE, max_chunk_size=500)
    for chunk in chunks:
        assert len(chunk['text']) <= 500


def test_python_chunks_cover_full_file() -> None:
    """The union of all Python chunks should cover all functions."""
    chunks = split_python_file_into_chunks(PYTHON_SOURCE, max_chunk_size=2000)
    combined = ''.join(c['text'] for c in chunks)
    assert 'def foo' in combined
    assert 'def bar' in combined
    assert 'def baz' in combined


def test_python_chunks_split_on_function_boundaries() -> None:
    """Each top-level function should be in its own chunk when it fits."""
    chunks = split_python_file_into_chunks(PYTHON_SOURCE, max_chunk_size=2000)
    # foo and bar each fit in a chunk, so we expect them in separate chunks
    texts = [c['text'] for c in chunks]
    assert any('def foo' in t and 'def bar' not in t for t in texts)
    assert any('def bar' in t and 'def foo' not in t for t in texts)


def test_markdown_chunks_respect_max_size() -> None:
    """No Markdown chunk should exceed max_chunk_size characters."""
    chunks = chunk_file('doc.md', MARKDOWN_SOURCE, max_chunk_size=50)
    for chunk in chunks:
        assert len(chunk['text']) <= 50


def test_markdown_chunks_split_on_headings() -> None:
    """Each section separated by a heading should start a new chunk."""
    chunks = chunk_file('doc.md', MARKDOWN_SOURCE, max_chunk_size=2000)
    texts = [c['text'] for c in chunks]
    assert any('Introduction' in t and 'Usage' not in t for t in texts)
    assert any('Usage' in t and 'Advanced' not in t for t in texts)


def test_chunk_file_routes_python_correctly() -> None:
    """chunk_file should use the Python strategy for .py files."""
    chunks = chunk_file('module.py', PYTHON_SOURCE, max_chunk_size=2000)
    # Python strategy only returns functions/classes, not bare imports
    texts = [c['text'] for c in chunks]
    assert all('def ' in t or 'class ' in t for t in texts)


def test_chunk_file_routes_markdown_correctly() -> None:
    """chunk_file should use the Markdown strategy for .md and .rst files."""
    chunks = chunk_file('README.md', MARKDOWN_SOURCE, max_chunk_size=2000)
    assert len(chunks) > 0


def test_character_indices_are_accurate() -> None:
    """Character indices should point to the exact text in the file."""
    chunks = chunk_file('module.py', PYTHON_SOURCE, max_chunk_size=2000)
    for chunk in chunks:
        start = chunk['first_character_index']
        end = chunk['last_character_index']
        assert PYTHON_SOURCE[start:end] == chunk['text']
