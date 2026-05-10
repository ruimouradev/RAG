"""
Splits file contents into smaller chunks suitable for indexing and retrieval.

Supports two chunking strategies:
- Python files: splits on function and class boundaries using the AST
- Markdown and plain text files: splits on headings and blank lines

Each chunk records its start and end character positions in the original file,
so results can be traced back to an exact location in the source.
"""

import ast
import re


def _size_split(
    text: str, abs_offset: int, max_chunk_size: int,
    window: int = 0,
) -> list[dict]:
    """Split text into chunks, preferring natural boundaries.

    Instead of splitting at exactly max_chunk_size chars, searches
    backwards from each ideal split point for a blank line or a line
    starting with # (headings, comments) within the given window.
    Defaults to max_chunk_size // 4 if window is 0.

    Args:
        text: The text to split.
        abs_offset: Character offset of text in the original file.
        max_chunk_size: Maximum chunk size in characters.

    Returns:
        List of chunk dicts with text and character indices.
    """
    chunks = []
    pos = 0
    # a non-positive size would keep pos from ever advancing
    max_chunk_size = max(1, max_chunk_size)
    if window == 0:
        window = max_chunk_size // 4

    while pos < len(text):
        ideal_end = min(pos + max_chunk_size, len(text))
        if ideal_end == len(text):
            end = ideal_end
        else:
            search_start = max(pos + 1, ideal_end - window)
            segment = text[search_start:ideal_end]
            best = -1
            blank_line_pos = segment.rfind('\n\n')
            if blank_line_pos >= 0:
                best = search_start + blank_line_pos + 2
            for match in re.finditer(r'\n[ \t]*#', segment):
                candidate = search_start + match.start() + 1
                if candidate > best:
                    best = candidate
            end = best if best > search_start else ideal_end

        chunk_text = text[pos:end]
        chunks.append({
            "text": chunk_text,
            "first_character_index": abs_offset + pos,
            "last_character_index": abs_offset + end,
        })
        pos = end

    return chunks


def split_python_file_into_chunks(
    file_content: str, max_chunk_size: int
) -> list[dict]:
    """
    Split a Python source file into chunks based on top-level functions
    and classes.

    Uses the AST to find natural boundaries (def, class) so chunks are
    semantically meaningful. Falls back to size-based splitting when a
    single function or class exceeds max_chunk_size.

    Args:
        file_content: The full text content of the Python file.
        max_chunk_size: Maximum number of characters allowed per chunk.

    Returns:
        A list of dicts with keys
        'text', 'first_character_index', 'last_character_index'.
    """
    # AST gives us the line numbers where each function/class starts and ends
    tree = ast.parse(file_content)

    # Finding the length of each line to get the character position where
    # each line starts
    # (keepends=True includes the \n, so the length of line N is exactly
    # the offset of line N+1)
    lines = file_content.splitlines(keepends=True)
    line_offsets = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)

    chunks = []

    # iterating over all top-level nodes (imports, functions, classes, etc.)
    # only keeping functions and classes since those are the meaningful chunks
    for node in ast.iter_child_nodes(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue

        # node.lineno is the first line of the def/class, end_lineno the
        # last; we convert these to character offsets via line_offsets
        # the None check is just for mypy — in practice they're always set
        if node.lineno is None or node.end_lineno is None:
            continue
        start = line_offsets[node.lineno - 1]
        last_line = lines[node.end_lineno - 1]
        end = line_offsets[node.end_lineno - 1] + len(last_line)
        text = file_content[start:end]

        if len(text) <= max_chunk_size:
            chunks.append({
                "text": text,
                "first_character_index": start,
                "last_character_index": end
            })
        else:
            # Splitting very large functions so the retriever never receives
            # a chunk larger than it can handle
            chunks.extend(_size_split(text, start, max_chunk_size))

    return chunks


def split_markdown_file_into_chunks(
    file_content: str, max_chunk_size: int, window: int = 0,
) -> list[dict]:
    """
    Split a Markdown or plain text file into chunks based on headings.

    Headings (lines starting with #) are treated as section boundaries.
    Chunks that exceed max_chunk_size are further split by size.

    Args:
        file_content: The full text content of the Markdown file.
        max_chunk_size: Maximum number of characters allowed per chunk.
        window: Boundary search window for size splits (0 = default).

    Returns:
        A list of dicts with keys
        'text', 'first_character_index', 'last_character_index'.
    """
    # Finding the length of each line to get the character position where
    # each line starts (keepends=True includes the \n, so the length of
    # line N is exactly the offset of line N+1)
    lines = file_content.splitlines(keepends=True)
    line_offsets = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)

    chunks = []
    section_start = 0

    for i, line in enumerate(lines):
        if line.startswith('#') and i > 0:
            section_text = file_content[section_start:line_offsets[i]]

            if len(section_text) <= max_chunk_size:
                chunks.append({
                    "text": section_text,
                    "first_character_index": section_start,
                    "last_character_index": line_offsets[i]
                })
            else:
                chunks.extend(
                    _size_split(
                        section_text, section_start, max_chunk_size, window
                    )
                )

            section_start = line_offsets[i]

    # close the last section by hand — the loop only closes a section
    # when it finds the next heading, and the last one has no heading after
    last_section_text = file_content[section_start:]
    if last_section_text.strip():
        if len(last_section_text) <= max_chunk_size:
            chunks.append({
                "text": last_section_text,
                "first_character_index": section_start,
                "last_character_index": len(file_content)
            })
        else:
            chunks.extend(
                _size_split(
                    last_section_text, section_start, max_chunk_size, window
                )
            )
    return chunks


def chunk_file(
    file_path: str, file_content: str, max_chunk_size: int
) -> list[dict]:
    """
    Route a file to the correct chunking strategy based on its extension.

    .py files go to split_python_file_into_chunks.
    .md and .rst files go to split_markdown_file_into_chunks.
    Unknown extensions are treated as plain text.

    Args:
        file_path: Path to the file (used only to determine the extension).
        file_content: The full text content of the file.
        max_chunk_size: Maximum number of characters allowed per chunk.

    Returns:
        A list of dicts with keys
        'text', 'first_character_index', 'last_character_index'.
    """
    if file_path.endswith(".py"):
        return split_python_file_into_chunks(file_content, max_chunk_size)
    elif file_path.endswith(".txt"):
        # plain-text files like CMakeLists.txt don't have a regular
        # heading structure, so we let the splitter look further back
        # for a good cut point before forcing a hard split
        txt_window = max_chunk_size * 3 // 4
        return split_markdown_file_into_chunks(
            file_content, max_chunk_size, txt_window
        )
    else:
        return split_markdown_file_into_chunks(file_content, max_chunk_size)
