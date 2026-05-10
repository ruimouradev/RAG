*This project has been created as part of the 42 curriculum by rusilva-.*

# RAG Against the Machine

## Description

A Retrieval-Augmented Generation (RAG) pipeline built around the vLLM codebase.
Given a natural language question, the system retrieves the most relevant source
chunks from the vLLM repository and uses a language model to generate a grounded
answer.

The pipeline covers the full RAG loop: ingestion -> chunking -> indexing ->
retrieval -> answer generation -> evaluation.

---

## Instructions

### Requirements

- Python ≥ 3.10
- [uv](https://github.com/astral-sh/uv) (installed automatically by `make install`)

### Installation

```bash
make install
```

Installs `uv`, creates a virtual environment, and syncs all dependencies.

### Running

```bash
make run       # start the CLI
make debug     # start with pdb debugger
make test      # run unit tests
make lint      # flake8 + mypy
make clean     # remove caches
```

---

## Example Usage

```bash
# 1 — index the vLLM repository
uv run python -m src index

# 2 — search a single question
uv run python -m src search "How does the scheduler work in vLLM?" --k 5

# 3 — search an entire dataset (docs)
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 5 --save_directory data/output/search_results

# 3b — search an entire dataset (code)
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
    --k 5 --save_directory data/output/search_results

# 4 — evaluate retrieval quality (docs)
uv run python -m src evaluate \
    data/output/search_results/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json --k 5

# 4b — evaluate retrieval quality (code)
uv run python -m src evaluate \
    data/output/search_results/dataset_code_public.json \
    data/datasets/AnsweredQuestions/dataset_code_public.json --k 5

# 5 — generate answers for a dataset (docs)
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer

# 5b — generate answers for a dataset (code)
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/dataset_code_public.json \
    --save_directory data/output/search_results_and_answer

# 6 — answer a single question
uv run python -m src answer "What is PagedAttention?" --k 5

# 7 — inspect a specific answer (compare expected vs predicted)
# change i and dataset to inspect any question from docs or code
i=42
dataset=dataset_docs_public   # or: dataset_code_public
jq -s --argjson i "$i" '
  . as [$docs, $results]
  | {
      question:  $docs.rag_questions[$i].question,
      expected:  $docs.rag_questions[$i].answer,
      predicted: $results.search_results[$i].answer
    }
' \
data/datasets/AnsweredQuestions/${dataset}.json \
data/output/search_results_and_answer/${dataset}.json
```

---

## System Architecture

```
vLLM repo (.py / .md / .rst / .txt)
        |
        v
   Ingestor          walks the repo tree, reads files
        |
        v
   Chunker           splits files into chunks with character indices
        |
        v
   BM25 Index        in-memory index saved to data/processed/index.json
        |
        v
   Retriever <-- question
        |
        v
   Top-k chunks
        |
        v
   LLM (Qwen3-0.6B) --> generates answer from retrieved context
        |
        v
   JSON output (StudentSearchResults / StudentSearchResultsAndAnswer)
```

Each component is a standalone module in `src/`:

| Module | Role |
|---|---|
| `chunker.py` | Splits files into indexable chunks |
| `ingestor.py` | Walks the repo and builds the index on disk |
| `retriever.py` | BM25, hybrid BM25+semantic, and semantic retrieval |
| `llm.py` | Answer generation via HuggingFace or vLLM server |
| `evaluator.py` | Recall@k computation with 5% overlap threshold |
| `cache.py` | MD5-keyed result cache to skip redundant calls |
| `models.py` | Pydantic models for all input/output structures |
| `__main__.py` | Python Fire CLI exposing all commands |

---

## Chunking Strategy

Two strategies are applied based on file extension:

**Python files (`.py`)** — the AST is used to locate the start and end line
of every top-level `def` and `class`. Each definition becomes one chunk.
This keeps semantically meaningful units together (a function is never split
in the middle unless it exceeds `max_chunk_size`). The character index of
each line is computed from `splitlines(keepends=True)` so that
`first_character_index` and `last_character_index` point precisely back to
the original file content.

**Markdown, RST, and plain text files (`.md`, `.rst`, `.txt`)** — sections
are delimited by heading lines (lines starting with `#`). Each section becomes
one chunk. Sections that exceed `max_chunk_size` are further split by character
count. Plain text files without headings become a single chunk (or multiple
if they exceed `max_chunk_size`). This strategy also covers build files such
as `CMakeLists.txt`, which contain version constraints and architecture
requirements relevant to code questions.

The maximum chunk size is **2000 characters** by default and is configurable
via the `--max_chunk_size` CLI argument.

---

## Retrieval Method

Primary method: **BM25** via the `bm25s` library.

BM25 (Best Match 25) is a probabilistic ranking function that improves on
TF-IDF by applying term frequency saturation (via `k1`) and document length
normalisation (via `b`). A term appearing many times in a document contributes
diminishing returns, and short documents are not unfairly penalised.

### Index enrichment

Four enrichments are applied to the text before indexing to close the
vocabulary gap between queries and source files:

1. **Filepath tokens** — the file path is split on `/`, `.`, `_`, and `-`
   and appended to the chunk text. A query mentioning "openai server" can
   therefore match `vllm/entrypoints/openai/api_server.py` even if the body
   does not contain those exact words.

2. **Heading repetition** — markdown headings (`#`, `##`, `###`) found in
   the chunk are extracted and appended. This gives section titles higher
   BM25 weight, so a query about "configuration" is more likely to find the
   "Configuration" section even when the body uses different phrasing.

3. **Python symbol names** — top-level `class` and `def` names are extracted
   from Python chunks and appended. Definition chunks score higher than usage
   chunks for queries about a specific class or function.

4. **Corpus synonyms** — a curated table maps domain-specific phrases to
   related terms (e.g. `prerequisites` → `requirements`, `vllm serve` →
   `cli command start`). When a phrase appears in a chunk, its synonyms are
   appended at index time, bridging the gap between query vocabulary and
   source vocabulary.

### Parameters

`k1=1.2, b=0.5` — selected by grid search over the public evaluation
datasets. Lower `b` (default is 0.75) reduces length normalisation, which
suits a corpus where long Python functions are inherently more informative
than short ones.

---

## Additional Features

### 1 — Hybrid BM25 + semantic retrieval

Combines BM25 rank-based scores with dense semantic cosine similarity
(default 60 % BM25, 40 % semantic). BM25 contributes a rank-based score
(`N - rank`, normalised to [0, 1]); the semantic model contributes cosine
similarity (already in [0, 1]).
Both are combined as `0.6 × bm25_score + 0.4 × semantic_score`.

```bash
uv run python -m src search "What is PagedAttention?" --k 5 --method hybrid
```

### 2 — Semantic-only retrieval

Dense retrieval using `sentence-transformers/all-MiniLM-L6-v2` (22 M params,
384-dimensional embeddings). Finds chunks by meaning rather than exact words.

```bash
uv run python -m src search "What is PagedAttention?" --k 5 --method semantic
```

### 3 — Query expansion

Splits camelCase/snake_case and appends domain synonyms before searching.
`KVCacheManager` -> `KV Cache Manager`; `memory` -> `RAM VRAM GPU memory`.

```bash
uv run python -c "from src.retriever import expand_query; print(expand_query('KVCacheManager memory speed'))"
```

### 4 — Index enrichment

Four enrichments are appended to each chunk before indexing, to close the
vocabulary gap between queries and source code: filepath tokens, repeated
Markdown headings, top-level Python `class`/`def` names, and a curated
corpus synonym table. Lifted code Recall@5 from 51% to 69%, docs to 90%
on the public dataset.

```bash
uv run python -c "
from src.retriever import (
    _filepath_words, _extract_headings,
    _extract_python_names, _expand_corpus_synonyms,
)
print(_filepath_words('vllm/entrypoints/openai/api_server.py'))
print(_extract_headings('# PagedAttention\n## KV Cache\nsome text'))
print(_extract_python_names('class KVCache:\n    pass\ndef start():\n    pass'))
print(_expand_corpus_synonyms('vllm serve and prerequisites'))
"
```

### 5 — vLLM server support

Sends prompts to a running vLLM server via HTTP (`/v1/chat/completions`,
OpenAI-compatible). Useful for remote GPU inference without loading the
model locally.

```bash
uv run python -c "from src.llm import load_vllm_model; print(load_vllm_model())"
```

### 6 — MD5 cache

Disk cache keyed by `md5(question + str(k) + method)`. Each result is
stored as a JSON file. Repeated queries are answered from disk without
re-running BM25. The method is included in the key so the same question
with different retrieval methods produces separate cache entries.

```bash
uv run python -c "from src.cache import compute_cache_key; print(compute_cache_key('What is PagedAttention?', 5))"
```

### 7 — Support for multiple LLM models

`load_model` and the CLI `--model_name` flag accept any HuggingFace model
identifier. The example below swaps the default Qwen3-0.6B for the even
smaller Qwen2.5-0.5B-Instruct without any code changes.

```bash
uv run python -m src answer "What is PagedAttention?" --k 3 --model_name "Qwen/Qwen2.5-0.5B-Instruct"
```

### Comparing retrieval methods

Run all three methods on the same dataset and evaluate to see the difference:

```bash
for method in bm25 hybrid semantic; do
  uv run python -m src search_dataset \
      data/datasets/UnansweredQuestions/dataset_docs_public.json \
      --k 10 --method $method --save_directory data/output/sr_$method
  uv run python -m src evaluate \
      data/output/sr_$method/dataset_docs_public.json \
      data/datasets/AnsweredQuestions/dataset_docs_public.json
done
```

---

## Performance Analysis

Measured on the public evaluation datasets at `k=5`:

| Dataset | Recall@5 | Minimum required |
|---|---|---|
| Documentation | **90%** | 80% |
| Code | **69%** | 50% |

**Indexing time**: ~30 seconds for the full vLLM repository.

**Cold start latency**: BM25 index is loaded from JSON in under 5 seconds.
No model loading is required for search-only operations.

**Warm retrieval throughput**: BM25 retrieval is in-memory and processes
1000 questions in well under 10 seconds after the index is loaded.

---

## Design Decisions

**Single flat JSON index** — all chunks from all file types are stored in one
`index.json` file. This simplifies the architecture: one load, one BM25
index, one search path. The subject allows separate indexes per file type,
but benchmarking showed no recall improvement from splitting.

**No re-indexing on every search** — the index is built once by `index` and
loaded by every subsequent command. This keeps cold start latency low and
avoids repeating the ~30 second ingestion step.

**`answer_dataset` re-reads chunk text from disk** — search results store
only file paths and character indices, not the full text. When generating
answers the text is re-read from the source file at the recorded indices.
This keeps the search output files small and avoids stale text in the cache.

**Pydantic models use `Sequence` instead of `List`** — `List` is invariant
in Python's type system, which causes mypy errors when assigning a
`List[MinimalAnswer]` to a `List[MinimalSearchResults]` field.
`Sequence` is covariant and solves this without workarounds.

**LLM prompt and generation params** — the system prompt instructs the model
to answer concisely using only the provided context and quote exact values,
commands, and identifiers verbatim. Generation uses greedy decoding
(`do_sample=False`) with `max_new_tokens=256`, `repetition_penalty=1.2`,
and `enable_thinking=False` to skip Qwen3's chain-of-thought tokens.
Context is trimmed to 3000 chars to mitigate the "lost in the middle"
effect (Liu et al., 2023).

---

## Challenges Faced

**Recall on code questions** — code queries tend to mention class names and
function names (e.g. `KVCacheManager`) that do not appear literally in the
chunk body. The filepath enrichment strategy was the key fix: splitting
`vllm/core/block_manager.py` into tokens `vllm core block manager` let BM25
match these queries against the right files, lifting code Recall@5
from 51% to 69% on the public dataset.

**Vocabulary mismatch in docs** — documentation queries use natural language
while the source uses technical headings. Repeating markdown heading text in
the indexed chunk gave those terms higher BM25 weight. Adding corpus synonym
expansion (e.g. mapping `prerequisites` → `requirements`, `vllm serve` →
`cli command start`) closed the remaining vocabulary gaps, lifting docs
Recall@5 to 90% on the public dataset.

---

## Resources

**Libraries**
- [bm25s](https://github.com/xhluca/bm25s) — fast BM25 implementation
- [sentence-transformers](https://www.sbert.net/) — semantic embeddings
- [transformers](https://huggingface.co/docs/transformers) — HuggingFace model loading
- [pydantic](https://docs.pydantic.dev/) — data validation and serialisation
- [Python Fire](https://github.com/google/python-fire) — CLI generation

**Articles and references**
- Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval.
- Lewis, P. et al. (2020). [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*](https://arxiv.org/abs/2005.11401) NeurIPS.
- Salton, G. & McGill, M. (1983). *Introduction to Modern Information Retrieval.* McGraw-Hill.
- Manning, C., Raghavan, P. & Schütze, H. [*Introduction to Information Retrieval.*](https://nlp.stanford.edu/IR-book/) Cambridge University Press. (free online)
- Gao, Y. et al. (2024). [*Retrieval-Augmented Generation for LLMs: A Survey.*](https://arxiv.org/abs/2312.10997) arXiv.
- Liu, N. et al. (2023). [*Lost in the Middle: How Language Models Use Long Contexts.*](https://arxiv.org/abs/2307.03172) arXiv.

**Videos**
- Karpathy, A. [*Intro to Large Language Models.*](https://www.youtube.com/watch?v=zjkBMFhNj_g) YouTube, 1h.
- 3Blue1Brown. [*But what is a GPT? Visual intro to transformers.*](https://www.youtube.com/watch?v=wjZofJX0v4M) YouTube, 26min.
- LangChain. [*RAG From Scratch* (14-video playlist).](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) YouTube.

**AI usage**

AI was used throughout this project. Specific tasks included:

- Discussing benchmarking retrieval strategies (BM25 parameter grid search, heading
  boost, filepath enrichment) and interpreting results
- Helping diagnose and explain runtime errors during development
- Reviewing and refining docstrings to PEP 257 Google style
- Helping writing and organising this README
