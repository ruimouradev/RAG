"""
CLI entry point — exposes all RAG commands via Python Fire.

Usage:
    uv run python -m src index
    uv run python -m src search "How to configure OpenAI server?" --k 10
    uv run python -m src search_dataset --dataset_path ... --k 10
    uv run python -m src answer "How to configure OpenAI server?" --k 10
    uv run python -m src answer_dataset --student_search_results_path ...
    uv run python -m src evaluate --search_results_path ... --ground_truth_path
"""

import json
import os

import fire
from tqdm import tqdm

from src.cache import (
    compute_cache_key,
    load_result_from_cache,
    save_result_to_cache,
)
from src.evaluator import evaluate_dataset
from src.ingestor import (
    build_knowledge_base,
    load_knowledge_base,
    DEFAULT_MAX_CHUNK_SIZE,
)
from src.llm import answer_question, load_model
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)
from src.retriever import (
    build_bm25_index,
    build_semantic_index,
    expand_query,
    search_with_bm25,
    search_with_hybrid,
    search_with_semantic,
)

DEFAULT_INDEX_PATH = "data/processed/index.json"
DEFAULT_REPO_PATH = "data/raw/vllm-0.10.1"
DEFAULT_CACHE_DIR = "data/output/cache"


def _read_chunk_text(
    file_path: str, first: int, last: int
) -> str:
    """Read the text of a chunk directly from the source file.

    Args:
        file_path: Path to the source file.
        first: Start character index.
        last: End character index.

    Returns:
        The chunk text, or an empty string if the file cannot be read.
    """
    try:
        with open(file_path, encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content[first:last]
    except OSError:
        return ''


def _load_chunks() -> list | None:
    """Load the knowledge base, printing a friendly error if it is missing.

    Returns:
        The list of chunks, or None if the index could not be loaded.
    """
    try:
        return load_knowledge_base(DEFAULT_INDEX_PATH)
    except RuntimeError as error:
        print(f"Error: {error}. Run the 'index' command first.")
        return None


def _validate_k(k: int) -> bool:
    """Check that k is a positive integer, printing an error otherwise.

    Args:
        k: The number of results requested.

    Returns:
        True if k is valid, False otherwise.
    """
    if not isinstance(k, int) or k < 1:
        print(f"Error: k must be a positive integer, got {k!r}.")
        return False
    return True


class RAG:
    """Groups all CLI commands under a single Fire-compatible class."""

    def index(self, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE) -> None:
        """Index the vLLM repo and save the knowledge base to disk.

        Args:
            max_chunk_size: Max characters per chunk (default 2000, max 2000).
        """
        if not isinstance(max_chunk_size, int) or max_chunk_size < 1:
            print(
                f"Error: max_chunk_size must be a positive integer,"
                f" got {max_chunk_size!r}."
            )
            return
        if max_chunk_size > DEFAULT_MAX_CHUNK_SIZE:
            print(
                f"Error: max_chunk_size {max_chunk_size} exceeds the maximum"
                f" allowed value of {DEFAULT_MAX_CHUNK_SIZE} characters."
            )
            return
        os.makedirs("data/processed", exist_ok=True)
        build_knowledge_base(
            DEFAULT_REPO_PATH, DEFAULT_INDEX_PATH, max_chunk_size
        )
        print(f"Index saved to {DEFAULT_INDEX_PATH}")

    def search(
        self, question: str, k: int = 10, method: str = "bm25"
    ) -> None:
        """Search the index for a single question and print results as JSON.

        Args:
            question: The query to search for.
            k: Number of results to retrieve (default 10).
            method: Retrieval method: bm25, hybrid, semantic (default bm25).
        """
        # Fire converts numeric-looking arguments to int/float
        question = str(question)
        if not _validate_k(k):
            return
        chunks = _load_chunks()
        if chunks is None:
            return
        if method == "hybrid":
            bm25_index = build_bm25_index(chunks)
            sem_model, embeddings = build_semantic_index(chunks)
            results = search_with_hybrid(
                expand_query(question), bm25_index, sem_model, embeddings,
                chunks, k
            )
        elif method == "semantic":
            sem_model, embeddings = build_semantic_index(chunks)
            results = search_with_semantic(
                expand_query(question), sem_model, embeddings, chunks, k
            )
        else:
            bm25_index = build_bm25_index(chunks)
            results = search_with_bm25(question, bm25_index, chunks, k)
        sources = [
            MinimalSource(
                file_path=r['file_path'],
                first_character_index=r['first_character_index'],
                last_character_index=r['last_character_index'],
            )
            for r in results
        ]
        q = UnansweredQuestion(question=question)
        output = MinimalSearchResults(
            question_id=q.question_id,
            question_str=question,
            retrieved_sources=sources,
        )
        print(output.model_dump_json(indent=2))

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        method: str = "bm25",
    ) -> None:
        """Run search over every question in a dataset and save results.

        Args:
            dataset_path: Path to the UnansweredQuestions JSON file.
            k: Number of results to retrieve per question (default 10).
            save_directory: Directory where results will be saved.
            method: Retrieval method: bm25, hybrid, semantic (default bm25).
        """
        if not _validate_k(k):
            return
        try:
            with open(str(dataset_path), encoding='utf-8') as f:
                dataset = json.load(f)
        except (OSError, ValueError) as e:
            print(f"Error reading dataset: {e}")
            return
        questions = (
            dataset.get('rag_questions') if isinstance(dataset, dict)
            else None
        )
        if not isinstance(questions, list):
            print("Error: dataset must contain a 'rag_questions' list.")
            return

        chunks = _load_chunks()
        if chunks is None:
            return
        if method == "hybrid":
            bm25_index = build_bm25_index(chunks)
            sem_model, embeddings = build_semantic_index(chunks)
        elif method == "semantic":
            sem_model, embeddings = build_semantic_index(chunks)
        else:
            bm25_index = build_bm25_index(chunks)

        os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
        search_results = []
        for q in tqdm(questions, desc="Searching"):
            if not isinstance(q, dict) or 'question' not in q \
                    or 'question_id' not in q:
                print(f"Warning: skipping malformed question entry: {q!r}")
                continue
            question = q['question']
            question_id = q['question_id']

            cache_key = compute_cache_key(question, k, method)
            cached = load_result_from_cache(DEFAULT_CACHE_DIR, cache_key)
            if cached is not None:
                result = MinimalSearchResults.model_validate(cached)
                search_results.append(result)
                continue

            if method == "hybrid":
                results = search_with_hybrid(
                    expand_query(question),
                    bm25_index, sem_model, embeddings, chunks, k
                )
            elif method == "semantic":
                results = search_with_semantic(
                    expand_query(question), sem_model, embeddings, chunks, k
                )
            else:
                results = search_with_bm25(question, bm25_index, chunks, k)
            sources = [
                MinimalSource(
                    file_path=r['file_path'],
                    first_character_index=r['first_character_index'],
                    last_character_index=r['last_character_index'],
                )
                for r in results
            ]
            result = MinimalSearchResults(
                question_id=question_id,
                question_str=question,
                retrieved_sources=sources,
            )
            save_result_to_cache(
                DEFAULT_CACHE_DIR, cache_key, result.model_dump()
            )
            search_results.append(result)

        output = StudentSearchResults(search_results=search_results, k=k)
        os.makedirs(save_directory, exist_ok=True)
        # output filename matches the input dataset filename
        filename = os.path.basename(dataset_path)
        save_path = os.path.join(save_directory, filename)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(output.model_dump_json(indent=2))
        print(f"Saved student_search_results to {save_path}")

    def answer(
        self,
        question: str,
        k: int = 10,
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Answer a single question using retrieved context, print as JSON.

        Args:
            question: The query to answer.
            k: Number of results to retrieve (default 10).
            model_name: HuggingFace model identifier (default Qwen3-0.6B).
        """
        # Fire converts numeric-looking arguments to int/float
        question = str(question)
        if not _validate_k(k):
            return
        chunks = _load_chunks()
        if chunks is None:
            return
        bm25_index = build_bm25_index(chunks)
        retrieved = search_with_bm25(question, bm25_index, chunks, k)
        try:
            tokenizer, model = load_model(model_name)
            answer_text = answer_question(
                question, retrieved, model, tokenizer
            )
        except Exception as gpu_err:
            # GPU was detected but doesn't work with this CUDA build,
            # switch to CPU and try again
            try:
                tokenizer, model = load_model(model_name, force_cpu=True)
                answer_text = answer_question(
                    question, retrieved, model, tokenizer
                )
            except Exception as cpu_err:
                print(
                    f"Warning: could not generate answer "
                    f"(GPU: {gpu_err}; CPU: {cpu_err})"
                )
                answer_text = ""
        sources = [
            MinimalSource(
                file_path=r['file_path'],
                first_character_index=r['first_character_index'],
                last_character_index=r['last_character_index'],
            )
            for r in retrieved
        ]
        q = UnansweredQuestion(question=question)
        output = MinimalAnswer(
            question_id=q.question_id,
            question_str=question,
            retrieved_sources=sources,
            answer=answer_text,
        )
        print(output.model_dump_json(indent=2))

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Generate answers for all questions in a search results file.

        Args:
            student_search_results_path: Path to search results JSON file.
            save_directory: Directory where answers will be saved.
            model_name: HuggingFace model identifier (default Qwen3-0.6B).
        """
        try:
            with open(
                str(student_search_results_path), encoding='utf-8'
            ) as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            print(f"Error reading search results: {e}")
            return
        results = (
            data.get('search_results') if isinstance(data, dict) else None
        )
        if not isinstance(results, list):
            print(
                "Error: search results file must contain"
                " a 'search_results' list."
            )
            return

        try:
            tokenizer, model = load_model(model_name)
            # try one tiny generation to check if the GPU is actually
            # compatible with CUDA; if not, switch to CPU before the
            # loop starts so we don't fail on every single question
            if next(model.parameters()).is_cuda:
                test_inputs = tokenizer("ok", return_tensors="pt").to(
                    model.device
                )
                model.generate(**test_inputs, max_new_tokens=1)
        except Exception as gpu_err:
            print(f"Warning: GPU unavailable ({gpu_err}); falling back to CPU")
            tokenizer, model = load_model(model_name, force_cpu=True)
        answers = []
        for result in tqdm(results, desc="Answering"):
            try:
                MinimalSearchResults.model_validate(result)
            except ValueError:
                print(f"Warning: skipping malformed search result: {result!r}")
                continue
            # re-reading the actual chunk text from disk so the LLM has context
            retrieved = [
                {
                    'text': _read_chunk_text(
                        s['file_path'],
                        s['first_character_index'],
                        s['last_character_index'],
                    ),
                    'file_path': s['file_path'],
                    'first_character_index': s['first_character_index'],
                    'last_character_index': s['last_character_index'],
                }
                for s in result['retrieved_sources']
            ]
            answer_text = answer_question(
                result['question_str'], retrieved, model, tokenizer
            )
            sources = [
                MinimalSource(
                    file_path=s['file_path'],
                    first_character_index=s['first_character_index'],
                    last_character_index=s['last_character_index'],
                )
                for s in result['retrieved_sources']
            ]
            answers.append(MinimalAnswer(
                question_id=result['question_id'],
                question_str=result['question_str'],
                retrieved_sources=sources,
                answer=answer_text,
            ))

        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=data.get('k', 10)
        )
        os.makedirs(save_directory, exist_ok=True)
        # output filename matches the input search results filename
        filename = os.path.basename(student_search_results_path)
        save_path = os.path.join(save_directory, filename)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(output.model_dump_json(indent=2))
        print(f"Saved student_search_results_and_answer to {save_path}")

    def evaluate(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 5,
    ) -> None:
        """Evaluate search results against ground truth, print Recall@k.

        Args:
            student_answer_path: Path to the student search results JSON.
            dataset_path: Path to the ground truth AnsweredQuestions JSON.
            k: Number of retrieved sources to consider (default 5).
        """
        evaluate_dataset(student_answer_path, dataset_path, k)


if __name__ == "__main__":
    fire.Fire(RAG)
