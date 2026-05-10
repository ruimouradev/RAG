"""
Evaluates retrieval quality using Recall@k with a 5% overlap threshold.
"""

# reading the search results and ground truth from JSON
import json


def check_source_overlap(
    retrieved_source: dict,
    ground_truth_source: dict,
    minimum_overlap_ratio: float = 0.05
) -> bool:
    """
    Return True if retrieved and ground truth source overlap by at
    least minimum_overlap_ratio.

    Args:
        retrieved_source: Chunk returned by the retriever.
        ground_truth_source: Chunk from the ground truth dataset.
        minimum_overlap_ratio: Minimum fraction of overlap required.

    Returns:
        True if the two chunks overlap enough, False otherwise.
    """
    # only chunks from the same file can overlap
    if retrieved_source['file_path'] != ground_truth_source['file_path']:
        return False

    overlap_start = max(
        int(retrieved_source['first_character_index']),
        int(ground_truth_source['first_character_index'])
    )
    overlap_end = min(
        int(retrieved_source['last_character_index']),
        int(ground_truth_source['last_character_index'])
    )
    overlap_length = max(0, overlap_end - overlap_start)

    ground_truth_length = (
        int(ground_truth_source['last_character_index'])
        - int(ground_truth_source['first_character_index'])
    )
    if ground_truth_length == 0:
        return False

    return (overlap_length / ground_truth_length) >= minimum_overlap_ratio


def calculate_recall_at_k(
    search_results: list[dict],
    ground_truth_questions: list[dict],
    k: int
) -> float:
    """
    Calculate Recall@k across all questions.

    A source is considered found if it overlaps with the ground truth
    by at least 5%. A question is considered answered if at least one
    of its ground truth sources is found in the top-k results.

    Args:
        search_results: List of per-question retrieval results.
        ground_truth_questions: List of answered questions with sources.
        k: Number of retrieved sources to consider.

    Returns:
        Recall@k as a float between 0 and 1.
    """
    ground_truth_by_id = {
        q['question_id']: q['sources']
        for q in ground_truth_questions
    }

    total_score = 0.0
    total = 0

    for result in search_results:
        question_id = result['question_id']
        if question_id not in ground_truth_by_id:
            continue

        total += 1
        top_k_sources = result['retrieved_sources'][:k]
        ground_truth_sources = ground_truth_by_id[question_id]

        number_found = sum(
            1 for ground_truth in ground_truth_sources
            if any(
                check_source_overlap(retrieved, ground_truth)
                for retrieved in top_k_sources
            )
        )
        total_score += number_found / len(ground_truth_sources)

    if total == 0:
        return 0.0
    return total_score / total


def evaluate_dataset(
    search_results_path: str,
    ground_truth_path: str,
    k: int = 5,
) -> dict:
    """
    Load search results and ground truth from disk, run evaluation,
    print and return the Recall@k score.

    Args:
        search_results_path: Path to the search results JSON file.
        ground_truth_path: Path to the ground truth JSON file.
        k: Number of retrieved sources to consider (default 5).

    Returns:
        A dict with the recall score.
    """
    try:
        with open(search_results_path, encoding='utf-8') as file:
            search_results = json.load(file)['search_results']
        with open(ground_truth_path, encoding='utf-8') as file:
            ground_truth = json.load(file)['rag_questions']
    except (OSError, ValueError, KeyError) as e:
        print(f"Error loading evaluation files: {e}")
        return {}

    recall = calculate_recall_at_k(search_results, ground_truth, k)
    print(f"Recall@{k}: {recall:.4f}")
    return {'recall': recall}
