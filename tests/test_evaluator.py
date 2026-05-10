"""Tests for Recall@k calculation and overlap detection."""

from src.evaluator import calculate_recall_at_k, check_source_overlap


def _make_source(
    file_path: str, first: int, last: int
) -> dict:
    return {
        'file_path': file_path,
        'first_character_index': first,
        'last_character_index': last,
    }


def test_overlap_detected_when_ranges_share_enough_characters() -> None:
    """check_source_overlap returns True when overlap >= 5% of ground truth."""
    retrieved = _make_source('a.py', 0, 100)
    ground_truth = _make_source('a.py', 50, 150)
    assert check_source_overlap(retrieved, ground_truth) is True


def test_overlap_rejected_when_ranges_share_too_few_characters() -> None:
    """check_source_overlap returns False when overlap < 5% threshold."""
    retrieved = _make_source('a.py', 0, 10)
    # ground truth spans 1000 chars, overlap is only 1 char (0.1%)
    ground_truth = _make_source('a.py', 9, 1009)
    assert check_source_overlap(retrieved, ground_truth) is False


def test_perfect_recall_when_all_sources_found() -> None:
    """Recall@k should be 1.0 when every ground truth source is retrieved."""
    search_results = [
        {
            'question_id': 'q1',
            'retrieved_sources': [_make_source('a.py', 0, 100)],
        }
    ]
    ground_truth = [
        {
            'question_id': 'q1',
            'sources': [_make_source('a.py', 0, 100)],
        }
    ]
    assert calculate_recall_at_k(search_results, ground_truth, k=5) == 1.0


def test_zero_recall_when_no_sources_found() -> None:
    """Recall@k should be 0.0 when no retrieved source matches ground truth."""
    search_results = [
        {
            'question_id': 'q1',
            'retrieved_sources': [_make_source('b.py', 0, 100)],
        }
    ]
    ground_truth = [
        {
            'question_id': 'q1',
            'sources': [_make_source('a.py', 0, 100)],
        }
    ]
    assert calculate_recall_at_k(search_results, ground_truth, k=5) == 0.0


def test_recall_is_proportion_of_questions_answered() -> None:
    """Recall@k equals the fraction of questions with at least one hit."""
    search_results = [
        {
            'question_id': 'q1',
            'retrieved_sources': [_make_source('a.py', 0, 100)],
        },
        {
            'question_id': 'q2',
            'retrieved_sources': [_make_source('b.py', 0, 100)],
        },
    ]
    ground_truth = [
        {
            'question_id': 'q1',
            'sources': [_make_source('a.py', 0, 100)],
        },
        {
            'question_id': 'q2',
            'sources': [_make_source('c.py', 0, 100)],
        },
    ]
    assert calculate_recall_at_k(search_results, ground_truth, k=5) == 0.5


def test_recall_ignores_results_beyond_k() -> None:
    """Sources ranked beyond position k should not contribute to recall."""
    search_results = [
        {
            'question_id': 'q1',
            # matching source is at position 2 (index 2), beyond k=1
            'retrieved_sources': [
                _make_source('x.py', 0, 100),
                _make_source('y.py', 0, 100),
                _make_source('a.py', 0, 100),
            ],
        }
    ]
    ground_truth = [
        {
            'question_id': 'q1',
            'sources': [_make_source('a.py', 0, 100)],
        }
    ]
    assert calculate_recall_at_k(search_results, ground_truth, k=1) == 0.0
