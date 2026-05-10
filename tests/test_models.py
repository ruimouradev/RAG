"""Tests for Pydantic model validation and serialisation."""

import json

import pytest
from pydantic import ValidationError

from src.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)


def test_minimal_source_requires_all_fields() -> None:
    """MinimalSource should reject creation when any field is missing."""
    with pytest.raises(ValidationError):
        MinimalSource.model_validate({"file_path": "src/abc.py"})


def test_minimal_source_rejects_non_integer_character_index() -> None:
    """MinimalSource should raise ValidationError for non-integer indices."""
    with pytest.raises(ValidationError):
        MinimalSource.model_validate({
            "file_path": "src/abc.py",
            "first_character_index": "not_an_int",
            "last_character_index": 100,
        })


def test_unanswered_question_fields() -> None:
    """UnansweredQuestion should store question_id and question correctly."""
    q = UnansweredQuestion(question="What is vLLM?")
    assert q.question == "What is vLLM?"
    assert isinstance(q.question_id, str)
    assert len(q.question_id) > 0


def test_answered_question_includes_sources() -> None:
    """AnsweredQuestion should accept a list of MinimalSource objects."""
    source = MinimalSource(
        file_path="src/abc.py",
        first_character_index=0,
        last_character_index=100,
    )
    q = AnsweredQuestion(
        question="What is vLLM?",
        sources=[source],
        answer="vLLM is a fast LLM serving library.",
    )
    assert len(q.sources) == 1
    assert q.sources[0].file_path == "src/abc.py"


def test_search_results_serialises_to_valid_json() -> None:
    """Search results should serialise to JSON in the expected format."""
    source = MinimalSource(
        file_path="src/abc.py",
        first_character_index=0,
        last_character_index=100,
    )
    result = MinimalSearchResults(
        question_id="q1",
        question_str="What is vLLM?",
        retrieved_sources=[source],
    )
    output = StudentSearchResults(search_results=[result], k=10)
    data = json.loads(output.model_dump_json())
    assert "search_results" in data
    assert data["k"] == 10
    assert data["search_results"][0]["question_id"] == "q1"


def test_search_results_with_answer_serialises_to_valid_json() -> None:
    """Search results with answers should serialise to JSON correctly."""
    source = MinimalSource(
        file_path="src/abc.py",
        first_character_index=0,
        last_character_index=100,
    )
    result = MinimalAnswer(
        question_id="q1",
        question_str="What is vLLM?",
        retrieved_sources=[source],
        answer="vLLM is a fast LLM serving library.",
    )
    output = StudentSearchResultsAndAnswer(search_results=[result], k=10)
    data = json.loads(output.model_dump_json())
    assert "search_results" in data
    expected = "vLLM is a fast LLM serving library."
    assert data["search_results"][0]["answer"] == expected
