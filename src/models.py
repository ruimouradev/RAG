"""
Pydantic models for the RAG pipeline.

Defines the data structures for sources, questions, search results,
and answers. The inheritance chain mirrors the subject specification.
"""

import uuid
from typing import List, Sequence

# BaseModel for typed/validated dataclasses, Field for default_factory on UUIDs
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A chunk location inside a file.

    Attributes:
        file_path: Path to the source file.
        first_character_index: Start position of the chunk in the file.
        last_character_index: End position of the chunk in the file.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question waiting to be answered.

    Attributes:
        question_id: Unique identifier, auto-generated via UUID4.
        question: The question text.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """An answered question with its supporting sources.

    Attributes:
        sources: List of source chunks that support the answer.
        answer: The generated answer text.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of questions, answered or unanswered.

    Attributes:
        rag_questions: List of questions, with or without answers.
    """

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The top-k sources retrieved for a single question.

    Attributes:
        question_id: ID of the question being answered.
        question_str: The question text.
        retrieved_sources: Ranked list of retrieved source chunks.
    """

    question_id: str
    question_str: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results enriched with a generated answer.

    Attributes:
        answer: The generated answer text.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Full output of a search operation over a dataset.

    Attributes:
        search_results: List of per-question search results.
        k: Number of sources retrieved per question.
    """

    search_results: Sequence[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Full output of an answer operation.

    Attributes:
        search_results: List of per-question results with answers included.
    """

    search_results: Sequence[MinimalAnswer]
