"""Groundedness and citation accuracy: is every answer supported by the retrieved sources?"""

import pytest

from tests.evals._gates import assert_gate

pytestmark = pytest.mark.gate


def test_retrieval_finds_expected_sources(eval_run):
    """If retrieval misses, the generator cannot be grounded; measured separately to localize failures."""
    assert_gate(eval_run, "retrieval_recall")


def test_answers_grounded_in_context(eval_run):
    """LLM judge: every claim in the answer is supported by the retrieved context."""
    assert_gate(eval_run, "groundedness")


def test_numbers_grounded_in_context(eval_run):
    """Deterministic: every number in an answer appears in the retrieved context."""
    assert_gate(eval_run, "numeric_grounding")


def test_citations_are_accurate(eval_run):
    """Deterministic: cites a retrieved expected source and never a fabricated document."""
    assert_gate(eval_run, "citation_accuracy")
