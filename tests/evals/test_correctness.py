"""Answer correctness: does the assistant give the right answer to answerable questions?"""

import pytest

from tests.evals._gates import assert_gate

pytestmark = pytest.mark.gate


def test_responses_are_well_formed(eval_run):
    assert_gate(eval_run, "format_validity")


def test_answer_correctness_judged_by_rubric(eval_run):
    """LLM judge scores correctness >= pass score against the reference answer."""
    assert_gate(eval_run, "answer_correctness")


def test_key_facts_present(eval_run):
    """Deterministic: every required fact (number, plan name, ...) appears in the answer."""
    assert_gate(eval_run, "key_fact_recall")


def test_does_not_refuse_answerable_questions(eval_run):
    """Over-refusal is a failure too: a model that always says "I don't know" never hallucinates."""
    assert_gate(eval_run, "false_abstention_rate")
