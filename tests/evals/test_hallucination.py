"""Hallucination detection: questions whose answer is NOT in the documents must be declined."""

import pytest

from tests.evals._gates import assert_gate, ids


@pytest.mark.gate
def test_declines_unanswerable_questions(eval_run):
    assert_gate(eval_run, "hallucination_abstention")


@pytest.mark.invariant
@pytest.mark.parametrize("case_id", ids(category="unanswerable"))
def test_unanswerable_answers_contain_no_invented_numbers(cases_by_id, case_id):
    """Hard requirement: if the model does answer, it must not invent figures (dates, prices...)."""
    case = cases_by_id[case_id]
    grounding = case["checks"].get("numeric_grounding")
    if grounding is None:  # the model abstained
        return
    assert grounding["passed"], (
        f"{case_id}: invented numbers {grounding['ungrounded_numbers']} in "
        f"answer {case['response']['answer']!r}"
    )
