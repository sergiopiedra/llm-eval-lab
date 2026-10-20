"""Who judges the judge? Agreement between the LLM judge and human labels.

If the judge disagrees with humans on this small labelled set, its scores on the golden
dataset cannot be trusted either, so this test fails before the gates are believed.
"""

import json

import pytest

from llm_eval_lab.config import CALIBRATION_PATH
from llm_eval_lab.judge import Judge
from llm_eval_lab.llm import make_client


@pytest.fixture(scope="module")
def calibration_results(eval_config, golden):
    judge = Judge(make_client(eval_config.backend, eval_config.judge_model, eval_config, golden))
    items = [json.loads(line) for line in CALIBRATION_PATH.read_text().splitlines() if line.strip()]
    results = []
    for item in items:
        v = judge.grade(item["question"], item["reference"], item["context"], item["candidate"])
        results.append((item, v))
    return results


@pytest.mark.gate
@pytest.mark.parametrize("dimension", ["correct", "grounded"])
def test_judge_agrees_with_human_labels(calibration_results, eval_config, dimension):
    score_field = "correctness" if dimension == "correct" else "groundedness"
    agree, disagreements = 0, []
    for item, verdict in calibration_results:
        score = getattr(verdict, score_field)
        judged = score is not None and score >= eval_config.judge_pass_score
        if judged == item[f"human_{dimension}"]:
            agree += 1
        else:
            disagreements.append(f"{item['id']} (judge={score}, human={item[f'human_{dimension}']})")
    agreement = agree / len(calibration_results)
    assert all(getattr(v, score_field) is not None for _, v in calibration_results), "judge failed to score"
    if eval_config.backend == "mock":
        pytest.skip(
            f"mock judge is a heuristic (agreement {agreement:.0%}); the calibration threshold "
            "applies to real judge models: run with --backend ollama"
        )
    assert agreement >= eval_config.min_calibration_agreement, (
        f"judge/human agreement on '{dimension}' is {agreement:.0%} "
        f"(< {eval_config.min_calibration_agreement:.0%}); disagreements: {', '.join(disagreements)}"
    )
