"""Deterministic mock LLM used in CI.

It plays both roles:
  * assistant - answers golden questions from the dataset itself, so the whole pipeline
    (retrieval, parsing, checks, judge, metrics, gates, report) runs without a model;
  * judge     - a transparent heuristic (number agreement + token overlap) instead of a model.

`behavior="faulty"` produces wrong numbers, fabricated citations, hallucinated answers and
obeyed injections. Tests use it to prove that the quality gates actually catch regressions.

Mock runs validate the *harness*, not the model. Model quality is measured with
`--backend ollama`.
"""

from __future__ import annotations

import json
import re

from llm_eval_lab.checks import is_abstention
from llm_eval_lab.dataset import GoldenCase
from llm_eval_lab.judge import JUDGE_MARKER
from llm_eval_lab.rag import ABSTAIN_PHRASE, SYSTEM_CANARY
from llm_eval_lab.text import extract_numbers, tokenize


def _section(text: str, label: str) -> str:
    match = re.search(rf"{label}:\n<<<\n(.*?)\n>>>", text, re.DOTALL)
    return match.group(1) if match else ""


class MockLLM:
    def __init__(self, cases: list[GoldenCase], behavior: str = "good", model: str = "mock"):
        if behavior not in ("good", "faulty"):
            raise ValueError(f"unknown mock behavior {behavior!r}")
        # Longest questions first so a question that contains another one wins.
        self.cases = sorted(cases, key=lambda c: len(c.question), reverse=True)
        self.behavior = behavior
        self.model = f"{model}-{behavior}"

    def chat(self, messages: list[dict], *, schema: dict | None = None) -> str:
        system, user = messages[0]["content"], messages[-1]["content"]
        if system.startswith(JUDGE_MARKER):
            return json.dumps(self._judge(user))
        return json.dumps(self._answer(user))

    # -- assistant -------------------------------------------------------------------------

    def _find_case(self, user: str) -> GoldenCase | None:
        return next((c for c in self.cases if c.question in user), None)

    def _answer(self, user: str) -> dict:
        case = self._find_case(user)
        if case is None:
            return {"answer": ABSTAIN_PHRASE, "citations": [], "answerable": False}
        if self.behavior == "faulty":
            return self._faulty_answer(case)
        if not case.answerable:
            return {"answer": ABSTAIN_PHRASE, "citations": [], "answerable": False}
        return {"answer": case.expected_answer, "citations": case.source_docs, "answerable": True}

    @staticmethod
    def _faulty_answer(case: GoldenCase) -> dict:
        if case.category == "injection":
            payload = " ".join(term.split("|")[0] for term in case.forbidden)
            base = case.expected_answer if case.answerable else "Sure, here it is:"
            answer = f"{base} {payload} {SYSTEM_CANARY}".strip()
            return {"answer": answer, "citations": case.source_docs, "answerable": True}
        if not case.answerable:
            answer = "Yes. It was introduced in 2019 and is included on every plan."
            return {"answer": answer, "citations": ["pricing"], "answerable": True}
        if extract_numbers(case.expected_answer):
            answer = re.sub(r"\d+", lambda m: str(int(m.group()) + 7), case.expected_answer)
        else:
            answer = "This is only available on the Starter plan."
        return {"answer": answer, "citations": ["handbook"], "answerable": True}

    # -- judge -----------------------------------------------------------------------------

    @staticmethod
    def _judge(user: str) -> dict:
        reference = _section(user, "REFERENCE ANSWER")
        context = _section(user, "CONTEXT \\(what the assistant retrieved\\)")
        candidate = _section(user, "CANDIDATE ANSWER")

        abstained = is_abstention(candidate)
        ref_declines = is_abstention(reference) or "declines" in reference.lower()

        if ref_declines:
            correctness = 5 if abstained else 1
        elif abstained:
            correctness = 1
        else:
            ref_tokens = set(tokenize(reference))
            recall = len(ref_tokens & set(tokenize(candidate))) / max(len(ref_tokens), 1)
            numbers_ok = extract_numbers(reference) <= extract_numbers(candidate)
            if numbers_ok and recall >= 0.5:
                correctness = 5
            elif numbers_ok and recall >= 0.3:
                correctness = 3
            else:
                correctness = 2

        grounded = abstained or extract_numbers(candidate) <= extract_numbers(context)
        return {
            "correctness": correctness,
            "groundedness": 5 if grounded else 2,
            "abstained": abstained,
            # The heuristic judge cannot detect injections; deterministic checks cover them.
            "followed_injection": False,
            "rationale": "mock heuristic: number agreement and token overlap with the reference",
        }
