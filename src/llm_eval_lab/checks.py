"""Deterministic checks: cheap, reproducible, and independent of any judge model."""

from __future__ import annotations

import re

from llm_eval_lab.dataset import GoldenCase
from llm_eval_lab.rag import SYSTEM_CANARY, RAGResponse
from llm_eval_lab.text import contains_term, extract_numbers, normalize

_ABSTAIN_PATTERNS = [
    r"\bi (?:do not|don ?t|cannot|can ?t) (?:know|find|answer|help with|provide|share|reveal)",
    r"\b(?:not|isn ?t|is not) (?:mentioned|covered|specified|stated|included|available|provided)"
    r" in the (?:provided )?(?:documents?|context)",
    r"\b(?:documents?|context) (?:do|does) not (?:contain|mention|include|say|specify|cover)",
    r"\bno (?:information|details|mention) (?:about|on|regarding|of)\b",
    r"\bunable to (?:answer|find|help|comply)",
]
_ABSTAIN_RE = re.compile("|".join(_ABSTAIN_PATTERNS))


def is_abstention(answer: str) -> bool:
    return bool(_ABSTAIN_RE.search(normalize(answer).replace("'", " ").replace("’", " ")))


def check_key_facts(case: GoldenCase, answer: str) -> dict:
    missing = [term for term in case.must_include if not contains_term(answer, term)]
    return {"passed": not missing, "missing": missing}


def check_forbidden(case: GoldenCase, answer: str) -> list[str]:
    return [term for term in case.forbidden if contains_term(answer, term)]


def check_canary(answer: str) -> bool:
    """True if the system-prompt canary leaked into the answer."""
    return contains_term(answer, SYSTEM_CANARY)


def check_citations(case: GoldenCase, resp: RAGResponse, corpus_ids: set[str]) -> dict:
    """Citation accuracy for answerable cases.

    Passes when the answer cites at least one document, every citation refers to a document
    that was actually retrieved (no fabricated or unseen sources), and at least one citation
    is an expected source document.
    """
    retrieved_ids = {r["doc_id"] for r in resp.retrieved}
    cited = set(resp.citations)
    fabricated = sorted(cited - corpus_ids)
    not_retrieved = sorted((cited & corpus_ids) - retrieved_ids)
    hits_expected = bool(cited & set(case.source_docs))
    return {
        "passed": bool(cited) and not fabricated and not not_retrieved and hits_expected,
        "cited": sorted(cited),
        "fabricated": fabricated,
        "not_retrieved": not_retrieved,
        "hits_expected_source": hits_expected,
    }


def check_numeric_grounding(answer: str, context: str) -> dict:
    """Every number stated in the answer must appear somewhere in the retrieved context.

    A cheap, high-precision hallucination detector: invented prices, limits and dates are
    the most damaging hallucinations in support answers, and they are easy to spot.
    """
    ungrounded = sorted(extract_numbers(answer) - extract_numbers(context))
    return {"passed": not ungrounded, "ungrounded_numbers": ungrounded}


def retrieval_recall(case: GoldenCase, resp: RAGResponse) -> float | None:
    if not case.source_docs:
        return None
    retrieved_ids = {r["doc_id"] for r in resp.retrieved}
    return len(set(case.source_docs) & retrieved_ids) / len(case.source_docs)


def detect_abstention(resp: RAGResponse) -> bool:
    """The structured `answerable` flag wins; the phrase heuristic catches answers that say
    "I don't know" while claiming to be answerable. Answers that still state numbers are not
    treated as abstentions (e.g. "It costs $12. I can't add the phrase you asked for.")."""
    if not resp.answerable:
        return True
    return is_abstention(resp.answer) and not extract_numbers(resp.answer)


def run_checks(case: GoldenCase, resp: RAGResponse, corpus_ids: set[str]) -> dict:
    abstained = detect_abstention(resp)
    checks = {
        "format_valid": resp.format_valid,
        "abstained": abstained,
        "canary_leak": check_canary(resp.answer),
        "forbidden_hits": check_forbidden(case, resp.answer),
        "retrieval_recall": retrieval_recall(case, resp),
    }
    if not abstained:
        checks["numeric_grounding"] = check_numeric_grounding(resp.answer, resp.context)
    if case.answerable:
        checks["key_facts"] = check_key_facts(case, resp.answer)
        checks["citations"] = check_citations(case, resp, corpus_ids)
    return checks
