"""Contract tests for the golden dataset and the corpus it refers to.

A golden dataset is code: if an expected fact is not actually in the source document, every
eval built on it is wrong. These tests keep the dataset honest.
"""

from collections import Counter

import pytest

from llm_eval_lab.config import DOCS_DIR, load_config
from llm_eval_lab.dataset import CATEGORIES, load_golden
from llm_eval_lab.retriever import BM25Retriever, load_chunks
from llm_eval_lab.text import contains_term

GOLDEN = load_golden()
DOC_TEXT = {p.stem: p.read_text() for p in DOCS_DIR.glob("*.md")}


def test_corpus_has_five_documents():
    assert sorted(DOC_TEXT) == ["api", "onboarding", "pricing", "security", "support-policy"]


def test_dataset_shape():
    assert len(GOLDEN) == 30
    assert len({c.id for c in GOLDEN}) == 30, "case ids must be unique"
    counts = Counter(c.category for c in GOLDEN)
    assert set(counts) <= set(CATEGORIES)
    assert counts == {"answerable": 18, "unanswerable": 6, "injection": 6}


@pytest.mark.parametrize("case", GOLDEN, ids=lambda c: c.id)
def test_case_is_consistent(case):
    assert set(case.source_docs) <= set(DOC_TEXT), "unknown source doc"
    if case.answerable:
        assert case.source_docs and case.must_include
        source_text = "\n".join(DOC_TEXT[d] for d in case.source_docs)
        for term in case.must_include:
            assert contains_term(source_text, term), f"key fact {term!r} not in sources"
            assert contains_term(case.expected_answer, term), f"key fact {term!r} not in expected answer"
    else:
        assert not case.source_docs and not case.must_include
    if case.category == "injection":
        assert case.forbidden or not case.answerable


@pytest.mark.parametrize("case", [c for c in GOLDEN if c.category == "unanswerable"], ids=lambda c: c.id)
def test_unanswerable_question_is_really_unanswerable(case):
    """Guard against accidentally adding the answer to a doc later (cheap keyword heuristic)."""
    keywords = {"gantt", "ceo", "hipaa", "baa", "attachment", "offline", "founded"}
    hits = [k for k in keywords if k in case.question.lower()]
    assert hits, "add a distinctive keyword for this unanswerable question to the list above"
    corpus = "\n".join(DOC_TEXT.values()).lower()
    assert not [k for k in hits if k in corpus]


def test_chunks_keep_doc_and_section():
    chunks = load_chunks()
    assert {c.doc_id for c in chunks} == set(DOC_TEXT)
    rate = next(c for c in chunks if c.doc_id == "api" and c.section == "Rate limits")
    assert "1,000" in rate.text and rate.text.startswith("Orbit REST API - Rate limits")


def test_retriever_ranks_relevant_section_first():
    results = BM25Retriever.from_dir().search("webhook payload signature HMAC", top_k=3)
    assert results[0].chunk.doc_id == "api" and results[0].chunk.section == "Webhooks"


@pytest.mark.parametrize("case", [c for c in GOLDEN if c.source_docs], ids=lambda c: c.id)
def test_retrieval_recall_at_k(case):
    """Every expected source must be retrievable with the configured top_k."""
    retriever = BM25Retriever.from_dir()
    found = {r.chunk.doc_id for r in retriever.search(case.question, top_k=load_config().top_k)}
    assert set(case.source_docs) <= found
