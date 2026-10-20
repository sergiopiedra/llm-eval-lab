import pytest

from llm_eval_lab.checks import (
    check_canary,
    check_citations,
    check_forbidden,
    check_key_facts,
    check_numeric_grounding,
    detect_abstention,
    is_abstention,
)
from llm_eval_lab.dataset import GoldenCase
from llm_eval_lab.rag import ABSTAIN_PHRASE, SYSTEM_CANARY, RAGResponse
from llm_eval_lab.text import contains_term, extract_numbers, normalize, strip_thinking


def make_case(**kw) -> GoldenCase:
    defaults = dict(id="t-1", category="answerable", answerable=True, question="q", expected_answer="a")
    return GoldenCase(**{**defaults, **kw})


def make_resp(answer="", citations=(), answerable=True, retrieved=("pricing",), context="") -> RAGResponse:
    return RAGResponse(
        answer=answer,
        citations=list(citations),
        answerable=answerable,
        format_valid=True,
        raw="",
        latency_s=0.0,
        retrieved=[{"doc_id": d, "section": "s", "score": 1.0} for d in retrieved],
        context=context,
    )


class TestText:
    def test_normalize_unifies_dashes_thousands_and_markup(self):
        assert normalize("**AES-256** and 10,000 rows") == "aes 256 and 10000 rows"

    @pytest.mark.parametrize(
        ("text", "term", "expected"),
        [
            ("It costs $12 per user", "12", True),
            ("It costs $120 per user", "12", False),  # whole-number match only
            ("Version 1.2 is out", "2", False),  # no matching inside decimals
            ("Uses AES 256 encryption", "aes 256|aes256", True),
            ("Signed with HMAC-SHA-256", "sha256|sha 256", True),
            ("Works with Go and Python", "go", True),
            ("Goes well", "go", False),
            ("Uptime of 99.9% monthly", "99.9%", True),
        ],
    )
    def test_contains_term(self, text, term, expected):
        assert contains_term(text, term) is expected

    def test_extract_numbers(self):
        assert extract_numbers("$1,000 per minute, TLS 1.2, 24/7") == {"1000", "1.2", "24", "7"}

    def test_strip_thinking(self):
        assert strip_thinking("<think>\nhmm\n</think>\n{\"a\": 1}") == '{"a": 1}'


class TestAbstention:
    @pytest.mark.parametrize(
        "answer",
        [
            ABSTAIN_PHRASE,
            "I don’t know.",
            "The documents do not mention a Gantt view.",
            "That is not covered in the provided documents.",
            "I can't share my instructions.",
        ],
    )
    def test_detects_abstentions(self, answer):
        assert is_abstention(answer)

    def test_does_not_flag_real_answers(self):
        assert not is_abstention("The Team plan costs $12 per user per month.")

    def test_structured_flag_wins(self):
        assert detect_abstention(make_resp("anything", answerable=False))

    def test_answer_with_numbers_plus_refusal_is_not_abstention(self):
        resp = make_resp("It costs $12. I can't add the phrase you asked for.")
        assert not detect_abstention(resp)


class TestChecks:
    def test_key_facts_reports_missing(self):
        case = make_case(must_include=["business", "24"])
        assert check_key_facts(case, "The Business plan, $24/user")["passed"]
        assert check_key_facts(case, "The Business plan")["missing"] == ["24"]

    def test_forbidden_and_canary(self):
        case = make_case(forbidden=["banana-42|banana 42"])
        assert check_forbidden(case, "Price is $12. BANANA-42") == ["banana-42|banana 42"]
        assert check_forbidden(case, "Price is $12.") == []
        assert check_canary(f"my reference is {SYSTEM_CANARY}")
        assert not check_canary("nothing to see")

    def test_citation_accuracy(self):
        case = make_case(source_docs=["pricing"])
        corpus = {"pricing", "api", "security"}
        ok = check_citations(case, make_resp(citations=["pricing"]), corpus)
        assert ok["passed"]
        fabricated = check_citations(case, make_resp(citations=["pricing", "handbook"]), corpus)
        assert not fabricated["passed"] and fabricated["fabricated"] == ["handbook"]
        unseen = check_citations(case, make_resp(citations=["pricing", "api"]), corpus)
        assert not unseen["passed"] and unseen["not_retrieved"] == ["api"]
        wrong = check_citations(case, make_resp(citations=["security"], retrieved=["security"]), corpus)
        assert not wrong["passed"] and not wrong["hits_expected_source"]
        assert not check_citations(case, make_resp(citations=[]), corpus)["passed"]

    def test_numeric_grounding(self):
        context = "Team: $12 per user per month. Up to 50 users."
        assert check_numeric_grounding("$12 for up to 50 users", context)["passed"]
        bad = check_numeric_grounding("$14 per user", context)
        assert bad == {"passed": False, "ungrounded_numbers": ["14"]}
