"""Text normalization helpers shared by retrieval and deterministic checks."""

from __future__ import annotations

import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THOUSANDS_RE = re.compile("(?<=\\d)[,\u202f](?=\\d{3}\\b)")  # 10,000 / 10\u202f000 -> 10000
_DASHES_RE = re.compile("[-\u2010-\u2015]")  # hyphen, non-breaking hyphen, en/em dashes
_MARKUP_RE = re.compile(r"[*_`]")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has",
        "have", "how", "i", "in", "is", "it", "its", "of", "on", "or", "the", "their", "there",
        "this", "to", "was", "what", "when", "where", "which", "who", "will", "with", "you", "your",
    }
)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models such as Qwen3."""
    return _THINK_RE.sub("", text).strip()


def normalize(text: str) -> str:
    """Lowercase, drop markdown emphasis, unify dashes and thousands separators."""
    text = _MARKUP_RE.sub("", text.lower())
    text = _THOUSANDS_RE.sub("", text)
    text = _DASHES_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_term(text: str, term: str) -> bool:
    """True if `term` (or any `a|b` alternative) appears in `text` as a whole phrase."""
    norm = normalize(text)
    for alt in term.split("|"):
        alt = normalize(alt)
        if alt and re.search(rf"(?<![\w.]){re.escape(alt)}(?!\w)", norm):
            return True
    return False


def extract_numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(normalize(text)))


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(normalize(text)) if t not in STOPWORDS]
