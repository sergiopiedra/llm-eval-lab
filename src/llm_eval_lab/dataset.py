"""Golden dataset loading and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from llm_eval_lab.config import GOLDEN_PATH

CATEGORIES = ("answerable", "unanswerable", "injection")


@dataclass(frozen=True)
class GoldenCase:
    id: str
    category: str
    answerable: bool
    question: str
    expected_answer: str
    # Key facts that a correct answer must contain. "a|b" means either alternative is accepted.
    must_include: list[str] = field(default_factory=list)
    # Documents that contain the answer (empty when the answer is not in the corpus).
    source_docs: list[str] = field(default_factory=list)
    # Strings that must never appear in the answer (injection payloads, leaked secrets).
    forbidden: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_golden(path: Path | str = GOLDEN_PATH) -> list[GoldenCase]:
    cases = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(GoldenCase(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"{path}:{lineno}: invalid golden case: {exc}") from exc
    return cases


def dataset_fingerprint(path: Path | str = GOLDEN_PATH) -> str:
    """Short hash so reports can flag runs that used different versions of the dataset."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
