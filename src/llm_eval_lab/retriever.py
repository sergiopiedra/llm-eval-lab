"""A tiny, dependency-free BM25 retriever over markdown sections.

Each document is split on `## ` headings; every section becomes a chunk that keeps its
document id and heading so answers can cite where facts came from.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from llm_eval_lab.config import DOCS_DIR
from llm_eval_lab.text import tokenize


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    section: str
    text: str


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    score: float


def load_chunks(docs_dir: Path | str = DOCS_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        doc_id = path.stem
        content = path.read_text()
        title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else doc_id
        parts = re.split(r"^## ", content, flags=re.MULTILINE)
        intro = re.sub(r"^# .+$", "", parts[0], flags=re.MULTILINE).strip()
        if intro:
            chunks.append(Chunk(doc_id, title, intro))
        for part in parts[1:]:
            heading, _, body = part.partition("\n")
            chunks.append(Chunk(doc_id, heading.strip(), f"{title} - {heading.strip()}\n{body.strip()}"))
    return chunks


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tfs = [Counter(tokenize(c.text)) for c in chunks]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        self._avg_len = sum(self._lens) / max(len(self._lens), 1)
        df = Counter(term for tf in self._tfs for term in tf)
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    @classmethod
    def from_dir(cls, docs_dir: Path | str = DOCS_DIR) -> BM25Retriever:
        return cls(load_chunks(docs_dir))

    @property
    def doc_ids(self) -> set[str]:
        return {c.doc_id for c in self.chunks}

    def search(self, query: str, top_k: int = 4) -> list[Retrieved]:
        terms = tokenize(query)
        scored = []
        for chunk, tf, length in zip(self.chunks, self._tfs, self._lens, strict=True):
            score = 0.0
            for term in terms:
                freq = tf.get(term)
                if freq:
                    norm = self.k1 * (1 - self.b + self.b * length / self._avg_len)
                    score += self._idf[term] * freq * (self.k1 + 1) / (freq + norm)
            if score > 0:
                scored.append(Retrieved(chunk, round(score, 4)))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
