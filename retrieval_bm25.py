"""
BM25 по корпусу чанков для гибрида с лексическим скором (rag_server.retrieve).
"""
from __future__ import annotations

import math
import os
from collections import Counter
from typing import Callable


class BM25Index:
    """Okapi BM25: документ = lex_text|text + title чанка."""

    def __init__(
        self,
        chunks: list[dict],
        tokenize: Callable[[str], list[str]],
    ) -> None:
        self._tokenize = tokenize
        self.N = len(chunks)
        self.df: dict[str, int] = {}
        self.k1 = float(os.environ.get("RAG_BM25_K1", "1.5"))
        self.b = float(os.environ.get("RAG_BM25_B", "0.75"))

        for ch in chunks:
            text = (ch.get("lex_text") or ch.get("text") or "") + " " + (
                ch.get("title") or ""
            )
            toks = self._tokenize(text)
            if not toks:
                continue
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1

        total_len = 0
        for ch in chunks:
            text = (ch.get("lex_text") or ch.get("text") or "") + " " + (
                ch.get("title") or ""
            )
            toks = self._tokenize(text)
            total_len += len(toks) if toks else 0
        self.avgdl = (total_len / self.N) if self.N else 0.0

    def score_doc(self, query_terms: set[str], chunk: dict) -> float:
        if not query_terms or self.N == 0:
            return 0.0
        text = (chunk.get("lex_text") or chunk.get("text") or "") + " " + (
            chunk.get("title") or ""
        )
        toks = self._tokenize(text)
        if not toks:
            return 0.0
        tf = Counter(toks)
        dl = len(toks)
        avgdl = self.avgdl if self.avgdl > 0 else 1.0
        s = 0.0
        for t in query_terms:
            if t not in tf:
                continue
            df = self.df.get(t, 0)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            f = tf[t]
            denom = f + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
            s += idf * ((f * (self.k1 + 1.0)) / denom)
        return float(s)


def build_bm25_index(
    chunks: list[dict],
    tokenize: Callable[[str], list[str]],
) -> BM25Index | None:
    if not chunks:
        return None
    return BM25Index(chunks, tokenize)
