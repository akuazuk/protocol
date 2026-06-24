"""Path lex shards: inverted index token→chunk_id per rubric (offline build)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from clinical_knowledge.lazy_rag_config import lex_shards_dir

_ICD_RE = re.compile(r"\b[A-Z]\d{2}(?:\.\d{1,2})?\b", re.IGNORECASE)


def _tokenize_ru(s: str) -> list[str]:
    s = s.lower().replace("ё", "е")
    return [t for t in re.findall(r"[а-яa-z]{2,}", s) if len(t) >= 2]


class PathLexIndex:
    """Lazy-loaded rubric shards for prefilter before scoring."""

    def __init__(self, shards_dir: Path) -> None:
        self.shards_dir = shards_dir
        self._loaded: dict[str, dict[str, list[str]]] = {}

    @classmethod
    def from_env(cls) -> PathLexIndex | None:
        d = lex_shards_dir()
        if not d.is_dir():
            return None
        if not any(d.glob("*.json")):
            return None
        return cls(d)

    def _load_rubric(self, rubric: str) -> dict[str, list[str]]:
        rub = str(rubric or "").strip()
        if rub in self._loaded:
            return self._loaded[rub]
        path = self.shards_dir / f"{rub}.json"
        if not path.is_file():
            self._loaded[rub] = {}
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            self._loaded[rub] = {}
            return {}
        index = data.get("index") if isinstance(data, dict) else {}
        if not isinstance(index, dict):
            index = {}
        self._loaded[rub] = {str(k): list(v) for k, v in index.items() if isinstance(v, list)}
        return self._loaded[rub]

    def query_chunk_ids(
        self,
        query: str,
        *,
        rubrics: list[str] | None = None,
        chunk_id_allowlist: set[str] | None = None,
        max_ids: int = 400,
    ) -> set[str]:
        tokens = set(_tokenize_ru(query))
        for m in _ICD_RE.findall(query or ""):
            tokens.add(m.upper())
        if not tokens:
            return set()
        rub_list = rubrics or []
        if not rub_list:
            rub_list = [p.stem for p in self.shards_dir.glob("*.json")]
        out: set[str] = set()
        for rub in rub_list:
            idx = self._load_rubric(rub)
            for t in tokens:
                for cid in idx.get(t, ()):
                    if chunk_id_allowlist is not None and cid not in chunk_id_allowlist:
                        continue
                    out.add(cid)
                    if len(out) >= max_ids:
                        return out
        return out

    def stats(self) -> dict[str, Any]:
        shards = list(self.shards_dir.glob("*.json")) if self.shards_dir.is_dir() else []
        return {
            "shards_dir": str(self.shards_dir),
            "shard_files": len(shards),
            "loaded_rubrics": len(self._loaded),
        }
