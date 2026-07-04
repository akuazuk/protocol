"""Векторный индекс для precomputed embeddings чанков (FAISS или numpy fallback)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

_index_vectors: np.ndarray | None = None
_index_chunk_indices: list[int] | None = None
_index_dim: int = 0
_faiss_index: Any = None


def vector_index_enabled() -> bool:
    return os.environ.get("RAG_VECTOR_INDEX", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def default_index_path() -> Path:
    raw = (os.environ.get("RAG_VECTOR_INDEX_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    data_root = (os.environ.get("RAG_CHUNKS_DIR") or "").strip()
    base = Path(data_root).expanduser().resolve() if data_root else ROOT
    return base / "corpus_vector_index"


def _chunk_has_embedding(ch: dict) -> bool:
    emb = ch.get("embedding")
    return isinstance(emb, list) and len(emb) >= 8 and isinstance(emb[0], (int, float))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return matrix / norms


def build_index_from_chunks(chunks: list[dict]) -> dict[str, Any]:
    """Построить индекс из чанков с полем embedding. Возвращает статистику."""
    global _index_vectors, _index_chunk_indices, _index_dim, _faiss_index
    rows: list[list[float]] = []
    idx_map: list[int] = []
    for i, ch in enumerate(chunks):
        if not _chunk_has_embedding(ch):
            continue
        rows.append([float(x) for x in ch["embedding"]])
        idx_map.append(i)
    if not rows:
        _index_vectors = None
        _index_chunk_indices = None
        _faiss_index = None
        return {"ok": False, "reason": "no_embeddings", "indexed": 0}

    matrix = np.asarray(rows, dtype=np.float32)
    _index_dim = int(matrix.shape[1])
    matrix = _normalize_rows(matrix)
    _index_vectors = matrix
    _index_chunk_indices = idx_map

    try:
        import faiss  # type: ignore

        _faiss_index = faiss.IndexFlatIP(_index_dim)
        _faiss_index.add(matrix)
        backend = "faiss"
    except ImportError:
        _faiss_index = None
        backend = "numpy"

    return {
        "ok": True,
        "indexed": len(idx_map),
        "dim": _index_dim,
        "backend": backend,
    }


def save_index(index_dir: Path, chunks: list[dict], *, model: str = "") -> dict[str, Any]:
    """Сохранить индекс на диск (vectors.npy + meta.json)."""
    stats = build_index_from_chunks(chunks)
    if not stats.get("ok"):
        return stats
    index_dir.mkdir(parents=True, exist_ok=True)
    assert _index_vectors is not None
    assert _index_chunk_indices is not None
    np.save(str(index_dir / "vectors.npy"), _index_vectors)
    meta = {
        "chunk_indices": _index_chunk_indices,
        "dim": _index_dim,
        "count": len(_index_chunk_indices),
        "embedding_model": model or os.environ.get("GEMINI_EMBEDDING_MODEL", ""),
        "backend": stats.get("backend"),
    }
    (index_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stats["path"] = str(index_dir)
    return stats


def load_index(index_dir: Path) -> dict[str, Any]:
    """Загрузить индекс с диска в память."""
    global _index_vectors, _index_chunk_indices, _index_dim, _faiss_index
    meta_path = index_dir / "meta.json"
    vec_path = index_dir / "vectors.npy"
    if not meta_path.is_file() or not vec_path.is_file():
        return {"ok": False, "reason": "missing_files"}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    matrix = np.load(str(vec_path))
    _index_vectors = matrix.astype(np.float32, copy=False)
    _index_chunk_indices = [int(x) for x in (meta.get("chunk_indices") or [])]
    _index_dim = int(meta.get("dim") or matrix.shape[1])
    _faiss_index = None
    backend = "numpy"
    try:
        import faiss  # type: ignore

        _faiss_index = faiss.IndexFlatIP(_index_dim)
        _faiss_index.add(_index_vectors)
        backend = "faiss"
    except ImportError:
        pass
    return {
        "ok": True,
        "indexed": len(_index_chunk_indices),
        "dim": _index_dim,
        "backend": backend,
        "path": str(index_dir),
    }


def load_index_from_env(chunks: list[dict] | None = None) -> dict[str, Any]:
    """Загрузить с диска или построить из chunks при старте."""
    if not vector_index_enabled():
        return {"ok": False, "reason": "disabled"}
    index_dir = default_index_path()
    if (index_dir / "meta.json").is_file():
        return load_index(index_dir)
    if chunks:
        return build_index_from_chunks(chunks)
    return {"ok": False, "reason": "not_built"}


def _normalize_query_vec(query_vec: list[float]) -> np.ndarray | None:
    q = np.asarray([float(x) for x in query_vec], dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        return None
    return q / norm


def _search_local_ids(q: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Top-K локальных позиций индекса с cosine score."""
    if _index_vectors is None or not _index_chunk_indices:
        return []
    k = min(k, len(_index_chunk_indices))
    if _faiss_index is not None:
        scores, ids = _faiss_index.search(q.reshape(1, -1), k)
        out: list[tuple[int, float]] = []
        for local_i, score in zip(ids[0], scores[0]):
            if int(local_i) >= 0:
                out.append((int(local_i), float(score)))
        return out
    scores = _index_vectors @ q
    hit_ids = np.argsort(-scores)[:k]
    return [(int(local_i), float(scores[local_i])) for local_i in hit_ids]


def search_with_scores(
    query_vec: list[float],
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Top-K глобальных индексов чанков с cosine score."""
    q = _normalize_query_vec(query_vec)
    if q is None or _index_chunk_indices is None:
        return []
    k = top_k or int(os.environ.get("RAG_VECTOR_TOP_K", "200"))
    hits: list[tuple[int, float]] = []
    for local_i, score in _search_local_ids(q, k):
        if 0 <= local_i < len(_index_chunk_indices):
            hits.append((_index_chunk_indices[local_i], score))
    return hits


def search_scoped_with_scores(
    query_vec: list[float],
    allowed_global_indices: set[int],
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Top-K чанков только из allowed_global_indices с cosine score."""
    if not allowed_global_indices:
        return []
    q = _normalize_query_vec(query_vec)
    if q is None or _index_vectors is None or not _index_chunk_indices:
        return []
    allowed_local: list[int] = []
    for local_i, global_i in enumerate(_index_chunk_indices):
        if global_i in allowed_global_indices:
            allowed_local.append(local_i)
    if not allowed_local:
        return []
    k = top_k or int(os.environ.get("PROTOCOL_SEMANTIC_TOP_K", "24"))
    k = min(k, len(allowed_local))
    local_matrix = _index_vectors[allowed_local]
    scores = local_matrix @ q
    order = np.argsort(-scores)[:k]
    out: list[tuple[int, float]] = []
    for pos in order:
        local_i = allowed_local[int(pos)]
        out.append((_index_chunk_indices[local_i], float(scores[pos])))
    return out


def search(query_vec: list[float], top_k: int | None = None) -> set[int]:
    """Top-K глобальных индексов чанков по cosine (IP на нормализованных векторах)."""
    return {idx for idx, _ in search_with_scores(query_vec, top_k=top_k)}


def index_stats() -> dict[str, Any]:
    return {
        "enabled": vector_index_enabled(),
        "loaded": _index_chunk_indices is not None,
        "indexed": len(_index_chunk_indices or []),
        "dim": _index_dim,
        "path": str(default_index_path()),
    }
