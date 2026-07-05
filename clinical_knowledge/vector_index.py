"""Векторный индекс для precomputed embeddings чанков (FAISS или numpy/mmap fallback)."""
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
_index_mmap: bool = False
_index_backend: str = ""
_chunk_id_to_global: dict[str, int] | None = None
_global_to_local: dict[int, int] | None = None
_sidecar_mtime: float = 0.0


def vector_index_enabled() -> bool:
    raw = os.environ.get("RAG_VECTOR_INDEX", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    idx = default_index_path()
    return (idx / "meta.json").is_file() and (idx / "vectors.npy").is_file()


def vector_mmap_enabled() -> bool:
    """mmap vectors.npy вместо полной загрузки в heap (Render Standard 2 GiB)."""
    raw = os.environ.get("RAG_VECTOR_MMAP", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return True


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


def _set_index_state(
    matrix: np.ndarray,
    idx_map: list[int],
    *,
    mmap: bool = False,
    backend: str = "",
) -> None:
    global _index_vectors, _index_chunk_indices, _index_dim, _faiss_index, _index_mmap, _index_backend
    global _chunk_id_to_global, _global_to_local
    _index_vectors = matrix
    _index_chunk_indices = idx_map
    _index_dim = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    _index_mmap = mmap
    _index_backend = backend
    _faiss_index = None
    _refresh_global_local_maps()
    if not mmap and _index_dim > 0:
        try:
            import faiss  # type: ignore

            _faiss_index = faiss.IndexFlatIP(_index_dim)
            _faiss_index.add(np.asarray(matrix, dtype=np.float32))
            _index_backend = backend or "faiss"
        except ImportError:
            _index_backend = backend or "numpy"


def build_index_from_chunks(chunks: list[dict]) -> dict[str, Any]:
    """Построить индекс из чанков с полем embedding. Возвращает статистику."""
    rows: list[list[float]] = []
    idx_map: list[int] = []
    for i, ch in enumerate(chunks):
        if not _chunk_has_embedding(ch):
            continue
        rows.append([float(x) for x in ch["embedding"]])
        idx_map.append(i)
    if not rows:
        _set_index_state(np.empty((0, 0), dtype=np.float32), [], mmap=False, backend="")
        return {"ok": False, "reason": "no_embeddings", "indexed": 0}

    matrix = _normalize_rows(np.asarray(rows, dtype=np.float32))
    backend = "numpy"
    _set_index_state(matrix, idx_map, mmap=False, backend=backend)
    if _faiss_index is not None:
        backend = "faiss"
        _index_backend = backend

    return {
        "ok": True,
        "indexed": len(idx_map),
        "dim": _index_dim,
        "backend": backend,
    }


def _refresh_global_local_maps() -> None:
    global _global_to_local
    if _index_chunk_indices is None:
        _global_to_local = None
        return
    _global_to_local = {int(g): i for i, g in enumerate(_index_chunk_indices)}


def _write_chunk_id_sidecar(index_dir: Path, chunks: list[dict]) -> int:
    """chunk_id → глобальный индекс корпуса (для lazy/manifest protocol semantic)."""
    assert _index_chunk_indices is not None
    mp: dict[str, int] = {}
    for global_i in _index_chunk_indices:
        if not (0 <= int(global_i) < len(chunks)):
            continue
        cid = chunks[int(global_i)].get("chunk_id")
        if cid:
            mp[str(cid)] = int(global_i)
    (index_dir / "chunk_id_global.json").write_text(
        json.dumps(mp, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return len(mp)


def _load_chunk_id_sidecar(index_dir: Path) -> int:
    global _chunk_id_to_global, _sidecar_mtime
    p = index_dir / "chunk_id_global.json"
    if not p.is_file():
        _chunk_id_to_global = {}
        _sidecar_mtime = 0.0
        return 0
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        _chunk_id_to_global = {}
        _sidecar_mtime = 0.0
        return 0
    _chunk_id_to_global = {str(k): int(v) for k, v in raw.items()}
    try:
        _sidecar_mtime = p.stat().st_mtime
    except OSError:
        _sidecar_mtime = 0.0
    return len(_chunk_id_to_global)


def _maybe_reload_chunk_id_sidecar() -> None:
    index_dir = default_index_path()
    p = index_dir / "chunk_id_global.json"
    if not p.is_file():
        return
    try:
        mt = p.stat().st_mtime
    except OSError:
        return
    if _chunk_id_to_global is None or mt != _sidecar_mtime:
        _load_chunk_id_sidecar(index_dir)


def global_index_for_chunk_id(chunk_id: str | None) -> int | None:
    _maybe_reload_chunk_id_sidecar()
    if not chunk_id or not _chunk_id_to_global:
        return None
    val = _chunk_id_to_global.get(str(chunk_id))
    return int(val) if val is not None else None


def cosine_for_global_index(global_i: int, query_vec: list[float]) -> float | None:
    """Cosine по строке vectors.npy (векторы уже L2-нормализованы при сборке индекса)."""
    ensure_index_loaded()
    if _index_vectors is None or _global_to_local is None:
        return None
    local_i = _global_to_local.get(int(global_i))
    if local_i is None:
        return None
    q = _normalize_query_vec(query_vec)
    if q is None:
        return None
    try:
        v = np.asarray(_index_vectors[int(local_i)], dtype=np.float32)
        return float(np.dot(q, v))
    except Exception:
        return None


def save_index(index_dir: Path, chunks: list[dict], *, model: str = "") -> dict[str, Any]:
    """Сохранить индекс на диск (vectors.npy + meta.json)."""
    stats = build_index_from_chunks(chunks)
    if not stats.get("ok"):
        return stats
    index_dir.mkdir(parents=True, exist_ok=True)
    assert _index_vectors is not None
    assert _index_chunk_indices is not None
    np.save(str(index_dir / "vectors.npy"), np.asarray(_index_vectors, dtype=np.float32))
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
    sidecar_n = _write_chunk_id_sidecar(index_dir, chunks)
    stats["chunk_id_map"] = sidecar_n
    stats["path"] = str(index_dir)
    return stats


def load_index(index_dir: Path) -> dict[str, Any]:
    """Загрузить индекс с диска (mmap по умолчанию - без копии ~1GB в heap)."""
    meta_path = index_dir / "meta.json"
    vec_path = index_dir / "vectors.npy"
    if not meta_path.is_file() or not vec_path.is_file():
        return {"ok": False, "reason": "missing_files"}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    idx_map = [int(x) for x in (meta.get("chunk_indices") or [])]
    dim = int(meta.get("dim") or 0)
    use_mmap = vector_mmap_enabled()
    if use_mmap:
        matrix = np.load(str(vec_path), mmap_mode="r")
        if matrix.dtype != np.float32:
            matrix = matrix.astype(np.float32, copy=False)
        _set_index_state(matrix, idx_map, mmap=True, backend="numpy_mmap")
    else:
        matrix = np.load(str(vec_path)).astype(np.float32, copy=False)
        _set_index_state(matrix, idx_map, mmap=False, backend="numpy")
    sidecar_n = _load_chunk_id_sidecar(index_dir)
    return {
        "ok": True,
        "indexed": len(idx_map),
        "dim": dim or (_index_dim if _index_dim else int(matrix.shape[1])),
        "backend": _index_backend,
        "mmap": use_mmap,
        "chunk_id_map": sidecar_n,
        "path": str(index_dir),
    }


def load_index_from_env(chunks: list[dict] | None = None) -> dict[str, Any]:
    """Загрузить с диска или построить из chunks при старте."""
    if not vector_index_enabled():
        return {"ok": False, "reason": "disabled"}
    if _index_chunk_indices is not None:
        return {
            "ok": True,
            "indexed": len(_index_chunk_indices),
            "dim": _index_dim,
            "backend": _index_backend,
            "mmap": _index_mmap,
            "cached": True,
        }
    index_dir = default_index_path()
    if (index_dir / "meta.json").is_file():
        return load_index(index_dir)
    if chunks:
        return build_index_from_chunks(chunks)
    return {"ok": False, "reason": "not_built"}


def ensure_index_loaded(chunks: list[dict] | None = None) -> dict[str, Any]:
    """Ленивая загрузка индекса перед vector search."""
    if not vector_index_enabled():
        return {"ok": False, "reason": "disabled"}
    if _index_chunk_indices is not None:
        return {"ok": True, "loaded": True, "indexed": len(_index_chunk_indices)}
    return load_index_from_env(chunks)


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
    if _faiss_index is not None and not _index_mmap:
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
    ensure_index_loaded()
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
    ensure_index_loaded()
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
    local_matrix = np.asarray(_index_vectors[allowed_local], dtype=np.float32)
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
        "mmap": _index_mmap,
        "backend": _index_backend or None,
        "chunk_id_map": len(_chunk_id_to_global or {}),
        "path": str(default_index_path()),
    }
