"""Protocol semantic: chunk_id → global_index mapping в manifest/lazy режиме."""
from __future__ import annotations

from clinical_knowledge.protocol_semantic_search import _attach_global_indices, _build_vector_hits
from clinical_knowledge.vector_index import load_index, save_index


def test_attach_global_indices_and_vector_hits(tmp_path):
    chunks = [
        {
            "path": "minzdrav_protocols/test.pdf",
            "chunk_id": "c1",
            "chunk_type": "treatment",
            "text": "Назначить диосмин и гесперидин при варикозе.",
            "embedding": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        {
            "path": "minzdrav_protocols/test.pdf",
            "chunk_id": "c2",
            "chunk_type": "diagnostics",
            "text": "УЗДС вен нижних конечностей.",
            "embedding": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    ]
    stats = save_index(tmp_path, chunks)
    assert stats["ok"]
    from clinical_knowledge import vector_index as vi

    vi._index_vectors = None
    vi._index_chunk_indices = None
    vi._chunk_id_to_global = None
    vi._global_to_local = None
    loaded = load_index(tmp_path)
    assert loaded.get("ok")
    assert loaded.get("chunk_id_map") == 2

    lazy_rows = [
        {
            "path": "minzdrav_protocols/test.pdf",
            "chunk_id": "c1",
            "kind": "treatment",
            "text": "Назначить диосмин и гесперидин при варикозе.",
        },
        {
            "path": "minzdrav_protocols/test.pdf",
            "chunk_id": "c2",
            "kind": "diagnostics",
            "text": "УЗДС вен нижних конечностей.",
        },
    ]
    _attach_global_indices(lazy_rows)
    assert lazy_rows[0].get("_global_index") == 0
    assert lazy_rows[1].get("_global_index") == 1

    hits = _build_vector_hits(
        lazy_rows,
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        top_k=4,
    )
    assert hits.get(0, 0) > 0.9
    assert hits.get(1, 0) < 0.5


def test_sidecar_not_clobbered_by_default_index_dir(tmp_path, monkeypatch):
    """Карта chunk_id -> global не должна подменяться sidecar'ом чужого каталога.

    Регресс: `_maybe_reload_chunk_id_sidecar` всегда читал
    `default_index_path()/chunk_id_global.json`. Если индекс загрузили из другого
    каталога, а по умолчанию на диске лежал другой sidecar (на машине разработчика
    это реальный `corpus_vector_index/`), карта молча заменялась на чужую:
    `global_index_for_chunk_id` возвращал None или, хуже, индекс чужой строки
    vectors.npy. В CI баг не проявлялся, потому что каталога по умолчанию нет.
    """
    from clinical_knowledge import vector_index as vi

    # Каталог "по умолчанию" с посторонней картой.
    default_dir = tmp_path / "default_index"
    default_dir.mkdir()
    (default_dir / "chunk_id_global.json").write_text(
        '{"somebody-elses-chunk": 999}', encoding="utf-8"
    )
    monkeypatch.setenv("RAG_VECTOR_INDEX_PATH", str(default_dir))

    # Настоящий индекс лежит в отдельном каталоге.
    real_dir = tmp_path / "real_index"
    chunks = [
        {
            "path": "minzdrav_protocols/test.pdf",
            "chunk_id": "c1",
            "chunk_type": "treatment",
            "text": "Назначить диосмин при варикозе.",
            "embedding": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    ]
    for attr in ("_index_vectors", "_index_chunk_indices", "_chunk_id_to_global"):
        monkeypatch.setattr(vi, attr, None, raising=False)
    monkeypatch.setattr(vi, "_sidecar_dir", None, raising=False)
    monkeypatch.setattr(vi, "_sidecar_mtime", 0.0, raising=False)

    assert save_index(real_dir, chunks)["ok"]
    assert load_index(real_dir).get("chunk_id_map") == 1

    # Именно здесь раньше происходила подмена карты.
    assert vi.global_index_for_chunk_id("c1") == 0
    assert vi.global_index_for_chunk_id("somebody-elses-chunk") is None
