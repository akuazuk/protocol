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
