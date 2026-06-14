"""Dedup of identical PDFs across rubric folders in assist retrieval."""
from __future__ import annotations

from rag_server import dedupe_protocols_list, dedupe_retrieval_by_basename


def test_dedupe_retrieval_by_basename_keeps_one_per_pdf() -> None:
    base = "КП_Диагностика_лечение_пациентов_взрос_с_доброкач_забол_прямой_кишки.pdf"
    rows = [
        {
            "path": f"minzdrav_protocols/khirurgiya/{base}",
            "score": 0.9,
            "text": "фрагмент A",
        },
        {
            "path": f"minzdrav_protocols/gastroenterologiya/{base}",
            "score": 0.85,
            "text": "фрагмент B",
        },
        {
            "path": f"minzdrav_protocols/dermatovenerologiya/{base}",
            "score": 0.8,
            "text": "фрагмент C",
        },
        {
            "path": "minzdrav_protocols/akusherstvo-ginekologiya/КП_Медицинское_наблюдение.pdf",
            "score": 0.7,
            "text": "гинекология",
        },
    ]
    out = dedupe_retrieval_by_basename(rows, prefer_slugs=["khirurgiya"])
    assert len(out) == 2
    bases = {r["path"].split("/")[-1] for r in out}
    assert base in bases
    assert "КП_Медицинское_наблюдение.pdf" in bases
    winner = next(r for r in out if r["path"].endswith(base))
    assert winner["path"].startswith("minzdrav_protocols/khirurgiya/")


def test_dedupe_protocols_list_merges_duplicate_paths() -> None:
    base = "КП_Диагностика_лечение_пациентов_взрос_с_доброкач_забол_прямой_кишки.pdf"
    protos = [
        {
            "path": f"minzdrav_protocols/novoobrazovaniya/{base}",
            "title": "Диагностика и лечение пациентов взрослых с доброкачественными заболеваниями прямой кишки",
            "confidence_score": 0.94,
            "match_reason": "Протокол охватывает геморрой.",
        },
        {
            "path": f"minzdrav_protocols/khirurgiya/{base}",
            "title": "Диагностика и лечение пациентов взрослых с доброкачественными заболеваниями прямой кишки",
            "confidence_score": 0.93,
            "match_reason": "Дубликат в другой рубрике.",
        },
        {
            "path": "minzdrav_protocols/akusherstvo-ginekologiya/КП_Медицинское_наблюдение.pdf",
            "title": "Медицинское наблюдение женщин в акушерстве и гинекологии",
            "confidence_score": 0.82,
            "match_reason": "Геморрой в послеродовом периоде.",
        },
    ]
    out = dedupe_protocols_list(protos, prefer_slugs=["khirurgiya"])
    assert len(out) == 2
    top = out[0]
    assert top["confidence_score"] == 0.94
    assert top["path"].endswith(base)
    dup = top.get("duplicate_catalog_paths") or []
    assert any("khirurgiya" in p for p in dup)
