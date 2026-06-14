"""Rerank протоколов при symptom-only запросах."""
from __future__ import annotations

from rag_server import _rerank_protocols_symptom_only


def test_symptom_only_demotes_mycobacteriosis():
    protos = [
        {"path": "a/саркоидоз.pdf", "title": "Саркоидоз", "confidence_score": 0.94},
        {"path": "b/пневмония.pdf", "title": "Внебольничная пневмония", "confidence_score": 0.88},
    ]
    icd = {"explicit_icd_in_query": False, "detected": [], "suggested": []}
    out = _rerank_protocols_symptom_only(protos, "кашель и температура 39", icd)
    assert out[0]["path"].endswith("пневмония.pdf")
