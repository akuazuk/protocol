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


def test_symptom_only_demotes_allergic_rhinitis_with_fever():
    protos = [
        {
            "path": "a/аллергический_ринит_дет.pdf",
            "title": "Аллергический ринит дет нас",
            "confidence_score": 0.98,
        },
        {
            "path": "b/орви.pdf",
            "title": "Диагностика лечение ОРВИ",
            "confidence_score": 0.82,
        },
    ]
    icd = {"explicit_icd_in_query": False, "detected": [], "suggested": []}
    out = _rerank_protocols_symptom_only(
        protos, "температура и кашель и болит горло", icd
    )
    assert out[0]["path"].endswith("орви.pdf")


def test_symptom_only_demotes_pediatric_orvi_without_child_context():
    protos = [
        {
            "path": "a/орви_дет_нас.pdf",
            "title": "Диагностика лечение ОРВИ дет нас",
            "confidence_score": 0.98,
        },
        {
            "path": "b/пневмония.pdf",
            "title": "Внебольничная пневмония взр и детс население",
            "confidence_score": 0.86,
        },
    ]
    icd = {"explicit_icd_in_query": False, "detected": [], "suggested": []}
    out = _rerank_protocols_symptom_only(protos, "кашель и температура 38", icd)
    assert out[0]["path"].endswith("пневмония.pdf")


def test_pediatric_cough_prefers_child_orvi_over_adult_bronchitis():
  protos = [
      {
          "path": "a/бронхит_взр.pdf",
          "title": "КП диагностики и лечения острого и хронического бронхита",
          "confidence_score": 0.95,
      },
      {
          "path": "b/орви_дет.pdf",
          "title": "Диагностика лечение острых респираторных вирусных инфекций дет нас",
          "confidence_score": 0.82,
      },
  ]
  icd = {"explicit_icd_in_query": False, "detected": [], "suggested": []}
  q = "кашель\nКонтекст подбора: детское население"
  out = _rerank_protocols_symptom_only(protos, q, icd)
  assert "орви" in out[0]["path"].lower() or "респиратор" in out[0]["title"].lower()
