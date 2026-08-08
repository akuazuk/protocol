"""Фазы 1-2: оркестратор Dx↔МКБ, compact-коды, aliases, MIS agreement."""
from __future__ import annotations

import icd_mkb
from clinical_knowledge.mo_icd_aliases import expand, match_query
from clinical_knowledge.mo_icd_match_pipeline import (
    evaluate_mo_icd_match,
    normalize_mis_agreement,
)
from clinical_knowledge.mo_icd_resolve import _normalize_code, resolve_icd_codes_from_mo


def test_compact_icd_in_reference() -> None:
    assert icd_mkb._canonicalize_icd_like_token("K293") == "K29.3"
    assert icd_mkb._canonicalize_icd_like_token("k293") == "K29.3"
    assert _normalize_code("K293") == "K29.3"


def test_compact_icd_rejects_unknown() -> None:
    assert icd_mkb._canonicalize_icd_like_token("K9999") is None
    assert icd_mkb._canonicalize_icd_like_token("Z9999") is None


def test_compact_icd_extracted_from_mo_text() -> None:
    case = {
        "clinical_diagnosis": "Хронический гастрит K293",
        "objective_status": "",
    }
    resolved = resolve_icd_codes_from_mo(case)
    assert "K29.3" in (resolved.get("all") or [])


def test_normalize_mis_agreement_export_and_legacy() -> None:
    assert normalize_mis_agreement("match") == "match"
    assert normalize_mis_agreement("mismatch") == "mismatch"
    assert normalize_mis_agreement("partial") == "partial"
    assert normalize_mis_agreement("unknown") == "unknown"
    assert normalize_mis_agreement("1") == "match"
    assert normalize_mis_agreement("0") == "mismatch"
    assert normalize_mis_agreement("") == "skip"


def test_alias_orvi_expands() -> None:
    info = expand("ОРВИ")
    assert info["seed_codes"]
    assert "J06.9" in info["seed_codes"]
    assert info["expanded_phrases"]
    q = match_query("ОРВИ")
    assert "инфекц" in q.lower() or "дыхательн" in q.lower()


def test_alias_word_expand_chronic() -> None:
    info = expand("хр. гастрит")
    assert "хроническ" in info["normalized"].lower()


def test_alias_alone_does_not_force_ok_without_directory(monkeypatch) -> None:
    """seed_codes не ставят chip ok без сверки со справочником."""
    monkeypatch.setattr(
        "icd_mkb.suggest_icd_from_russian",
        lambda text, max_results=8: [],
    )
    # мусор + alias stem который не сматчится suggest
    pipe = evaluate_mo_icd_match(
        {
            "clinical_diagnosis": "ааа ббб ввв без нозологии",
            "mkb_code_agreement": "unknown",
        }
    )
    assert pipe["chip"]["status"] != "ok" or pipe["pipeline_verdict"] != "ok"
    assert pipe["chip"]["status"] in {
        "not_in_directory",
        "weak_name",
        "missing_dx",
        "ok",
    }
    # явный мусор без кода - не ok
    assert pipe["chip"]["status"] != "ok"


def test_pipeline_cystitis_etalon(monkeypatch) -> None:
    monkeypatch.setattr(
        "icd_mkb.suggest_icd_from_russian",
        lambda text, max_results=8: [
            {
                "code": "N30.0",
                "title_ru": "N30.0 - Острый цистит",
                "score": 0.6,
                "match_method": "lexicon_ru",
            }
        ]
        if "цистит" in (text or "").lower()
        else [],
    )
    pipe = evaluate_mo_icd_match(
        {
            "clinical_diagnosis": "Острый цистит N30.0",
            "mkb_code_main": "N30.0",
        }
    )
    assert pipe["chip"]["status"] == "ok"
    assert pipe["pipeline_verdict"] == "ok"
    assert "N30.0" in pipe["codes"]


def test_pipeline_orvi_after_alias(monkeypatch) -> None:
    def _suggest(text, max_results=8):
        low = (text or "").lower()
        if "инфекц" in low or "дыхательн" in low or "орви" in low:
            return [
                {
                    "code": "J06.9",
                    "title_ru": "J06.9 - Острая инфекция верхних дыхательных путей неуточненная",
                    "score": 0.5,
                    "match_method": "lexicon_ru",
                }
            ]
        return []

    monkeypatch.setattr("icd_mkb.suggest_icd_from_russian", _suggest)
    pipe = evaluate_mo_icd_match(
        {
            "clinical_diagnosis": "ОРВИ J06.9",
            "mkb_code_main": "J06.9",
        }
    )
    assert pipe["chip"]["status"] in {"ok", "weak_name"}
    assert pipe["name_only"].get("verdict") in {"ok", "review"}
    assert "J06.9" in (pipe.get("seed_codes") or []) or pipe["alias_expanded"]


def test_pipeline_gastritis_wrong_code_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        "icd_mkb.suggest_icd_from_russian",
        lambda text, max_results=8: [
            {
                "code": "K29.3",
                "title_ru": "K29.3 - Хронический поверхностный гастрит",
                "score": 0.55,
                "match_method": "lexicon_ru",
            }
        ]
        if "гастрит" in (text or "").lower()
        else [],
    )
    pipe = evaluate_mo_icd_match(
        {
            "clinical_diagnosis": "гастрит M60",
            "mkb_code_main": "M60",
        }
    )
    codes = {f.get("code") for f in pipe["findings"]}
    assert "B_icd_dir_text_mismatch" in codes or pipe["chip"]["status"] in {
        "weak_name",
        "not_in_directory",
    }


def test_pipeline_mis_mismatch_finding() -> None:
    pipe = evaluate_mo_icd_match(
        {
            "clinical_diagnosis": "Острый цистит N30.0",
            "mkb_code_main": "N30.0",
            "mkb_code_agreement": "mismatch",
            "mkb_code_mis": "K29.3",
        }
    )
    assert pipe["mis_agreement"] == "mismatch"
    assert any(f.get("code") == "B_icd_mismatch_mis" for f in pipe["findings"])
    # chip не портится осью E
    assert pipe["chip"]["status"] != "missing_dx"


def test_deep_eval_mis_mismatch_string() -> None:
    from clinical_knowledge.kz_deep_eval import evaluate_kz_deep

    deep = evaluate_kz_deep(
        {
            "clinical_diagnosis": "Острый цистит",
            "mkb_code_main": "N30.0",
            "mkb_code_agreement": "mismatch",
            "mkb_code_mis": "K29.3",
            "complaints": "боль при мочеиспускании",
            "objective_status": "живот мягкий",
        },
        protocol_ctx=None,
        drug_ctx={},
    )
    codes = {f.get("code") for f in (deep.get("findings") or [])}
    # primary concordance axis B3
    assert "B_icd_mismatch_mis" in codes
