"""Подбор протоколов МЗ РБ к случаю МО (не L1-балл оформления).

См. docs/plans/2026-08-08-mo-icd-first-kp-suggest-v1.md.

Cascade: текст диагноза → иначе МКБ (RU title) → иначе жалобы/анамнез.
Ищем по названию и по содержанию КП. Нет clinical hit → «нет протокола».
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from clinical_knowledge.applicability import is_child_only_kp_name
from clinical_knowledge.protocol_links import protocol_display_name, protocol_nav_api_path

ENGINE = "case_protocol_suggest_v5"
# Ниже этого score при ICD-пути не считаем «хит» (отсекает baseline ~15 specialty).
_ICD_PATH_MIN_SCORE = 40.0
_ICD_FIT_CLINICAL = 0.85
_ICD_FIT_WEAK = 0.55
MATCH_KIND_LABELS = {
    "clinical": "Клиника",
    "ddx": "Дифдиагноз",
    "specialty": "Специальность",
}

# Замечания, которые не должны попадать в reasons / gaps suggest
# (safety/doc gaps тащат чужие КП, напр. ЧЛХ по C_nsaid_dup).
_GAP_SKIP_PREFIXES = (
    "D_reg55",
    "E_template",
    "A_missing_",
    "C_nsaid",
    "C_ddi",
    "C_high_alert",
    "C_drug",
    "B_icd",
    "MED_",
    "completeness_ru",
    "stage_a_ru",
    "stage_b_ru",
)

# Жёсткий блок путей каталога по специальности случая.
_SPECIALTY_PATH_BLOCK: dict[str, tuple[str, ...]] = {
    "urolog": (
        "stomatolog",
        "chelust",
        "челюст",
        "zabolevaniya_chelust",
        "zub",
        "области рта",
        "область рта",
        "члх",
    ),
    "уролог": (
        "stomatolog",
        "chelust",
        "челюст",
        "zabolevaniya_chelust",
        "zub",
        "области рта",
        "область рта",
        "члх",
    ),
    "neurolog": ("stomatolog", "chelust", "akusher", "ginekolog", "челюст", "члх"),
    "невролог": ("stomatolog", "chelust", "akusher", "ginekolog", "челюст", "члх"),
}


def suggest_enabled() -> bool:
    raw = (os.environ.get("CASE_PROTOCOL_SUGGEST") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _gap_allowed(code: str) -> bool:
    cid = str(code or "").strip()
    if not cid:
        return False
    return not any(cid.startswith(prefix) for prefix in _GAP_SKIP_PREFIXES)


def _suggest_title(source_path: str | None, registry_title: str | None) -> str:
    return protocol_display_name(
        source_path,
        fallback=str(registry_title or "") or "Протокол МЗ",
        registry_title=registry_title,
        prefer_filename_if_truncated=True,
    )


def _diagnosis_text(clinical: dict[str, Any]) -> str:
    """Текст Dx для suggest: клиническая формулировка раньше сырого mis_diagnos/кода.

    Не обогащаем titles из справочника МКБ здесь: иначе text→ICD bridge
    «засасывает» чужие коды (Q67 позвоночник) через свои же titles.
    Bridge titles остаются на стороне карточки КП; ICD-first path добавляет
    title отдельно через `_diag_text_for_match`.
    """
    from clinical_knowledge.dx_query_expand import expand_diagnosis_query, strip_icd_tokens

    preferred = " ".join(
        str(clinical.get(key) or "")
        for key in ("clinical_diagnosis", "diagnosis_main_text", "diagnosis_short")
        if clinical.get(key)
    ).strip()
    # mis_diagnos часто только код - добавляем после текста, затем strip кодов для match
    mis = str(clinical.get("mis_diagnos") or "").strip()
    if mis and mis.upper() not in preferred.upper():
        preferred = f"{preferred} {mis}".strip()
    cleaned = strip_icd_tokens(preferred)
    expanded = expand_diagnosis_query(cleaned or preferred)
    return (expanded or cleaned or preferred).strip()


def _codes_from_case(
    clinical: dict[str, Any] | None,
    record: dict[str, Any] | None,
) -> list[str]:
    """Валидные по формату коды МКБ из МО/record (main первым)."""
    clinical = clinical if isinstance(clinical, dict) else {}
    record = record if isinstance(record, dict) else {}
    case = {**clinical, **record}
    try:
        from clinical_knowledge.diagnosis_icd import prioritize_codes
        from clinical_knowledge.mo_icd_resolve import resolve_icd_codes_from_mo

        resolved = resolve_icd_codes_from_mo(case)
        codes = [str(c).strip().upper() for c in (resolved.get("all") or []) if c]
        main = str(resolved.get("main") or "").strip().upper()
        if main and main not in codes:
            codes.insert(0, main)
        return prioritize_codes(codes)[:8]
    except Exception:  # noqa: BLE001
        return []


def _codes_in_directory(codes: list[str]) -> list[str]:
    if not codes:
        return []
    try:
        import icd_mkb

        out: list[str] = []
        seen: set[str] = set()
        for raw in codes:
            mapped = icd_mkb.canonical_ru_code(raw)
            if not mapped or not icd_mkb.is_code_in_ru_reference(mapped):
                continue
            if mapped in seen:
                continue
            seen.add(mapped)
            out.append(mapped)
        return out
    except Exception:  # noqa: BLE001
        return []


def _ru_titles_for_codes(codes: list[str]) -> list[str]:
    out: list[str] = []
    try:
        import icd_mkb
    except Exception:  # noqa: BLE001
        return out
    seen: set[str] = set()
    for code in codes[:4]:
        try:
            title = (icd_mkb.ru_title(code) or "").strip()
        except Exception:  # noqa: BLE001
            title = ""
        if title and title.lower() not in seen:
            seen.add(title.lower())
            out.append(title)
    return out


def _free_text_substantive(diag_text: str) -> bool:
    try:
        from clinical_knowledge.mo_icd_directory_eval import free_text_is_substantive

        return free_text_is_substantive(diag_text)
    except Exception:  # noqa: BLE001
        tokens = re.findall(r"[а-яёa-z]{4,}", (diag_text or "").lower())
        return len(tokens) >= 2


def _text_fits_icd_titles(diag_text: str, titles: list[str]) -> bool:
    if not diag_text or not titles:
        return False
    try:
        from clinical_knowledge.mo_icd_directory_eval import (
            TEXT_FIT_REVIEW,
            title_match_score,
        )

        return max(title_match_score(diag_text, t) for t in titles) >= TEXT_FIT_REVIEW
    except Exception:  # noqa: BLE001
        return False


def _diag_text_for_match(diag_text: str, codes_in_dir: list[str]) -> str:
    """Текст для matcher: при ICD-path дополняем RU title кода."""
    titles = _ru_titles_for_codes(codes_in_dir)
    parts = [p for p in (diag_text.strip(), " ".join(titles)) if p]
    return " ".join(parts).strip()


def _prefer_icd_path(diag_text: str, codes_in_dir: list[str]) -> bool:
    """МКБ как основа поиска только если диагноза нет (код определяет формулировку)."""
    if not codes_in_dir:
        return False
    return not _free_text_substantive(diag_text)


def _complaints_anamnesis_query(graph: dict[str, Any]) -> str:
    parts: list[str] = []
    for complaint in (graph.get("complaints") or [])[:6]:
        text = str(complaint or "").strip()
        if text:
            parts.append(text)
    anamnesis = str(graph.get("anamnesis") or "").strip()
    if anamnesis:
        parts.append(anamnesis[:400])
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def resolve_kp_query(
    *,
    diag_text: str,
    codes_in_dir: list[str],
    graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Основа поиска КП: диагноз, иначе МКБ. Жалобы КП не ищут."""
    graph = graph if isinstance(graph, dict) else {}
    if _free_text_substantive(diag_text):
        titles = _ru_titles_for_codes(codes_in_dir)
        query = diag_text
        if titles and _text_fits_icd_titles(diag_text, titles):
            query = _diag_text_for_match(diag_text, codes_in_dir)
        return {
            "source": "diagnosis",
            "query": query,
            "use_icd": False,
            "mode": "text",
        }
    if codes_in_dir:
        titles = _ru_titles_for_codes(codes_in_dir)
        query = _diag_text_for_match(" ".join(titles), codes_in_dir) or " ".join(codes_in_dir)
        return {
            "source": "icd",
            "query": query,
            "use_icd": True,
            "mode": "icd_first",
        }
    return {
        "source": "none",
        "query": "",
        "use_icd": False,
        "mode": "none",
    }


def _empty_protocol_reason(source: str) -> str:
    if source == "icd":
        return "Нет клинического протокола МЗ по этому коду МКБ"
    if source == "complaints_anamnesis":
        return "Нет клинического протокола МЗ по жалобам и анамнезу"
    if source == "none":
        return "Нет данных, чтобы подобрать протокол МЗ"
    return "Нет клинического протокола МЗ по этому диагнозу"


def _audience_from_case(
    clinical: dict[str, Any],
    record: dict[str, Any],
) -> str:
    """adult|child|unknown: годы, иначе ДР + дата визита (не сегодня)."""
    from clinical_knowledge.patient_age import resolve_patient_age

    return str(resolve_patient_age(clinical, record).get("audience") or "unknown")


def _search_query(graph: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in graph.get("diagnoses") or []:
        text = str(item.get("text") or "").strip()
        if text:
            parts.append(text[:160])
    for complaint in (graph.get("complaints") or [])[:2]:
        if complaint and not parts:
            parts.append(str(complaint)[:80])
    query = " ".join(parts).strip()
    return query[:160]


def _search_url(query: str) -> str:
    return "/doctor/search?q=" + quote(query, safe="")


def build_case_fact_graph(
    *,
    clinical: dict[str, Any] | None,
    record: dict[str, Any] | None = None,
    findings: list[dict[str, Any]] | None = None,
    llm_judge: dict[str, Any] | None = None,
    history_bundle: dict[str, Any] | None = None,
    history_visits: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Нормализованные факты случая для suggest (без сырого result).

    Коды МКБ кладём в diagnoses.icd10; путь ICD-first решает suggest.
    История пациента может обогатить query того же эпизода Dx.
    """
    clinical = clinical if isinstance(clinical, dict) else {}
    record = record if isinstance(record, dict) else {}
    findings = findings if isinstance(findings, list) else []
    _ = llm_judge  # reserved for future DDx hints

    icd_codes = _codes_from_case(clinical, record)
    codes_in_dir = _codes_in_directory(icd_codes)
    diag_text = _diagnosis_text(clinical)
    if not diag_text.strip() and codes_in_dir:
        diag_text = " ".join(_ru_titles_for_codes(codes_in_dir)).strip()
    episode_meta: dict[str, Any] = {}
    try:
        from clinical_knowledge.mo_dx_episode import resolve_dx_episode_for_suggest

        episode_meta = resolve_dx_episode_for_suggest(
            clinical=clinical,
            history_bundle=history_bundle,
            history_visits=history_visits,
        )
        episode_query = str(episode_meta.get("query") or "").strip()
        if episode_query:
            # episode query уже без чужих Dx; expand алиасами
            from clinical_knowledge.dx_query_expand import expand_diagnosis_query

            diag_text = expand_diagnosis_query(episode_query) or episode_query
    except Exception:  # noqa: BLE001
        episode_meta = {}

    complaints_raw = str(clinical.get("complaints") or "")
    complaints = [part.strip() for part in re.split(r"[;\n]+", complaints_raw) if part.strip()][:12]
    specialty = str(
        record.get("specialty")
        or record.get("specialization")
        or record.get("doctor_specialty")
        or record.get("doctor_specialization")
        or clinical.get("doctor_specialization")
        or clinical.get("specialty")
        or clinical.get("specialization")
        or ""
    ).strip()
    specialty_slug = None
    try:
        from clinical_knowledge.rubric_extractors import specialty_to_rubric

        specialty_slug = specialty_to_rubric(specialty)
    except Exception:  # noqa: BLE001
        specialty_slug = None

    gaps: list[dict[str, str]] = []
    for finding in findings[:20]:
        if not isinstance(finding, dict):
            continue
        code = str(finding.get("code") or finding.get("finding_code") or "").strip()
        if not _gap_allowed(code):
            continue
        title = str(finding.get("title_ru") or finding.get("title") or finding.get("detail_ru") or "").strip()
        if code or title:
            gaps.append({"code": code or "finding", "detail": title[:240]})

    diagnoses: list[dict[str, Any]] = []
    if diag_text or icd_codes:
        diagnoses.append(
            {
                "text": diag_text,
                "role": "primary",
                "icd10": list(icd_codes),
                "icd10_in_directory": list(codes_in_dir),
            }
        )

    from clinical_knowledge.patient_age import resolve_patient_age

    age_meta = resolve_patient_age(clinical, record)
    audience = str(age_meta.get("audience") or "unknown")
    if not specialty_slug and codes_in_dir:
        try:
            from clinical_knowledge.rubric_extractors import rubric_from_icd

            specialty_slug = rubric_from_icd(codes_in_dir)
        except Exception:  # noqa: BLE001
            specialty_slug = specialty_slug

    return {
        "case_id": str(record.get("visit_id") or record.get("case_id") or record.get("mis_id") or ""),
        "audience": audience,
        "age_years": age_meta.get("age_years"),
        "visit_date": age_meta.get("visit_date"),
        "age_source": age_meta.get("age_source"),
        "specialty": {"label": specialty, "slug": specialty_slug},
        "complaints": complaints,
        "diagnoses": diagnoses,
        "icd10": list(icd_codes),
        "icd10_in_directory": list(codes_in_dir),
        "dx_episode": {
            "mode": episode_meta.get("mode"),
            "matched_n": len(episode_meta.get("matched_visits") or []),
            "current_stem": episode_meta.get("current_stem") or "",
        },
        "plan": {
            "exam": str(clinical.get("exam_recommendations") or "")[:500],
            "treatment": str(clinical.get("treatment_recommendations") or "")[:500],
        },
        "gaps": gaps[:15],
        "objective_status": str(clinical.get("objective_status") or "")[:800],
        "anamnesis": str(clinical.get("anamnesis_doctor") or clinical.get("anamnesis_auto") or "")[:800],
    }


def _diag_overlap(item: dict[str, Any], graph: dict[str, Any]) -> float:
    from clinical_knowledge.dx_query_expand import bridge_icd_candidates
    from clinical_knowledge.protocol_match import _diag_text_overlap, _text_icd_bridge_score

    diag = " ".join(str(d.get("text") or "") for d in (graph.get("diagnoses") or []))
    if not _free_text_substantive(diag):
        diag = " ".join(_ru_titles_for_codes(list(graph.get("icd10_in_directory") or [])))
    if not diag.strip():
        return 0.0
    from clinical_knowledge.kp_validity import looks_omnibus, omnibus_lexical_card

    cardish = {
        "title": item.get("title"),
        "condition_label": item.get("matched_condition"),
        "source_path": item.get("source_path"),
        "icd10_primary": item.get("icd10_primary") or [],
        "icd10_all": [],
    }
    if looks_omnibus(cardish) or looks_omnibus(item):
        cardish = omnibus_lexical_card(cardish)
    cands = bridge_icd_candidates(diag)
    focus = {str(c.get("code") or "").upper() for c in cands if c.get("code")} or None
    return max(
        _diag_text_overlap(diag, cardish, focus_codes=focus),
        _text_icd_bridge_score(diag, cardish, bridge_cands=cands),
    )

def _icd_root(code: str) -> str:
    text = (code or "").upper().strip()
    return text[:3] if len(text) >= 3 else text


def _case_icd_roots(graph: dict[str, Any]) -> set[str]:
    return {
        _icd_root(str(code))
        for code in (graph.get("icd10_in_directory") or [])
        if code
    }


def _card_primary_roots(item: dict[str, Any]) -> set[str]:
    return {
        _icd_root(str(code))
        for code in (item.get("icd10_primary") or [])
        if code
    }


def _passes_dx_gate(item: dict[str, Any], graph: dict[str, Any]) -> bool:
    """Карта должна пересекаться с диагнозом по названию или коду. Иначе отсекаем."""
    from clinical_knowledge.kp_validity import looks_omnibus

    overlap = _diag_overlap(item, graph)
    if overlap >= 0.5:
        return True
    case_roots = _case_icd_roots(graph)
    primary = _card_primary_roots(item)
    if looks_omnibus(item):
        return bool(primary & case_roots) and overlap >= 0.35
    if primary:
        return bool(primary & case_roots) or overlap >= 0.35
    if not case_roots:
        return overlap >= 0.35
    return overlap >= 0.35


def _best_icd_fit_weight(item: dict[str, Any]) -> float:
    fits = item.get("icd_fit") or []
    if not isinstance(fits, list) or not fits:
        return 0.0
    return max(float(f.get("weight") or 0) for f in fits if isinstance(f, dict))


def _match_kind(item: dict[str, Any], graph: dict[str, Any]) -> str:
    score = float(item.get("match_score") or 0)
    icd_w = _best_icd_fit_weight(item)
    from clinical_knowledge.kp_validity import looks_omnibus

    if not _passes_dx_gate(item, graph):
        return "specialty"
    overlap = _diag_overlap(item, graph)
    primary_hit = bool(_card_primary_roots(item) & _case_icd_roots(graph))
    if not looks_omnibus(item) and primary_hit and score >= _ICD_PATH_MIN_SCORE:
        return "clinical"
    if not looks_omnibus(item) and icd_w >= _ICD_FIT_CLINICAL and score >= _ICD_PATH_MIN_SCORE:
        if primary_hit or overlap >= 0.35:
            return "clinical"
    if overlap >= 0.5 or (overlap >= 0.35 and score >= 70):
        return "clinical"
    if graph.get("gaps") and score >= 55 and overlap >= 0.25:
        return "ddx"
    return "specialty"


def _specialty_tokens(graph: dict[str, Any]) -> list[str]:
    label = str((graph.get("specialty") or {}).get("label") or "").lower()
    slug = str((graph.get("specialty") or {}).get("slug") or "").lower()
    return [token for token in (label, slug) if token]


def _path_blocked_for_specialty(row: dict[str, Any], graph: dict[str, Any]) -> bool:
    blob = (
        str(row.get("source_path") or "")
        + " "
        + str(row.get("title") or "")
        + " "
        + str(row.get("specialty_slug") or "")
    ).lower()
    for token in _specialty_tokens(graph):
        for key, blocked in _SPECIALTY_PATH_BLOCK.items():
            if key in token:
                if any(part in blob for part in blocked):
                    return True
    return False


def _protocol_dedup_key(row: dict[str, Any]) -> str:
    """Один PDF в двух рубриках (хирургия + кровообращение) - одна карточка."""
    sha = str(row.get("sha256") or "").strip().lower()
    if sha:
        return f"sha:{sha}"
    approval = row.get("approval") if isinstance(row.get("approval"), dict) else {}
    number = str(approval.get("number") or "").strip()
    day = str(approval.get("date") or "").strip()
    if number and day:
        return f"apr:{day}:{number}"
    path = str(row.get("source_path") or "").replace("\\", "/")
    name = path.rsplit("/", 1)[-1].strip().lower()
    if name:
        return f"file:{name}"
    return str(row.get("protocol_id") or id(row))


def _dedup_protocol_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = _protocol_dedup_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _icd_primary_hits(row: dict[str, Any], codes: list[str]) -> int:
    primary = {str(x).upper() for x in (row.get("icd10_primary") or []) if x}
    return sum(1 for c in codes if c in primary)


def _rehab_or_noise_penalty(row: dict[str, Any], graph: dict[str, Any]) -> int:
    """Выше = хуже. Детская БА/J45: штрафуем реабилитацию и чужой adult noise."""
    blob = (
        str(row.get("source_path") or "")
        + " "
        + str(row.get("title") or "")
        + " "
        + str(row.get("protocol_id") or "")
    ).lower()
    codes = {
        str(c).upper()
        for c in (graph.get("icd10_in_directory") or graph.get("icd10") or [])
        if c
    }
    diag_parts = [
        str(item.get("text") or "")
        for item in (graph.get("diagnoses") or [])
        if isinstance(item, dict)
    ]
    diag = " ".join(diag_parts).lower()
    audience = str(graph.get("audience") or "").lower()
    asthmaish = any(c.startswith("J45") or c.startswith("J46") for c in codes) or "астм" in diag
    pediatric = audience in {"child", "pediatric", "paediatric"}
    penalty = 0
    if "реабилитац" in blob or "rehabil" in blob:
        penalty += 5 if asthmaish else 2
    if asthmaish and ("взросл" in blob or "adult" in blob) and (pediatric or "астм" in diag):
        penalty += 3
    if asthmaish and "астм" in blob:
        penalty -= 2  # boost asthma KP
    if asthmaish and ("д-нас" in blob or "детск" in blob or "д_нас" in blob or "д-нас" in blob):
        penalty -= 3
    return penalty


def _rank_rows(
    matched: list[dict[str, Any]],
    graph: dict[str, Any],
    limit: int,
    *,
    case_codes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Предпочесть clinical; внутри - ICD fit, primary hit, score; штраф rehab."""
    codes = [str(c).upper() for c in (case_codes or graph.get("icd10_in_directory") or []) if c]
    filtered = [row for row in matched if not _path_blocked_for_specialty(row, graph)]
    decorated: list[tuple[int, int, int, float, int, float, dict[str, Any]]] = []
    audience = str(graph.get("audience") or "unknown").lower()
    unknown_aud = audience in {"", "unknown", "any"}
    for row in filtered:
        kind = _match_kind(row, graph)
        tier = {"clinical": 0, "ddx": 1, "specialty": 2}.get(kind, 3)
        blob = (
            str(row.get("source_path") or "")
            + " "
            + str(row.get("title") or "")
            + " "
            + str(row.get("population") or "")
        )
        if audience == "adult" and is_child_only_kp_name(blob):
            continue
        child_unknown = 0
        if unknown_aud and (
            is_child_only_kp_name(blob) or str(row.get("population") or "") == "child"
        ):
            child_unknown = 1
        decorated.append(
            (
                tier,
                child_unknown,
                _rehab_or_noise_penalty(row, graph),
                -_best_icd_fit_weight(row),
                -_icd_primary_hits(row, codes),
                -float(row.get("match_score") or 0),
                row,
            )
        )
    decorated.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5]))
    strong = [row for tier, *_rest, row in decorated if tier == 0]
    if len(strong) >= limit:
        return strong[:limit]
    out = strong[:]
    for *_, row in decorated:
        if row in out:
            continue
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _history_bundle_for_record(
    *,
    clinical: dict[str, Any] | None,
    record: dict[str, Any] | None,
    history_bundle: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Бандл истории пациента для suggest (из готового или со склада)."""
    if isinstance(history_bundle, dict) and history_bundle.get("engine"):
        return history_bundle
    clinical = clinical if isinstance(clinical, dict) else {}
    record = record if isinstance(record, dict) else {}
    try:
        from clinical_knowledge.mo_patient_history_bundle import attach_bundle_to_case

        case = {
            "patient_id": str(
                record.get("patient_id")
                or clinical.get("patient_id")
                or record.get("patientId")
                or ""
            ),
            "patient_key": str(record.get("patient_key") or ""),
            "visit_date": str(record.get("date") or record.get("visit_date") or "")[:10],
            "doctor_id": str(
                record.get("doctor_id")
                or record.get("specialist_id_from_visit")
                or clinical.get("doctor_id")
                or ""
            ),
            "doctor_key": str(record.get("doctor_key") or ""),
            "doctor_fio": str(record.get("doctor_fio") or ""),
            "specialty": str(
                record.get("specialty")
                or record.get("specialization")
                or clinical.get("doctor_specialization")
                or ""
            ),
            "diagnosis_code": str(
                record.get("diagnosis_code")
                or record.get("mkb_code_main")
                or clinical.get("mis_diagnos")
                or ""
            ),
            "mis_id": str(record.get("mis_id") or record.get("id") or ""),
            "visit_id": str(record.get("visit_id") or record.get("case_id") or ""),
        }
        if not case["patient_id"] and not case["patient_key"]:
            return None
        if len(case["visit_date"]) < 10:
            return None
        return attach_bundle_to_case(case)
    except Exception:  # noqa: BLE001
        return None


def clinical_kp_hit(
    suggest: dict[str, Any] | None,
    *,
    min_score: float = 50.0,
) -> dict[str, Any] | None:
    """Первый clinical-хит suggest (для рубрики/№55)."""
    if not isinstance(suggest, dict):
        return None
    for item in suggest.get("items") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("match_kind") or "") != "clinical":
            continue
        if float(item.get("score") or 0) < float(min_score):
            continue
        return item
    return None


def suggest_protocols_for_mo_case(
    *,
    clinical: dict[str, Any] | None,
    record: dict[str, Any] | None = None,
    findings: list[dict[str, Any]] | None = None,
    llm_judge: dict[str, Any] | None = None,
    history_bundle: dict[str, Any] | None = None,
    history_visits: list[dict[str, Any]] | None = None,
    limit: int = 3,
    attach_history: bool = True,
) -> dict[str, Any]:
    """Suggest для МО: при возможности подтягивает patient_history_bundle."""
    bundle = history_bundle
    if attach_history and bundle is None and not history_visits:
        bundle = _history_bundle_for_record(
            clinical=clinical, record=record, history_bundle=None
        )
    return suggest_protocols_for_case(
        clinical=clinical,
        record=record,
        findings=findings,
        llm_judge=llm_judge,
        history_bundle=bundle,
        history_visits=history_visits,
        limit=limit,
    )


def suggest_protocols_for_case(
    *,
    clinical: dict[str, Any] | None,
    record: dict[str, Any] | None = None,
    findings: list[dict[str, Any]] | None = None,
    llm_judge: dict[str, Any] | None = None,
    history_bundle: dict[str, Any] | None = None,
    history_visits: list[dict[str, Any]] | None = None,
    limit: int = 3,
) -> dict[str, Any]:
    """Top-K протоколов МЗ: диагноз, иначе МКБ; иначе нет протокола."""
    if not suggest_enabled():
        return {
            "ok": True,
            "available": False,
            "reason": "Подбор протоколов выключен (CASE_PROTOCOL_SUGGEST=0)",
            "engine": ENGINE,
            "items": [],
            "gaps": [],
        }

    from .protocol_match import match_protocol_cards, match_protocol_cards_by_diagnosis_text

    graph = build_case_fact_graph(
        clinical=clinical,
        record=record,
        findings=findings,
        llm_judge=llm_judge,
        history_bundle=history_bundle,
        history_visits=history_visits,
    )
    clinical = clinical if isinstance(clinical, dict) else {}
    raw_diag = _diagnosis_text(clinical)
    graph_diag = " ".join(str(item.get("text") or "") for item in (graph.get("diagnoses") or [])).strip()
    codes_in_dir = [str(c).upper() for c in (graph.get("icd10_in_directory") or []) if c]
    # Граф может подставить RU title кода - это не «диагноз известен».
    # Эпизод истории - да, если там есть своя формулировка.
    episode_n = int((graph.get("dx_episode") or {}).get("matched_n") or 0)
    if _free_text_substantive(raw_diag):
        resolve_text = raw_diag
    elif episode_n and _free_text_substantive(graph_diag):
        resolve_text = graph_diag
    else:
        resolve_text = raw_diag
    resolved = resolve_kp_query(diag_text=resolve_text, codes_in_dir=codes_in_dir, graph=graph)
    use_icd = bool(resolved.get("use_icd"))
    match_diag = str(resolved.get("query") or "")
    query_source = str(resolved.get("source") or "none")
    mode = str(resolved.get("mode") or ("icd_first" if use_icd else "text"))
    audience = str(graph.get("audience") or "unknown")
    facts = {
        "patient_context": {
            "adult_or_child": audience,
            "age_years": graph.get("age_years"),
            "visit_date": graph.get("visit_date"),
        },
        "consultation": {
            "icd10": list(codes_in_dir) if use_icd else [],
            "diagnosis_text": match_diag,
            "complaints": [],
            "conditions_hint": [match_diag] if match_diag else [],
            "performed_exams": [],
        },
    }
    specialty_label = str((graph.get("specialty") or {}).get("label") or "")
    specialty_slug = (graph.get("specialty") or {}).get("slug")
    fetch_limit = max(12, limit * 4)

    def _run_match(slug: str | None) -> list[dict[str, Any]]:
        if use_icd:
            return match_protocol_cards(
                facts,
                specialty_slug=slug,
                limit=fetch_limit,
                use_icd=True,
            )
        return match_protocol_cards_by_diagnosis_text(
            facts, specialty_slug=slug, limit=fetch_limit
        )

    matched: list[dict[str, Any]] = []
    if specialty_slug:
        matched = _run_match(str(specialty_slug))
    if len(matched) < limit:
        extra = _run_match(None)
        seen_ids = {str(row.get("protocol_id") or row.get("source_path") or "") for row in matched}
        for row in extra:
            pid = str(row.get("protocol_id") or row.get("source_path") or "")
            if pid in seen_ids or _path_blocked_for_specialty(row, graph):
                continue
            matched.append(row)
            seen_ids.add(pid)

    matched = _dedup_protocol_rows(matched)
    matched = [row for row in matched if _passes_dx_gate(row, graph)]
    try:
        from clinical_knowledge.kp_validity import looks_omnibus

        matched = [
            row
            for row in matched
            if not looks_omnibus(row) or _diag_overlap(row, graph) >= 0.5
        ]
    except Exception:  # noqa: BLE001
        pass
    if use_icd:
        # Отсечь specialty baseline без ICD/text clinical сигнала.
        # Содержание КП (геморрой в тексте №22) тоже clinical, даже если кода нет на карточке.
        matched = [
            row
            for row in matched
            if float(row.get("match_score") or 0) >= _ICD_PATH_MIN_SCORE
            or _best_icd_fit_weight(row) >= _ICD_FIT_WEAK
            or _diag_overlap(row, graph) >= 0.5
        ]

    ranked = _rank_rows(
        matched,
        graph,
        limit=limit,
        case_codes=codes_in_dir if use_icd else None,
    )
    # Если есть clinical-хиты - не разбавляем specialty-filler'ами в топе.
    # Без clinical не показываем 33-балльный мусор (аневризма/ОПН при I84/K64).
    clinical_ranked = [row for row in ranked if _match_kind(row, graph) == "clinical"]
    if clinical_ranked:
        ranked = clinical_ranked[:limit]
    else:
        ranked = []
    search_query = (match_diag or _search_query(graph))[:160]
    search_url = _search_url(search_query) if search_query else ""
    items: list[dict[str, Any]] = []
    for row in ranked:
        kind = _match_kind(row, graph)
        title = _suggest_title(row.get("source_path"), row.get("title"))
        reasons: list[dict[str, str]] = []
        icd_label = str(row.get("icd_fit_label") or "").strip()
        if use_icd and icd_label and kind == "clinical":
            reasons.append({"code": "icd_fit", "text": f"Совпадение по МКБ: {icd_label}"})
        if match_diag and kind == "clinical":
            short = match_diag[:120] + ("..." if len(match_diag) > 120 else "")
            reason_code = "diagnosis_fit"
            reason_prefix = "Совпадение с диагнозом"
            if query_source == "complaints_anamnesis":
                reason_code = "complaints_fit"
                reason_prefix = "Совпадение с жалобами/анамнезом"
            elif query_source == "icd":
                reason_prefix = "Совпадение с формулировкой МКБ"
            reasons.append({"code": reason_code, "text": f"{reason_prefix}: {short}"})
        if specialty_label and kind == "specialty":
            reasons.append({"code": "specialty", "text": f"Специальность случая: {specialty_label}"})
        elif specialty_label and specialty_slug and str(row.get("specialty_slug") or "") == specialty_slug:
            reasons.append({"code": "specialty", "text": f"Рубрика: {specialty_label}"})
        for gap in (graph.get("gaps") or [])[:2]:
            if gap.get("detail") and kind == "ddx":
                reasons.append(
                    {
                        "code": f"gap_{(gap.get('code') or 'x')[:40]}",
                        "text": f"Клинический разрыв: {gap['detail'][:160]}",
                    }
                )
        if not reasons:
            reasons.append(
                {
                    "code": "lexical",
                    "text": "Совпадение по диагнозу (код МКБ и/или формулировка)",
                }
            )
        source_path = str(row.get("source_path") or "")
        items.append(
            {
                "protocol_id": row.get("protocol_id"),
                "title": title,
                "source_path": source_path,
                "score": round(float(row.get("match_score") or 0), 1),
                "match_kind": kind,
                "match_kind_label": MATCH_KIND_LABELS.get(kind, kind),
                "reasons": reasons[:4],
                "covered_gaps": [g.get("code") for g in (graph.get("gaps") or [])[:3] if g.get("code")],
                "warnings": [],
                "viewer_url": protocol_nav_api_path(source_path) if source_path else None,
                "search_query": search_query,
                "search_url": search_url,
                "icd_fit": list(row.get("icd_fit") or [])[:6],
            }
        )
    reason = None
    if not items:
        reason = _empty_protocol_reason(query_source)
    return {
        "ok": True,
        "available": bool(items),
        "engine": ENGINE,
        "mode": mode,
        "query_source": query_source,
        "case_id": graph.get("case_id"),
        "gaps": graph.get("gaps") or [],
        "dx_episode": graph.get("dx_episode") or {},
        "search_query": search_query,
        "search_url": search_url,
        "items": items,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "reason": reason,
    }
