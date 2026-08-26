"""Глубокая оценка КЗ: детекторы ошибок диагноза/лечения + оси A/B/C/D + risk-gate.

Реализация §3.4 (цепочка согласованности), §3.5 (параметрическая рубрика) и §6 ТЗ
`docs/plans/2026-07-22-kz-deep-eval-db-task-v1.md`.

Точка входа:
    evaluate_kz_deep(case, protocol_ctx=None, drug_ctx=None, icd_client=None) -> dict

Принципы:
- **Мягкая деградация**: если нет протокола/базы ЛС/возраста - соответствующие
  параметры помечаются `not_applicable`, а не штрафуются (объективность).
- **Опора на цитату**: каждый finding несёт `evidence` (фрагмент КЗ) и `source_ref`.
- **Risk-gate**: наличие P0 (потенциальный вред) ограничивает overall независимо от
  среднего - «не пропустить опасное важнее точности» (§8).
- Главный риск - ложный матч ЛС; при низкой уверенности нормализации finding
  помечается `needs_human`, а не как факт (§11).

Finding shape:
    {code, axis, severity(P0..P3|ok), passed(bool), title_ru, detail_ru,
     evidence, source_ref, needs_human(bool)}
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .drug_normalizer import extract_drugs

_DRUG_SAFETY_DIR = Path(__file__).resolve().parent.parent / "data" / "drug_safety"
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# Дефолтные пороги risk-gate; переопределяются config/deep_thresholds.yaml (Э4-калибровка).
_DEFAULT_DEEP_CFG = {
    "t_good": 80.0,
    "t_acc": 60.0,
    "min_axis_review": None,        # если задан: любая ось ниже -> не выше "review"
    "harm_flag_overall_cutoff": 0.0,
}


@lru_cache(maxsize=1)
def _load_deep_config_yaml() -> dict:
    """Пороги deep-оценки из config/deep_thresholds.yaml. Мягкая деградация к дефолтам."""
    cfg = dict(_DEFAULT_DEEP_CFG)
    p = _CONFIG_DIR / "deep_thresholds.yaml"
    if not p.is_file():
        return cfg
    try:
        import yaml

        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return cfg
    stt = data.get("status_thresholds") or {}
    if stt.get("good") is not None:
        cfg["t_good"] = float(stt["good"])
    if stt.get("acceptable") is not None:
        cfg["t_acc"] = float(stt["acceptable"])
    ma = data.get("min_axis_review")
    cfg["min_axis_review"] = None if ma in (None, "null", "None") else float(ma)
    hc = data.get("harm_flag_overall_cutoff")
    if hc is not None:
        cfg["harm_flag_overall_cutoff"] = float(hc)
    return cfg


def load_deep_config() -> dict:
    cfg = dict(_load_deep_config_yaml())
    try:
        from clinical_knowledge.mo_scoring_profile import apply_profile_to_deep_config

        return apply_profile_to_deep_config(cfg)
    except Exception:  # noqa: BLE001
        return cfg

try:
    from .term_catalog import expand_term
except Exception:  # noqa: BLE001
    def expand_term(term: str) -> tuple[str, ...]:  # type: ignore
        return (term,)

# --- severity → числовой штраф и порядок risk-gate ---
SEVERITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "ok": 9}
_AXES = ("documentation", "clinical_concordance", "safety", "regulatory")

_MKB_RE = re.compile(r"^[A-ZА-Я]\d{2}(?:\.\d{1,2})?$", re.I)
_RED_FLAG_UNCERTAINTY = re.compile(
    r"(нельзя исключить|не исключен|подозрение на|dd\b|диф(?:\.|ференциальн)"
    r"|r/o|rule out|неясн|требует уточнени)", re.I,
)


def _txt(case: dict, *keys: str) -> str:
    return " ".join(str(case.get(k) or "").strip() for k in keys if case.get(k)).strip()


def _get(obj: Any, name: str, default=None):
    """Duck-typed доступ: protocol_ctx может быть dict или ConditionSummary."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_text_list(items: Any) -> list[str]:
    out: list[str] = []
    for it in items or []:
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, dict):
            out.append(str(it.get("name") or it.get("text") or it.get("title") or ""))
        else:
            for attr in ("name", "text", "title", "drug", "group"):
                v = getattr(it, attr, None)
                if v:
                    out.append(str(v))
                    break
    return [x for x in out if x.strip()]


def _finding(code, axis, severity, passed, title, detail="", evidence="", source_ref="", needs_human=False) -> dict:
    return {
        "code": code, "axis": axis, "severity": severity, "passed": passed,
        "title_ru": title, "detail_ru": detail, "evidence": evidence[:400],
        "source_ref": source_ref, "needs_human": needs_human,
    }


# ---------------------------------------------------------------------------
# Ось A - документирование (PDQI-9-lite, детерминированные прокси)
# ---------------------------------------------------------------------------
_A_BLOCKS = [
    ("complaints", "жалобы", "complaints"),
    ("anamnesis", "анамнез", "anamnesis_doctor,anamnesis_auto"),
    ("objective_status", "объективный статус", "objective_status"),
    ("diagnosis", "диагноз", "clinical_diagnosis,diagnosis_main_text"),
    ("exams", "рекомендации по обследованию", "exam_recommendations"),
    ("treatment", "рекомендации по лечению", "treatment_recommendations"),
    ("follow_up", "план наблюдения", "dispensary_info,return_date"),
]


def _axis_documentation(case: dict) -> tuple[float, list[dict]]:
    findings: list[dict] = []
    present = 0
    for block_id, ru, keys in _A_BLOCKS:
        val = _txt(case, *keys.split(","))
        ok = len(val) >= 3
        if ok:
            present += 1
        else:
            sev = "P2" if block_id in ("diagnosis", "complaints") else "P3"
            findings.append(_finding(
                f"A_missing_{block_id}", "documentation", sev, False,
                f"Не заполнен блок: {ru}", source_ref="PDQI-9/№55",
            ))
    score = round(100.0 * present / len(_A_BLOCKS), 1)
    return score, findings


# ---------------------------------------------------------------------------
# Ось B - клиническое соответствие (цепочка §3.4)
# ---------------------------------------------------------------------------
def icd_validate(code: str, dx_text: str, icd_client=None) -> tuple[bool, str]:
    """Локальная проверка формы кода МКБ. icd_client (опц.) - вторичная сверка."""
    raw = (code or "").strip()
    # Кириллица Н/К + пробел («Н 52.1») → H52.1 до проверки формата
    try:
        from clinical_knowledge.mo_icd_resolve import _normalize_code

        code = _normalize_code(raw) if raw else ""
    except Exception:  # noqa: BLE001
        code = raw.upper().replace(" ", "")
    if not code:
        return False, "нет кода МКБ"
    if not _MKB_RE.match(code) and not re.match(r"^[A-TV-Z]\d{2}(?:\.\d{1,2})?$", code, re.I):
        return False, f"код «{raw or code}» не соответствует формату МКБ-10"
    if icd_client is not None:
        try:
            ok = bool(icd_client.validate(code))  # duck-typed
            if not ok:
                return False, f"код «{code}» не найден в справочнике МКБ"
        except Exception:  # noqa: BLE001
            pass
    return True, ""


def _match_any(needles: list[str], haystack: str) -> bool:
    hay = haystack.lower()
    for n in needles:
        n = (n or "").strip().lower()
        if not n:
            continue
        for syn in set(expand_term(n)) | {n}:
            s = syn.strip().lower()
            if len(s) >= 3 and s in hay:
                return True
    return False


def _axis_concordance(case: dict, protocol_ctx, icd_client=None) -> tuple[float | None, list[dict]]:
    findings: list[dict] = []
    params: list[float] = []

    dx_text = _txt(case, "clinical_diagnosis", "diagnosis_main_text")
    complaints = _txt(case, "complaints")
    anamnesis = _txt(case, "anamnesis_doctor", "anamnesis_auto")
    objective = _txt(case, "objective_status", "exam_data")
    exams_done = _txt(case, "exam_recommendations", "exam_data", "manipulations", "service_names")
    treatment = _txt(case, "treatment_recommendations")

    # B1: диагноз следует из жалоб/анамнеза/осмотра (детерминированный прокси)
    if dx_text:
        has_support = bool(complaints or anamnesis or objective)
        params.append(100.0 if has_support else 0.0)
        if not has_support:
            findings.append(_finding(
                "B_dx_no_support", "clinical_concordance", "P1", False,
                "Диагноз не подкреплён жалобами/анамнезом/осмотром",
                evidence=dx_text, source_ref="§3.4 chain",
            ))

    # B2: код МКБ - отсутствие при тексте диагноза не дефект.
    from .mo_icd_resolve import assess_icd_code_requirement, resolve_icd_codes_from_mo

    icd_assess = assess_icd_code_requirement(case)
    icd_resolved = resolve_icd_codes_from_mo(case)
    code = str(icd_assess.get("code") or "").strip()
    ok = bool(icd_assess.get("ok"))
    reason = str(icd_assess.get("reason_ru") or "")
    if ok and code and icd_client is not None:
        # вторичная сверка справочником только для уже валидного по форме кода
        ok, reason = icd_validate(code, dx_text, icd_client)
    scan_hay = dx_text or " ".join(
        f"{item.get('field')}={item.get('code')}"
        for item in (icd_resolved.get("sources") or [])[:4]
    )
    if (
        dx_text
        or icd_resolved.get("present")
        or case.get("mkb_code_main")
        or icd_assess.get("status") == "missing_both"
    ):
        params.append(100.0 if ok else 0.0)
        if not ok:
            findings.append(_finding(
                "B_icd_invalid", "clinical_concordance", "P2", False,
                str(icd_assess.get("title_ru") or "Проблема кодирования МКБ"),
                detail=reason,
                evidence=code or scan_hay,
                source_ref="МКБ-10",
            ))

    # B3: согласие кода с МИС (export: match|partial|mismatch|unknown; legacy 0/1)
    try:
        from .mo_icd_match_pipeline import normalize_mis_agreement

        agree_norm = normalize_mis_agreement(case.get("mkb_code_agreement"))
    except Exception:  # noqa: BLE001
        agree_raw = str(case.get("mkb_code_agreement") or "").strip().lower()
        if agree_raw in ("0", "false", "no", "mismatch"):
            agree_norm = "mismatch"
        elif agree_raw in ("1", "true", "yes", "match"):
            agree_norm = "match"
        elif agree_raw in ("partial",):
            agree_norm = "partial"
        else:
            agree_norm = "unknown"
    if agree_norm == "mismatch":
        findings.append(_finding(
            "B_icd_mismatch_mis", "clinical_concordance", "P2", False,
            "Код МКБ в тексте не совпадает с диагнозом в МИС",
            evidence=f"text={code} vs mis={case.get('mkb_code_mis')}", source_ref="mis_data.diagnos",
        ))
        params.append(0.0)
    elif agree_norm == "match":
        params.append(100.0)
    # partial / unknown - без finding mismatch

    # B4: покрытие обязательных обследований протокола
    req_exams = _as_text_list(_get(protocol_ctx, "required_exams"))
    if req_exams:
        covered = sum(1 for e in req_exams if _match_any([e], exams_done))
        cov = 100.0 * covered / len(req_exams)
        params.append(round(cov, 1))
        if covered < len(req_exams):
            missing = [e for e in req_exams if not _match_any([e], exams_done)]
            findings.append(_finding(
                "B_exams_gap", "clinical_concordance", "P2" if cov < 50 else "P3", cov >= 99,
                f"Не отражены обязательные обследования протокола ({covered}/{len(req_exams)})",
                detail="; ".join(missing[:8]), source_ref="protocol.required_exams",
            ))

    # B5: критерии диагноза протокола отражены
    crit = _as_text_list(_get(protocol_ctx, "diagnostic_criteria"))
    if crit:
        hit = sum(1 for c in crit if _match_any([c], f"{dx_text} {objective} {exams_done}"))
        cov = 100.0 * hit / len(crit)
        params.append(round(cov, 1))
        if hit == 0:
            findings.append(_finding(
                "B_criteria_absent", "clinical_concordance", "P3", False,
                "Критерии диагноза по протоколу не отражены",
                source_ref="protocol.diagnostic_criteria",
            ))

    # B6: лечение соответствует группам протокола
    tx_groups = _as_text_list(_get(protocol_ctx, "treatment"))
    if tx_groups and treatment:
        hit = sum(1 for g in tx_groups if _match_any([g], treatment))
        if hit == 0:
            findings.append(_finding(
                "B_tx_offprotocol", "clinical_concordance", "P1", False,
                "Назначенное лечение не соответствует группам протокола",
                detail="; ".join(tx_groups[:6]), evidence=treatment,
                source_ref="protocol.treatment", needs_human=True,
            ))
            params.append(0.0)
        else:
            params.append(round(100.0 * hit / len(tx_groups), 1))

    score = round(sum(params) / len(params), 1) if params else None
    return score, findings


# ---------------------------------------------------------------------------
# Ось C - безопасность (red flags + DDI + high-alert + STOPP/Beers)
# ---------------------------------------------------------------------------
def _axis_safety(case: dict, protocol_ctx, drug_ctx: dict | None) -> tuple[float, list[dict]]:
    findings: list[dict] = []
    penalty = 0.0
    drug_ctx = drug_ctx or {}

    complaints = _txt(case, "complaints")
    objective = _txt(case, "objective_status")
    treatment = _txt(case, "treatment_recommendations")
    routing = _txt(case, "dispensary_info", "return_date", "treatment_recommendations")
    age = case.get("patient_age_years")
    try:
        age = int(float(age)) if age not in (None, "") else None
    except (TypeError, ValueError):
        age = None

    # C1: red flags протокола присутствуют в клинике, но нет маршрутизации/действия
    red_flags = _as_text_list(_get(protocol_ctx, "red_flags"))
    for rf in red_flags:
        if _match_any([rf], f"{complaints} {objective}") and not routing:
            findings.append(_finding(
                "C_red_flag_unrouted", "safety", "P0", False,
                "Тревожный признак (red flag) без плана действий/маршрутизации",
                detail=rf, evidence=f"{complaints} {objective}"[:300],
                source_ref="protocol.red_flags",
            ))
            penalty += 40

    # C1b: формулировки неопределённости («нельзя исключить …») без маршрутизации
    m = _RED_FLAG_UNCERTAINTY.search(f"{_txt(case,'clinical_diagnosis')} {objective}")
    if m and not routing:
        findings.append(_finding(
            "C_uncertainty_unrouted", "safety", "P1", False,
            "Диагностическая неопределённость без плана дообследования/наблюдения",
            evidence=m.group(0), source_ref="§3.4",
        ))
        penalty += 20

    # --- разбор назначенных ЛС ---
    drugs = extract_drugs(treatment) if treatment else []
    inns = [d["inn"] for d in drugs if d.get("inn")]
    low_conf = [d for d in drugs if 0 < d.get("confidence", 0) < 0.86]

    # C2: дублирование системных НПВП (не скобки-альтернативы, не гель+таблетка)
    from .medication_safety import concurrent_systemic_nsaids, ddi_pair_has_topical_partner

    nsaids = concurrent_systemic_nsaids(treatment)
    if len(nsaids) >= 2:
        findings.append(_finding(
            "C_nsaid_dup", "safety", "P1", False,
            "Одновременно ≥2 НПВП", detail=", ".join(nsaids[:6]),
            evidence=treatment, source_ref="ISMP/клин.практика",
        ))
        penalty += 20

    # C3: DDI по DDInter среди назначенных
    def _ddi_label(drug: dict | None, inn: str) -> str:
        surface = str((drug or {}).get("surface") or "").strip()
        canon = str(inn or "").strip()
        if surface and canon and surface.lower() != canon.lower():
            return f"{surface} / {canon}"
        return surface or canon

    drug_by_inn = {
        str(d.get("inn") or "").lower(): d
        for d in drugs
        if d.get("inn")
    }
    pairs = (drug_ctx.get("ddinter") or {}).get("pairs") if isinstance(drug_ctx.get("ddinter"), dict) else None
    if pairs and len(inns) >= 2:
        seen = set()
        for i in range(len(inns)):
            for j in range(i + 1, len(inns)):
                a, b = sorted((inns[i].lower(), inns[j].lower()))
                key = f"{a}||{b}"
                if key in seen:
                    continue
                seen.add(key)
                lvl = pairs.get(key)
                if lvl in ("Major", "Moderate"):
                    left = _ddi_label(drug_by_inn.get(inns[i].lower()), inns[i])
                    right = _ddi_label(drug_by_inn.get(inns[j].lower()), inns[j])
                    pair_label = f"{left} + {right}"
                    topical = ddi_pair_has_topical_partner(
                        treatment,
                        surfaces=[
                            str((drug_by_inn.get(inns[i].lower()) or {}).get("surface") or ""),
                            str((drug_by_inn.get(inns[j].lower()) or {}).get("surface") or ""),
                            left,
                            right,
                        ],
                        inns=[inns[i], inns[j]],
                    )
                    # Топический НПВП (гель/мазь) + системный партнёр: DDInter Major
                    # часто избыточен - понижаем до Умеренно, не в полосу «Критично».
                    effective_lvl = lvl
                    sev = "P1" if lvl == "Major" else "P2"
                    title_lvl = lvl
                    if topical and lvl == "Major":
                        effective_lvl = "Moderate"
                        sev = "P2"
                        title_lvl = "Major, топический путь - понижено"
                    evidence = pair_label
                    if treatment:
                        evidence = f"{evidence}. Фрагмент плана: {treatment[:280]}"
                    finding = _finding(
                        "C_ddi", "safety", sev, False,
                        f"Лекарственное взаимодействие ({title_lvl}): {pair_label}",
                        evidence=evidence,
                        source_ref="DDInter", needs_human=True,
                    )
                    if topical:
                        finding["topical_ddi"] = True
                        finding["route"] = "topical"
                        finding["ddi_level_raw"] = lvl
                        finding["ddi_level_effective"] = effective_lvl
                    findings.append(finding)
                    penalty += 20 if sev == "P1" else 10

    # C4: high-alert без дозы/мониторинга
    ha = (drug_ctx.get("high_alert") or {}).get("high_alert") if isinstance(drug_ctx.get("high_alert"), dict) else None
    if ha:
        ha_by_inn = {(_r.get("inn") or "").lower(): _r for _r in ha}
        has_dose = bool(re.search(r"\d+\s*(мг|mg|мкг|ме|ед|мл|г\b)", treatment, re.I))
        for inn in inns:
            r = ha_by_inn.get(inn.lower())
            if r and not has_dose:
                findings.append(_finding(
                    "C_high_alert_no_dose", "safety", "P1", False,
                    f"High-alert препарат без дозы/режима: {inn}",
                    detail="требуется: " + ", ".join(r.get("requires") or []),
                    evidence=treatment, source_ref="ISMP high-alert",
                ))
                penalty += 15

    # C5: STOPP/Beers - возраст-специфичные (только при известном возрасте ≥65)
    rules = (drug_ctx.get("stopp") or {}).get("rules") if isinstance(drug_ctx.get("stopp"), dict) else None
    if rules and age is not None:
        for rule in rules:
            if rule.get("kind") != "avoid":
                continue
            if age < int(rule.get("min_age", 65)):
                continue
            r_inns = [x.lower() for x in (rule.get("inn") or [])]
            if any(inn.lower() in r_inns for inn in inns):
                findings.append(_finding(
                    f"C_{rule.get('rule_id','stopp')}", "safety",
                    "P2" if rule.get("severity") == "high" else "P3", False,
                    rule.get("text") or "Потенциально нежелательное назначение у пожилого",
                    detail=f"{rule.get('source')} (возраст {age})", source_ref=rule.get("source"),
                ))
                penalty += 10

    # низкая уверенность нормализации ЛС - пометить needs_human (не факт)
    if low_conf:
        findings.append(_finding(
            "C_drug_unresolved", "safety", "P3", True,
            "Часть назначений не удалось надёжно нормализовать - нужна проверка",
            detail="; ".join(f"{d['surface']}→{d['inn']}?" for d in low_conf[:6]),
            needs_human=True,
        ))

    score = max(0.0, round(100.0 - penalty, 1))
    return score, findings


# ---------------------------------------------------------------------------
# Итог + risk-gate
# ---------------------------------------------------------------------------
def _apply_risk_gate(
    overall: float | None,
    findings: list[dict],
    axes: dict | None = None,
    cfg: dict | None = None,
) -> tuple[float | None, str]:
    cfg = cfg or load_deep_config()
    worst = min((SEVERITY_ORDER.get(f["severity"], 9) for f in findings if not f["passed"]), default=9)
    if overall is None:
        return None, "insufficient_data"
    if worst == 0:  # P0 - потенциальный вред
        capped = min(overall, 40.0)
        return capped, "critical"
    if worst == 1:  # P1
        capped = min(overall, 60.0)
        status = "review" if capped >= 50 else "poor"
        return capped, status
    # правило слабой оси: сильная ось не должна маскировать провал другой (Э4-калибровка)
    min_axis = cfg.get("min_axis_review")
    if min_axis is not None and axes:
        present = [v for v in axes.values() if isinstance(v, (int, float))]
        if present and min(present) < float(min_axis):
            return overall, "review"
    if overall >= cfg.get("t_good", 80.0):
        return overall, "good"
    if overall >= cfg.get("t_acc", 60.0):
        return overall, "acceptable"
    return overall, "review"


@lru_cache(maxsize=1)
def load_drug_ctx() -> dict:
    """Загрузить базы ЛС (DDInter/high-alert/STOPP) один раз. Мягкая деградация."""
    ctx: dict[str, Any] = {}
    for key, fname in (
        ("ddinter", "ddinter_pairs.json"),
        ("high_alert", "high_alert.json"),
        ("stopp", "stopp_start_beers.json"),
    ):
        p = _DRUG_SAFETY_DIR / fname
        if p.is_file():
            try:
                ctx[key] = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
    return ctx


def resolve_protocol_ctx(case: dict) -> dict | None:
    """Подобрать протокол (ConditionSummary) по коду МКБ / тексту диагноза.

    Возвращает плоский dict с required_exams/conditional_exams/red_flags/
    diagnostic_criteria/treatment/kz_checklist или None. Мягкая деградация.
    """
    try:
        from .protocol_summary.loader import find_conditions_by_icd, find_conditions_by_text
    except Exception:  # noqa: BLE001
        return None

    conds = []
    from .mo_icd_resolve import resolve_icd_codes_from_mo

    code = str(resolve_icd_codes_from_mo(case).get("main") or "").strip()
    if code:
        try:
            conds = find_conditions_by_icd(code) or []
        except Exception:  # noqa: BLE001
            conds = []
    if not conds:
        dx = _txt(case, "clinical_diagnosis", "diagnosis_main_text")
        if dx:
            try:
                conds = find_conditions_by_text(dx, limit=3) or []
            except Exception:  # noqa: BLE001
                conds = []
    if not conds:
        return None
    c = conds[0]
    return {
        "condition_id": _get(c, "condition_id"),
        "name": _get(c, "name"),
        "required_exams": _get(c, "required_exams") or [],
        "conditional_exams": _get(c, "conditional_exams") or [],
        "red_flags": _get(c, "red_flags") or [],
        "diagnostic_criteria": _get(c, "diagnostic_criteria") or [],
        "treatment": _get(c, "treatment") or [],
        "kz_checklist": _get(c, "kz_checklist") or [],
    }


def evaluate_kz_deep(
    case: dict,
    protocol_ctx=None,
    drug_ctx: dict | None = None,
    icd_client=None,
    label_ctx: dict | None = None,
) -> dict:
    """Главная точка входа. См. модульную документацию."""
    # Бандл истории один раз на case - до concordance / name_only (читают summary)
    try:
        from .mo_patient_history_bundle import attach_bundle_to_case, patient_history_enabled

        if patient_history_enabled() and isinstance(case, dict):
            attach_bundle_to_case(case)
    except Exception:  # noqa: BLE001
        pass

    a_score, a_find = _axis_documentation(case)
    b_score, b_find = _axis_concordance(case, protocol_ctx, icd_client)
    c_score, c_find = _axis_safety(case, protocol_ctx, drug_ctx)

    findings = a_find + b_find + c_find

    axes = {
        "documentation": a_score,
        "clinical_concordance": b_score,
        "safety": c_score,
        "regulatory": None,
    }
    # ось D (№55) - section-pack прил.1 разд. V (0/0.5/1, n/a вне знаменателя)
    reg55 = None
    try:
        from .mo_reg55_section import evaluate_reg55_section, to_reg55_detail_payload

        section = evaluate_reg55_section(case)
        reg55 = to_reg55_detail_payload(section)
        axes["regulatory"] = reg55.get("regulatory_compliance_pct")
        measures = list(reg55.get("measures") or [])
        if measures:
            detail = "; ".join(
                f"{m.get('point')}: {m.get('title')} ({m.get('score')})"
                for m in measures[:6]
            )
            band = str(reg55.get("reg55_band") or "")
            sev = "P1" if band == "noncompliant" else "P2"
            findings.append(_finding(
                "D_reg55_gap", "regulatory", sev, False,
                "Недостатки по пунктам постановления МЗ № 55 (разд. V)",
                detail=detail,
                source_ref="Пост. №55 · разд. V",
            ))
    except Exception:  # noqa: BLE001
        pass

    shadow_findings: list[dict] = []
    try:
        from .mo_concordance_findings import (
            concordance_findings_enabled,
            concordance_primary_enabled,
            evaluate_mo_concordance,
        )

        if concordance_findings_enabled():
            shadow_findings = evaluate_mo_concordance(case)
            if concordance_primary_enabled() and shadow_findings:
                # Только при явном флаге: влияет на overall / risk-gate.
                findings.extend({**item, "shadow": False} for item in shadow_findings)
    except Exception:  # noqa: BLE001 - мягкая деградация
        shadow_findings = []

    # ICD pipeline v3: directory + name_match (+ aliases/compact внутри helpers)
    try:
        from .mo_icd_directory_eval import (
            icd_directory_eval_enabled,
            icd_directory_primary_enabled,
        )
        from .mo_icd_match_pipeline import (
            directory_findings_from_pipeline,
            evaluate_mo_icd_match,
            icd_pipeline_enabled,
            name_findings_from_pipeline,
        )
        from .mo_icd_name_match import (
            icd_name_match_enabled,
            icd_name_match_primary_enabled,
        )

        if icd_pipeline_enabled() and (
            icd_directory_eval_enabled() or icd_name_match_enabled()
        ):
            pipe = evaluate_mo_icd_match(case)
            if icd_directory_eval_enabled():
                dir_shadow = directory_findings_from_pipeline(pipe)
                if dir_shadow:
                    shadow_findings = list(shadow_findings) + list(dir_shadow)
                    if icd_directory_primary_enabled():
                        findings.extend({**item, "shadow": False} for item in dir_shadow)
            if icd_name_match_enabled():
                name_shadow = name_findings_from_pipeline(pipe)
                if name_shadow:
                    shadow_findings = list(shadow_findings) + list(name_shadow)
                    if icd_name_match_primary_enabled():
                        findings.extend({**item, "shadow": False} for item in name_shadow)
        else:
            from .mo_icd_directory_eval import evaluate_mo_icd_directory
            from .mo_icd_name_match import evaluate_mo_icd_name_match

            if icd_directory_eval_enabled():
                dir_shadow = evaluate_mo_icd_directory(case)
                if dir_shadow:
                    shadow_findings = list(shadow_findings) + list(dir_shadow)
                    if icd_directory_primary_enabled():
                        findings.extend({**item, "shadow": False} for item in dir_shadow)
            if icd_name_match_enabled():
                name_shadow = evaluate_mo_icd_name_match(case)
                if name_shadow:
                    shadow_findings = list(shadow_findings) + list(name_shadow)
                    if icd_name_match_primary_enabled():
                        findings.extend({**item, "shadow": False} for item in name_shadow)
    except Exception:  # noqa: BLE001
        pass

    # История пациента (бандл + одно shadow МО); patient_id наружу не отдаём
    try:
        from .mo_patient_history_bundle import (
            evaluate_mo_patient_history,
            patient_history_enabled,
            patient_history_primary_enabled,
        )

        if patient_history_enabled():
            hist_shadow = evaluate_mo_patient_history(case)
            if hist_shadow:
                shadow_findings = list(shadow_findings) + list(hist_shadow)
                if patient_history_primary_enabled():
                    findings.extend({**item, "shadow": False} for item in hist_shadow)
    except Exception:  # noqa: BLE001
        pass

    # Rceth label-check: только shadow, не в overall / risk-gate / очередь Критично
    try:
        from .rceth_label_findings import (
            evaluate_rceth_label_findings,
            rceth_label_findings_enabled,
            rceth_label_primary_enabled,
        )

        if rceth_label_findings_enabled():
            rceth_shadow = evaluate_rceth_label_findings(case, label_ctx=label_ctx)
            if rceth_shadow:
                shadow_findings = list(shadow_findings) + list(rceth_shadow)
                if rceth_label_primary_enabled():
                    findings.extend({**item, "shadow": False} for item in rceth_shadow)
    except Exception:  # noqa: BLE001
        pass

    try:
        from .mo_lab_bundle import lab_bundle_enabled
        from .mo_lab_shadow import (
            evaluate_lab_for_case,
            lab_shadow_enabled,
        )

        if lab_bundle_enabled() and lab_shadow_enabled():
            _, extra = evaluate_lab_for_case(case)
            for item in extra:
                if item.get("is_shadow") or item.get("shadow"):
                    shadow_findings.append(item)
                else:
                    findings.append(item)
    except Exception:  # noqa: BLE001
        pass

    # overall: среднее доступных осей (объективность - без штрафа за отсутствие протокола)
    present_axes = [v for k, v in axes.items() if v is not None]
    overall = round(sum(present_axes) / len(present_axes), 1) if present_axes else None
    overall, status = _apply_risk_gate(overall, findings, axes=axes)

    n_by_sev = {s: sum(1 for f in findings if f["severity"] == s and not f["passed"])
                for s in ("P0", "P1", "P2", "P3")}

    return {
        "axes": axes,
        "overall_pct": overall,
        "overall_status": status,
        "findings": findings,
        "shadow_findings": shadow_findings,
        "n_findings": sum(1 for f in findings if not f["passed"]),
        "n_by_severity": n_by_sev,
        "has_potential_harm": n_by_sev["P0"] > 0,
        "reg55": reg55,
        "protocol_used": protocol_ctx is not None,
    }
