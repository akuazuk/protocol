"""Онкориск: байесовский триаж (LR + priors) с уровнем дозаполнения и B2C-вопросами.

Советующий слой (decision-support), НЕ диагноз и НЕ send_gate. Рантайм - только
локальный lookup + арифметика, без вызовов внешних API.

Источники чисел: data/onco_risk/*.yaml (CAPER, NICE NG12, GLOBOCAN/РНПЦ Belarus).
Подробности и оговорки: docs/onco-risk-assessment-plan.md.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "onco_risk"

# Доля вклада второго и последующих признаков (поправка на зависимость, см. план).
_SHRINK_SECONDARY = 0.5
# Кап итоговой вероятности по сайту (не выдаём ложную «уверенность»).
_P_CAP = 0.9
# Базовая концентрация Beta для доверительного интервала (масштабируется полнотой).
_BETA_KAPPA_BASE = 25.0

_SURVEILLANCE_MARKERS = (
    "химиотерап", "полихимиотерап", "лучевая терап", "метастаз", "tnm",
    "на учёте у онколог", "на учете у онколог", "динамическое наблюдение",
    "состояние после", "ремисси", "онкоцентр", "онкодиспансер",
)
_HEREDITARY_MARKERS = (
    "семейн", "наследствен", "brca", "линч", "lynch", "отягощённый анамнез",
    "отягощенный анамнез",
)


def _norm(text: str) -> str:
    s = (text or "").lower().replace("ё", "е")
    return re.sub(r"\s+", " ", s)


@lru_cache(maxsize=1)
def _load(name: str) -> dict[str, Any]:
    path = DATA_DIR / name
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _lr_data() -> dict[str, Any]:
    return _load("likelihood_ratios.yaml")


def _priors() -> dict[str, Any]:
    return _load("priors_belarus.yaml")


def _thresholds() -> dict[str, Any]:
    return _load("thresholds_nice_ng12.yaml")


def _required() -> dict[str, Any]:
    return _load("required_inputs.yaml")


def _b2c() -> dict[str, Any]:
    return _load("b2c_question_templates.yaml")


def _pediatric() -> dict[str, Any]:
    return _load("pediatric_signals.yaml")


def _screening() -> dict[str, Any]:
    return _load("screening_belarus.yaml")


# --------------------------------------------------------------------------- #
# Модели данных
# --------------------------------------------------------------------------- #
@dataclass
class OncoInputs:
    text: str = ""
    age: int | None = None
    sex: str = "unknown"  # male | female | unknown
    labs_text: str = ""
    smoking: bool | None = None
    family_history: bool | None = None
    bmi: float | None = None
    symptom_duration_known: bool = False
    adult_or_child: str = "adult"  # adult | child


@dataclass
class FeatureHit:
    id: str
    label_ru: str
    cancer_site: str
    lr: float | None
    ppv_single: float | None
    is_lab: bool
    source: dict[str, Any]


@dataclass
class Contributor:
    feature: str
    label_ru: str
    lr_effective: float
    p_before: float
    p_after: float


@dataclass
class SiteRisk:
    site: str
    p: float
    ci_low: float
    ci_high: float
    contributors: list[Contributor] = field(default_factory=list)


@dataclass
class Completeness:
    score: float
    missing: list[str]
    band: str  # qualitative_only | quantitative_wide_ci | quantitative


@dataclass
class OncoAssessment:
    context: str
    features: list[FeatureHit]
    completeness: Completeness
    sites: list[SiteRisk]
    any_cancer_p: float
    any_cancer_ci: tuple[float, float]
    triage_level: str
    b2c_questions: list[str]
    advisory_note: str


# --------------------------------------------------------------------------- #
# Извлечение признаков и контекста
# --------------------------------------------------------------------------- #
def _strata_ok(strata: dict[str, Any], age: int | None, sex: str) -> bool:
    s_sex = str(strata.get("sex") or "any")
    if s_sex != "any" and sex != "unknown" and sex != s_sex:
        return False
    a_min = strata.get("age_min")
    if a_min is not None and age is not None and age < int(a_min):
        return False
    return True


def _match_feature(entry: dict[str, Any], blob: str, is_lab: bool,
                   age: int | None, sex: str) -> FeatureHit | None:
    kws = [_norm(k) for k in (entry.get("keywords") or [])]
    if not any(k and k in blob for k in kws):
        return None
    if not _strata_ok(entry.get("strata") or {}, age, sex):
        return None
    return FeatureHit(
        id=str(entry.get("id")),
        label_ru=str(entry.get("label_ru") or entry.get("id")),
        cancer_site=str(entry.get("cancer_site") or "unknown"),
        lr=entry.get("lr"),
        ppv_single=entry.get("ppv_single"),
        is_lab=is_lab,
        source=entry.get("source") or {},
    )


def extract_features(inp: OncoInputs) -> list[FeatureHit]:
    blob = _norm(inp.text)
    lab_blob = _norm(f"{inp.text} {inp.labs_text}")
    data = _lr_data()
    hits: list[FeatureHit] = []
    seen: set[str] = set()
    for entry in data.get("features") or []:
        h = _match_feature(entry, blob, False, inp.age, inp.sex)
        if h and h.id not in seen:
            hits.append(h)
            seen.add(h.id)
    for entry in data.get("lab_features") or []:
        h = _match_feature(entry, lab_blob, True, inp.age, inp.sex)
        if h and h.id not in seen:
            hits.append(h)
            seen.add(h.id)
    return hits


def classify_context(inp: OncoInputs, features: list[FeatureHit]) -> str:
    blob = _norm(inp.text)
    if any(m in blob for m in _SURVEILLANCE_MARKERS):
        return "surveillance"
    if inp.adult_or_child == "child":
        return "pediatric"
    if features:
        return "symptomatic"
    if inp.family_history or any(m in blob for m in _HEREDITARY_MARKERS):
        return "hereditary"
    if _screening_eligible(inp.age, inp.sex):
        return "screening"
    return "none"


def _screening_eligible(age: int | None, sex: str) -> bool:
    if age is None:
        return False
    for prog in _screening().get("programs") or []:
        p_sex = str(prog.get("sex") or "any")
        if p_sex != "any" and sex != "unknown" and sex != p_sex:
            continue
        if int(prog.get("age_min", 0)) <= age <= int(prog.get("age_max", 200)):
            return True
    return False


# --------------------------------------------------------------------------- #
# Полнота данных
# --------------------------------------------------------------------------- #
def data_completeness(inp: OncoInputs, features: list[FeatureHit]) -> Completeness:
    req = _required()
    weights: dict[str, float] = req.get("weights") or {}
    present: dict[str, bool] = {
        "age": inp.age is not None,
        "sex": inp.sex in ("male", "female"),
        "symptom_present": bool(features),
        "symptom_duration": bool(inp.symptom_duration_known),
        "relevant_labs": any(f.is_lab for f in features) or bool(inp.labs_text.strip()),
        "smoking": inp.smoking is not None,
        "family_history": inp.family_history is not None,
        "bmi": inp.bmi is not None,
    }
    total_w = sum(weights.values()) or 1.0
    score = sum(w for k, w in weights.items() if present.get(k)) / total_w
    missing = [k for k in weights if not present.get(k)]

    bands = req.get("completeness_bands") or {}
    q_only = float(bands.get("qualitative_only", 0.40))
    wide = float(bands.get("quantitative_wide_ci", 0.70))
    if score < q_only:
        band = "qualitative_only"
    elif score < wide:
        band = "quantitative_wide_ci"
    else:
        band = "quantitative"
    return Completeness(score=round(score, 3), missing=missing, band=band)


# --------------------------------------------------------------------------- #
# Байес
# --------------------------------------------------------------------------- #
def _baseline(site: str) -> float:
    sites = _priors().get("sites") or {}
    cfg = sites.get(site) or {}
    p0 = cfg.get("baseline_symptomatic")
    return float(p0) if p0 else 0.001


def _ppv_to_lr(ppv: float, odds0: float) -> float:
    ppv = min(max(float(ppv), 1e-6), 0.95)
    odds_post = ppv / (1.0 - ppv)
    return max(odds_post / odds0, 1.0) if odds0 > 0 else 1.0


def _beta_ci(p: float, completeness: float) -> tuple[float, float]:
    p = min(max(p, 1e-6), 1 - 1e-6)
    kappa = _BETA_KAPPA_BASE * (0.4 + 0.6 * completeness)
    a = p * kappa
    b = (1 - p) * kappa
    try:
        from scipy.stats import beta as _beta  # type: ignore
        lo = float(_beta.ppf(0.025, a, b))
        hi = float(_beta.ppf(0.975, a, b))
    except Exception:
        sd = math.sqrt(p * (1 - p) / (kappa + 1))
        lo = max(0.0, p - 1.96 * sd)
        hi = min(1.0, p + 1.96 * sd)
    return round(lo, 4), round(hi, 4)


def posterior_risk(features: list[FeatureHit], completeness: float) -> list[SiteRisk]:
    by_site: dict[str, list[FeatureHit]] = {}
    for f in features:
        if f.cancer_site and f.cancer_site != "unknown":
            by_site.setdefault(f.cancer_site, []).append(f)

    out: list[SiteRisk] = []
    for site, feats in by_site.items():
        p0 = _baseline(site)
        odds0 = p0 / (1 - p0)

        effective: list[tuple[FeatureHit, float]] = []
        for f in feats:
            if f.lr:
                lr = float(f.lr)
            elif f.ppv_single:
                lr = _ppv_to_lr(float(f.ppv_single), odds0)
            else:
                lr = 1.0
            effective.append((f, lr))
        effective.sort(key=lambda x: x[1], reverse=True)

        odds = odds0
        p_cur = p0
        contributors: list[Contributor] = []
        for idx, (f, lr) in enumerate(effective):
            weight = 1.0 if idx == 0 else _SHRINK_SECONDARY
            lr_eff = lr ** weight
            p_before = p_cur
            odds *= lr_eff
            p_cur = odds / (1 + odds)
            if p_cur > _P_CAP:
                p_cur = _P_CAP
                odds = p_cur / (1 - p_cur)
            contributors.append(Contributor(
                feature=f.id, label_ru=f.label_ru,
                lr_effective=round(lr_eff, 2),
                p_before=round(p_before, 4), p_after=round(p_cur, 4),
            ))
        lo, hi = _beta_ci(p_cur, completeness)
        out.append(SiteRisk(site=site, p=round(p_cur, 4), ci_low=lo, ci_high=hi,
                            contributors=contributors))
    out.sort(key=lambda s: s.p, reverse=True)
    return out


def any_cancer_risk(sites: list[SiteRisk]) -> float:
    prod = 1.0
    for s in sites:
        prod *= (1 - s.p)
    return round(min(1 - prod, _P_CAP), 4)


def triage_level(p: float, is_child: bool = False) -> str:
    th = _thresholds()
    refer = float(th.get("pediatric_ppv_threshold" if is_child else "referral_ppv_threshold", 0.03))
    low_max = 0.01
    if p >= refer:
        return "refer"
    if p >= low_max:
        return "low_not_no"
    return "low"


# --------------------------------------------------------------------------- #
# B2C: нейтральные вопросы врачу
# --------------------------------------------------------------------------- #
def _has_forbidden(text: str, forbidden: list[str]) -> bool:
    low = _norm(text)
    return any(w and w in low for w in forbidden)


def b2c_questions(features: list[FeatureHit], context: str,
                  completeness: Completeness, inp: OncoInputs) -> list[str]:
    cfg = _b2c()
    forbidden = [_norm(w) for w in (cfg.get("forbidden_words") or [])]
    by_feature = cfg.get("by_feature") or {}
    by_context = cfg.get("by_context") or {}
    intake_hints = cfg.get("intake_hints_b2c") or {}

    out: list[str] = []

    if context == "surveillance":
        out.extend(by_context.get("surveillance") or [])
    elif context == "pediatric":
        out.extend(_pediatric().get("b2c_questions_parent") or [])
    else:
        for f in features:
            out.extend(by_feature.get(f.id) or [])
        if context == "screening":
            out.extend(by_context.get("screening") or [])
        if context == "hereditary" or inp.family_history:
            out.extend(by_context.get("hereditary") or [])

    # Подсказки дозаполнения - только если данных мало и контекст про текущие симптомы.
    if completeness.band != "quantitative" and context not in ("surveillance", "pediatric"):
        for miss in completeness.missing:
            out.extend(intake_hints.get(miss) or [])

    # Safety-netting - если есть хотя бы один содержательный вопрос.
    if out:
        out.extend(cfg.get("safety_netting") or [])

    # Дедуп + фильтр запрещённых слов (страховка).
    seen: set[str] = set()
    clean: list[str] = []
    for q in out:
        q = q.strip()
        if not q or q in seen:
            continue
        if _has_forbidden(q, forbidden):
            continue
        seen.add(q)
        clean.append(q)
    return clean


# --------------------------------------------------------------------------- #
# Сборка
# --------------------------------------------------------------------------- #
def assess(inp: OncoInputs) -> OncoAssessment:
    features = extract_features(inp)
    context = classify_context(inp, features)
    completeness = data_completeness(inp, features)

    is_child = inp.adult_or_child == "child"
    # Количественную оценку не считаем для наблюдения/скрининга и при низкой полноте.
    if context in ("surveillance", "screening", "none") or completeness.band == "qualitative_only":
        sites: list[SiteRisk] = []
        any_p = 0.0
        any_ci = (0.0, 0.0)
        level = "low"
    else:
        sites = posterior_risk(features, completeness.score)
        any_p = any_cancer_risk(sites)
        any_ci = _beta_ci(any_p, completeness.score) if any_p else (0.0, 0.0)
        top = max((s.p for s in sites), default=0.0)
        level = triage_level(top, is_child)

    note = (
        "Советующая оценка онконастороженности (не диагноз). "
        "Числа ориентировочные, с доверительным интервалом; решение принимает врач."
    )
    return OncoAssessment(
        context=context,
        features=features,
        completeness=completeness,
        sites=sites,
        any_cancer_p=any_p,
        any_cancer_ci=any_ci,
        triage_level=level,
        b2c_questions=b2c_questions(features, context, completeness, inp),
        advisory_note=note,
    )
