"""Инфраструктура gold-разметки и калибровки scorer v3 (Workstream K ТЗ overnight-v1).

За ночную итерацию нельзя создать экспертный gold без методистов, поэтому здесь -
схема аннотации, детерминированный sample builder, поля двойной разметки/арбитража,
валидатор и evaluator (QWK/MAE/harm recall/false critical). LLM-метки хранятся как
proxy, не gold (§15).

ПДн не хранятся: аннотация ссылается на ``visit_ref`` (обезличенный идентификатор).
"""
from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Verdict = Literal[
    "compliant", "mostly_compliant", "partially_compliant", "non_compliant", "manual_review",
]
_VERDICT_ORDINAL = {
    "non_compliant": 0,
    "manual_review": 1,
    "partially_compliant": 2,
    "mostly_compliant": 3,
    "compliant": 4,
}
HarmClass = Literal["big_three", "medication", "follow_up", "none"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Pdqi9(_Base):
    """9 атрибутов PDQI-9 (шкала 1-5)."""

    up_to_date: int | None = None
    accurate: int | None = None
    thorough: int | None = None
    useful: int | None = None
    organized: int | None = None
    comprehensible: int | None = None
    succinct: int | None = None
    synthesized: int | None = None
    internally_consistent: int | None = None


class GoldAnnotation(_Base):
    visit_ref: str  # обезличенный идентификатор (не visit_id из МИС)
    annotator_id: str
    pdqi9: Pdqi9 = Field(default_factory=Pdqi9)
    axis_documentation: int | None = None  # 0-100
    axis_clinical: int | None = None
    axis_safety: int | None = None
    potential_harm: bool = False
    harm_class: HarmClass = "none"
    verdict: Verdict = "manual_review"
    notes: str = ""
    is_synthetic: bool = False
    source: Literal["human", "llm_proxy"] = "human"


class DoubleAnnotation(_Base):
    visit_ref: str
    annotation_a: GoldAnnotation | None = None
    annotation_b: GoldAnnotation | None = None
    adjudication: GoldAnnotation | None = None
    adjudication_status: Literal["pending", "agreed", "arbitrated"] = "pending"

    def consensus(self) -> GoldAnnotation | None:
        if self.adjudication is not None:
            return self.adjudication
        if self.annotation_a and self.annotation_b:
            va = _VERDICT_ORDINAL.get(self.annotation_a.verdict, 1)
            vb = _VERDICT_ORDINAL.get(self.annotation_b.verdict, 1)
            if abs(va - vb) <= 1:
                # согласие в пределах 1 балла -> берём A как консенсус
                return self.annotation_a
        return None


def validate_annotation(a: GoldAnnotation) -> list[str]:
    issues: list[str] = []
    if not a.visit_ref:
        issues.append("no visit_ref")
    if not a.annotator_id:
        issues.append("no annotator_id")
    if a.potential_harm and a.harm_class == "none":
        issues.append("potential_harm=true но harm_class=none")
    for name in ("axis_documentation", "axis_clinical", "axis_safety"):
        v = getattr(a, name)
        if v is not None and not (0 <= v <= 100):
            issues.append(f"{name} вне 0-100")
    return issues


def deterministic_visit_ref(seed: str) -> str:
    """Стабильный обезличенный идентификатор из seed (без ПДн)."""
    return "kzref_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def build_sample_slots(strata: dict[str, int], seed: int = 0) -> list[dict[str, int | str]]:
    """Детерминированный план выборки по стратам (специальность×банд×...).

    Возвращает список слотов (без обращения к ПДн) - сколько КЗ нужно из каждой страты.
    """
    slots: list[dict[str, int | str]] = []
    for key, n in sorted(strata.items()):
        for i in range(n):
            slots.append({"stratum": key, "index": i, "visit_ref": deterministic_visit_ref(f"{seed}:{key}:{i}")})
    return slots


# --------------------------------------------------------------------------- #
# Evaluator: QWK / MAE / harm recall / false critical
# --------------------------------------------------------------------------- #
def quadratic_weighted_kappa(a: list[int], b: list[int], k: int = 5) -> float | None:
    """QWK для ординальных меток 0..k-1."""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    o = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        if 0 <= x < k and 0 <= y < k:
            o[x][y] += 1
    row = [sum(o[i]) for i in range(k)]
    col = [sum(o[i][j] for i in range(k)) for j in range(k)]
    num = 0.0
    den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            e = row[i] * col[j] / n
            num += w * o[i][j]
            den += w * e
    if den == 0:
        return 1.0
    return round(1.0 - num / den, 3)


def evaluate_scorer_vs_gold(
    gold: list[GoldAnnotation], scorer_status: list[str], scorer_harm: list[bool],
) -> dict[str, float | int | None]:
    """Метрики согласия scorer vs консенсус (§15)."""
    verdict_map = {
        "good": "mostly_compliant", "acceptable": "mostly_compliant",
        "review": "manual_review", "limited_evidence": "manual_review",
        "insufficient_evidence": "manual_review", "insufficient_data": "manual_review",
        "critical": "non_compliant", "poor": "non_compliant",
    }
    g_ord = [_VERDICT_ORDINAL.get(a.verdict, 1) for a in gold]
    s_ord = [_VERDICT_ORDINAL.get(verdict_map.get(s, "manual_review"), 1) for s in scorer_status]
    qwk = quadratic_weighted_kappa(g_ord, s_ord)
    mae = round(sum(abs(x - y) for x, y in zip(g_ord, s_ord)) / len(g_ord), 3) if g_ord else None

    # harm recall + false critical
    tp = sum(1 for a, s in zip(gold, scorer_harm) if a.potential_harm and s)
    fn = sum(1 for a, s in zip(gold, scorer_harm) if a.potential_harm and not s)
    harm_recall = round(tp / (tp + fn), 3) if (tp + fn) else None
    fp_crit = sum(
        1 for a, s in zip(gold, scorer_status)
        if s in ("critical", "poor") and a.verdict in ("compliant", "mostly_compliant")
    )
    total = len(gold)
    false_critical = round(fp_crit / total, 3) if total else None
    return {
        "n": total,
        "qwk": qwk,
        "mae_ordinal": mae,
        "harm_recall": harm_recall,
        "false_critical_rate": false_critical,
    }


def synthetic_example() -> DoubleAnnotation:
    """Полностью синтетическая запись для инструкции методисту (без ПДн)."""
    a = GoldAnnotation(
        visit_ref=deterministic_visit_ref("example"),
        annotator_id="annot_1", axis_documentation=80, axis_clinical=60, axis_safety=90,
        potential_harm=False, harm_class="none", verdict="mostly_compliant",
        notes="Синтетический пример: полный первичный приём, диагноз обоснован.",
        is_synthetic=True,
    )
    b = a.model_copy(update={"annotator_id": "annot_2", "verdict": "partially_compliant"})
    return DoubleAnnotation(visit_ref=a.visit_ref, annotation_a=a, annotation_b=b)
