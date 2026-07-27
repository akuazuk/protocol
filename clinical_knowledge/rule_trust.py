"""Уровни доверия к правилам протокола (Workstream B ТЗ overnight-v1).

Главный инвариант ТЗ (§4):
    Scorer не имеет права жёстко штрафовать КЗ по правилу, если применимость,
    источник и медицинский смысл не подтверждены достаточным trust level.

Уровни (§6.1):
    A = approved_by_methodist   - утверждено методистом
    B = validated_with_source   - валидная summary + подтверждённая цитата + приемлемый review
    C = auto_extracted          - llm/auto извлечение, summary без review
    D = heuristic               - path template, rich table, inferred/fallback

Политика влияния на score (§6.3):
    A/B  -> могут создавать missing, штраф и risk finding (только при точной цитате);
    C    -> только needs_human/подсказка, снижают confidence, НЕ штрафуют;
    D    -> только retrieval/routing hint, НЕ штрафуют;
    Critical/P0 от C/D НЕ блокирует send gate без независимого подтверждения.

Модуль без тяжёлых импортов: принимает ``ProtocolRule`` или ``dict`` (duck-typed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Упорядочены от наивысшего доверия к наименьшему.
TRUST_A = "A"
TRUST_B = "B"
TRUST_C = "C"
TRUST_D = "D"
TRUST_ORDER = (TRUST_A, TRUST_B, TRUST_C, TRUST_D)

_PENALTY_LEVELS = (TRUST_A, TRUST_B)

# rule_source / extraction_method -> базовый уровень (до учёта review/quote)
_HEURISTIC_SOURCES = {
    "table",
    "rich_table",
    "path",
    "path_template",
    "path_condition",
    "fallback",
    "inferred",
    "heuristic",
}
_AUTO_SOURCES = {
    "summary",
    "llm",
    "llm_draft",
    "llm_extracted",
    "auto",
    "auto_extracted",
    "enrichment",
    "legacy",
}
_MANUAL_SOURCES = {"manual", "curated", "reviewed", "methodist"}

_APPROVED_STATUSES = {"approved"}
_REVIEWED_STATUSES = {"reviewed"}


@dataclass
class RuleTrustInfo:
    """Итог классификации доверия одного правила."""

    trust_level: str = TRUST_D
    review_status: str = "not_reviewed"
    extraction_method: str = "unknown"
    source_quote_verified: bool = False
    applicability_verified: bool = False
    penalty_allowed: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trust_level": self.trust_level,
            "review_status": self.review_status,
            "extraction_method": self.extraction_method,
            "source_quote_verified": self.source_quote_verified,
            "applicability_verified": self.applicability_verified,
            "penalty_allowed": self.penalty_allowed,
            "reasons": list(self.reasons),
        }


def _get(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _source_quote(rule: Any) -> str:
    src = _get(rule, "source")
    quote = _get(src, "quote") if src is not None else None
    if not quote:
        # legacy dict может держать цитату в source.quote или напрямую
        quote = _get(rule, "quote")
    return str(quote or "").strip()


def _raw_source_token(rule: Any) -> str:
    for key in ("extraction_method", "rule_source", "source_kind", "generator"):
        val = _get(rule, key)
        if val:
            return str(val).strip().lower()
    # ProtocolRule.generated_from_summary -> summary
    if _get(rule, "generated_from_summary"):
        return "summary"
    return "unknown"


def _review_status(rule: Any) -> str:
    for key in ("review_status", "reviewed", "status"):
        val = _get(rule, key)
        if isinstance(val, bool):
            return "reviewed" if val else "not_reviewed"
        if val:
            return str(val).strip().lower()
    return "not_reviewed"


def trust_for_rule(rule: Any) -> RuleTrustInfo:
    """Консервативно классифицировать доверие к правилу (§6.2).

    Никогда не повышает C/D до B автоматически. Для A/B требуется подтверждённая
    цитата (§6.3): без цитаты правило не может быть penalty-eligible.
    """
    method = _raw_source_token(rule)
    review = _review_status(rule)
    quote = _source_quote(rule)
    has_quote = len(quote) >= 8
    reasons: list[str] = []

    if method in _HEURISTIC_SOURCES:
        base = TRUST_D
        reasons.append(f"heuristic source: {method}")
    elif method in _MANUAL_SOURCES:
        base = TRUST_B
        reasons.append(f"curated source: {method}")
    elif method in _AUTO_SOURCES:
        base = TRUST_C
        reasons.append(f"auto source: {method}")
    else:
        base = TRUST_D
        reasons.append(f"unknown source ({method}) -> heuristic")

    # review методиста повышает до A
    if review in _APPROVED_STATUSES:
        level = TRUST_A
        reasons.append("approved_by_methodist")
    elif review in _REVIEWED_STATUSES and base in (TRUST_B, TRUST_C) and has_quote:
        level = TRUST_B
        reasons.append("reviewed_with_source")
    else:
        level = base

    # Инвариант §6.3: A/B без подтверждённой цитаты не штрафуют -> понизить до C.
    if level in _PENALTY_LEVELS and not has_quote:
        level = TRUST_C
        reasons.append("no verified quote -> downgraded to advisory C")

    applicability_verified = bool(_get(rule, "applicability_verified"))
    penalty_allowed = level in _PENALTY_LEVELS and has_quote

    return RuleTrustInfo(
        trust_level=level,
        review_status=review,
        extraction_method=method,
        source_quote_verified=has_quote,
        applicability_verified=applicability_verified,
        penalty_allowed=penalty_allowed,
        reasons=reasons,
    )


def penalty_allowed(trust_level: str) -> bool:
    """Может ли правило данного уровня жёстко штрафовать (§6.3)."""
    return trust_level in _PENALTY_LEVELS


def can_hard_gate(trust_level: str) -> bool:
    """Может ли finding данного уровня блокировать send gate (§6.3).

    Только A/B. C/D critical не блокирует без независимого подтверждения.
    """
    return trust_level in _PENALTY_LEVELS


def rule_trust_diagnostics(rules: list[Any]) -> dict[str, int]:
    """Диагностика по набору правил (§6.4)."""
    total = 0
    penalty_eligible = 0
    advisory = 0
    heuristic = 0
    for rule in rules or []:
        info = trust_for_rule(rule)
        total += 1
        if info.penalty_allowed:
            penalty_eligible += 1
        elif info.trust_level == TRUST_C:
            advisory += 1
        else:
            heuristic += 1
    return {
        "rules_total": total,
        "rules_penalty_eligible": penalty_eligible,
        "rules_advisory": advisory,
        "rules_heuristic": heuristic,
    }
