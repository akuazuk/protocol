"""Рекомендации врачу «на что обратить внимание» после подбора протоколов."""
from __future__ import annotations

import re
from typing import Any


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _funnel_population(query: str) -> str | None:
    nq = _norm(query)
    if "контекст подбора: детское" in nq or "детское население" in nq:
        return "child"
    if "контекст подбора: взрослое" in nq or "взрослое население" in nq:
        return "adult"
    if "контекст подбора: беремен" in nq or "беременные" in nq:
        return "pregnant"
    if "контекст подбора: неотлож" in nq:
        return "emergency"
    return None


def _path_audience_hint(path: str, title: str) -> str | None:
    blob = _norm(f"{path} {title}".replace("_", " "))
    if any(x in blob for x in ("дет_нас", "д-нас", "детс", "pediatr", "детск")):
        return "pediatric"
    if any(x in blob for x in ("взр_нас", "в-нас", "взросл")):
        return "adult"
    if any(x in blob for x in ("беремен", "акушер", "гинекolog", "гинеколог")):
        return "pregnant"
    return None


def build_clinical_attention(
    *,
    query: str,
    proto_list: list[dict[str, Any]] | None,
    red_flags: list[str] | None,
    audience_inferred: str | None,
    diagnostic_notice: str | None = None,
    diagnostic_mode: str | None = None,
) -> list[dict[str, str]]:
    """Список пунктов {text, severity: high|medium|info} для UI врача."""
    items: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(text: str, severity: str = "info") -> None:
        t = re.sub(r"\s+", " ", (text or "").strip())
        if not t or len(t) < 12:
            return
        key = t[:80]
        if key in seen:
            return
        seen.add(key)
        items.append({"text": t[:320], "severity": severity})

    for rf in red_flags or []:
        add(str(rf), "high")

    funnel_pop = _funnel_population(query)
    top = None
    for pr in proto_list or []:
        if isinstance(pr, dict) and pr.get("path"):
            top = pr
            break
    if top:
        path = str(top.get("path") or "")
        title = str(top.get("title") or path)
        hint = _path_audience_hint(path, title)
        if funnel_pop == "adult" and hint == "pediatric":
            add(
                "Первый протокол в выдаче помечен как детский — сверьте аудиторию пациента и при необходимости выберите взрослый КП из списка.",
                "high",
            )
        elif funnel_pop == "child" and hint == "adult":
            add(
                "Первый протокол ориентирован на взрослых — для детского случая проверьте другие позиции в top-3.",
                "high",
            )
        elif funnel_pop == "pregnant" and hint == "pediatric":
            add(
                "При подборе для беременных в top-1 детский протокол — выберите КП из акушерства/гинекологии.",
                "high",
            )
        try:
            conf = float(top.get("confidence_score") or 0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf >= 0.85:
            add(
                "Высокое соответствие top-1 запросу — откройте матрицу КЗ и отметьте обязательные пункты протокола перед оформлением заключения.",
                "info",
            )
        elif conf < 0.65:
            add(
                "Уверенность в top-1 ниже 65% — уточните МКБ/аудиторию или просмотрите протоколы из top-3 вручную.",
                "medium",
            )

    if audience_inferred and funnel_pop and audience_inferred != funnel_pop:
        if not (funnel_pop == "adult" and audience_inferred == "pediatric"):
            add(
                f"Аудитория в запросе ({funnel_pop}) не совпадает с автоопределением ({audience_inferred}) — проверьте шаг «аудитория пациента».",
                "medium",
            )

    if diagnostic_notice and diagnostic_mode in ("symptom_only", "symptom_inferred"):
        add(str(diagnostic_notice), "medium")

    if not (red_flags or []):
        add(
            "Сверьте выбранный протокол с полным PDF на сайте Минздрава: автоматический подбор не заменяет клиническое суждение.",
            "info",
        )

    return items[:8]
