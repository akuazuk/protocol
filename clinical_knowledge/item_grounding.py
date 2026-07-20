"""Grounding клинических пунктов (препараты, обследования, методы лечения) в тексте протокола.

Для каждого извлечённого пункта считаем оценку опоры (support 0..1): насколько пункт
реально присутствует в тексте протокола, с привязкой к цитате и странице. Дополнительно
сверяем со структурным ICD-профилем (obligation: required/recommended).

Детерминированно, без LLM и внешних зависимостей.
"""
from __future__ import annotations

import re
from typing import Any

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-zа-я0-9]+", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+|;\s+")

# Служебные и слишком общие слова, которые не должны создавать ложную опору.
_STOP = {
    "и", "в", "во", "на", "по", "с", "со", "к", "у", "о", "об", "от", "до", "за",
    "для", "при", "или", "не", "но", "а", "то", "же", "как", "что", "это", "этот",
    "все", "всех", "его", "её", "их", "также", "если", "чтобы", "быть", "может",
    "проводится", "проводят", "назначается", "назначают", "рекомендуется",
    "рекомендуют", "показано", "включает", "является", "являются", "путем", "путём",
    "виде", "случае", "случаях", "числе", "раза", "раз", "сутки", "день", "дней",
    "через", "после", "перед", "более", "менее", "около", "затем", "также", "того",
}


def _norm(text: str | None) -> str:
    return _WS.sub(" ", (text or "").strip().lower().replace("ё", "е"))


# Медицинские аббревиатуры -> токены раскрытия. Раскрытие двунаправленное:
# и «ОАК», и «общий анализ крови» дают один набор токенов, поэтому grounding
# не теряет пункты из-за сокращений (важно при RAG_EXTRACT_GROUNDING_DROP=1).
_ABBREV: dict[str, tuple[str, ...]] = {
    "оак": ("общий", "анализ", "крови"),
    "оам": ("общий", "анализ", "мочи"),
    "бак": ("биохимический", "анализ", "крови"),
    "бх": ("биохимический", "анализ", "крови"),
    "узи": ("ультразвуковое", "исследование"),
    "уздг": ("ультразвуковая", "допплерография"),
    "кт": ("компьютерная", "томография"),
    "мрт": ("магнитно", "резонансная", "томография"),
    "экг": ("электрокардиография", "электрокардиограмма"),
    "эхокг": ("эхокардиография",),
    "эхо": ("эхокардиография",),
    "фгдс": ("фиброгастродуоденоскопия", "эзофагогастродуоденоскопия"),
    "эгдс": ("эзофагогастродуоденоскопия", "фиброгастродуоденоскопия"),
    "фг": ("флюорография",),
    "ффг": ("флюорография",),
    "огк": ("органов", "грудной", "клетки"),
    "обп": ("органов", "брюшной", "полости"),
    "сое": ("скорость", "оседания", "эритроцитов"),
    "срб": ("реактивный", "белок"),
    "тбс": ("тазобедренный", "сустав"),
    "цнс": ("центральная", "нервная", "система"),
    "ад": ("артериальное", "давление"),
    "чсс": ("частота", "сердечных", "сокращений"),
    "мно": ("международное", "нормализованное", "отношение"),
    "рентген": ("рентгенография", "рентгенологическое"),
}


def _tokens(text: str | None) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(_norm(text)):
        if tok in _STOP:
            continue
        if tok.isdigit():
            # числа (дозы) полезны, но короткие годы/номера - нет
            if len(tok) <= 4:
                out.append(tok)
            continue
        if len(tok) >= 3 or tok in _ABBREV:
            out.append(tok)
    return out


def _content_tokens(text: str | None) -> set[str]:
    toks = set(_tokens(text))
    extra: set[str] = set()
    for t in toks:
        exp = _ABBREV.get(t)
        if exp:
            extra.update(exp)
    return toks | extra


def _split_sentences(text: str | None) -> list[str]:
    t = _WS.sub(" ", (text or "").strip())
    if not t:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(t) if p.strip()]
    return parts or [t]


def _trim_quote(sent: str, max_chars: int) -> str:
    s = sent.strip()
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    sp = cut.rfind(" ")
    if sp >= int(max_chars * 0.5):
        cut = cut[:sp]
    return cut.rstrip(" ,;-") + "…"


def score_item(
    item: str,
    sentences: list[tuple[str, Any]],
    *,
    max_quote_chars: int = 220,
) -> dict[str, Any]:
    """Опора одного пункта в списке предложений [(text, page), ...]."""
    it_tokens = _content_tokens(item)
    result: dict[str, Any] = {
        "support": 0.0,
        "page": None,
        "quote": "",
    }
    if not it_tokens:
        return result
    it_norm = _norm(item)
    # самое длинное содержательное слово - якорь названия препарата/метода
    anchor = max((w for w in it_tokens if not w.isdigit()), key=len, default="")
    best_ratio = 0.0
    best_sent = ""
    best_page = None
    for sent, page in sentences:
        sent_norm = _norm(sent)
        if not sent_norm:
            continue
        s_tokens = _content_tokens(sent)
        if not s_tokens:
            continue
        inter = it_tokens & s_tokens
        ratio = len(inter) / float(len(it_tokens))
        # прямое вхождение короткого пункта в предложение - сильный сигнал
        if it_norm and len(it_norm) <= 80 and it_norm in sent_norm:
            ratio = max(ratio, 0.95)
        # якорное слово присутствует - гарантируем ощутимую опору
        elif anchor and len(anchor) >= 5 and anchor in sent_norm:
            ratio = max(ratio, 0.5 + 0.5 * ratio)
        if ratio > best_ratio:
            best_ratio = ratio
            best_sent = sent
            best_page = page
            if best_ratio >= 0.99:
                break
    result["support"] = round(min(1.0, best_ratio), 3)
    if best_sent and best_ratio >= 0.3:
        result["quote"] = _trim_quote(best_sent, max_quote_chars)
        result["page"] = best_page
    return result


def _sentences_from_chunks(chunks: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for ch in chunks or []:
        text = ch.get("text") or ""
        page = ch.get("page_from") or ch.get("page")
        for sent in _split_sentences(text):
            if len(sent) >= 8:
                out.append((sent, page))
    return out


def _obligation_lookup(profile_items: list[dict[str, Any]] | None) -> list[tuple[set[str], str]]:
    out: list[tuple[set[str], str]] = []
    for it in profile_items or []:
        if isinstance(it, dict):
            txt = it.get("text") or ""
            obl = it.get("obligation") or "recommended"
        else:
            txt, obl = str(it), "recommended"
        toks = _content_tokens(txt)
        if toks:
            out.append((toks, str(obl)))
    return out


def _match_obligation(
    item: str, lookup: list[tuple[set[str], str]]
) -> str | None:
    it_tokens = _content_tokens(item)
    if not it_tokens:
        return None
    best_obl: str | None = None
    best_ratio = 0.0
    for toks, obl in lookup:
        inter = it_tokens & toks
        if not inter:
            continue
        ratio = len(inter) / float(min(len(it_tokens), len(toks)))
        if ratio > best_ratio:
            best_ratio = ratio
            best_obl = obl
    return best_obl if best_ratio >= 0.5 else None


def ground_items(
    items: list[str],
    chunks: list[dict[str, Any]],
    *,
    profile_items: list[dict[str, Any]] | None = None,
    min_support: float = 0.34,
    max_quote_chars: int = 220,
) -> list[dict[str, Any]]:
    """Список пунктов с опорой: [{text, support, verified, page, quote, obligation, source}]."""
    sentences = _sentences_from_chunks(chunks)
    lookup = _obligation_lookup(profile_items)
    out: list[dict[str, Any]] = []
    for raw in items or []:
        text = str(raw).strip()
        if not text:
            continue
        scored = score_item(text, sentences, max_quote_chars=max_quote_chars)
        obligation = _match_obligation(text, lookup)
        support = float(scored["support"])
        source = "protocol_text" if support >= min_support else "unverified"
        if obligation is not None:
            # структурный профиль подтверждает пункт - не ниже 0.6
            support = max(support, 0.6)
            source = "icd_profile"
        out.append(
            {
                "text": text,
                "support": round(min(1.0, support), 3),
                "verified": support >= min_support,
                "page": scored.get("page"),
                "quote": scored.get("quote") or "",
                "obligation": obligation,
                "source": source,
            }
        )
    return out


def build_extraction_grounding(
    ext: dict[str, Any],
    chunks: list[dict[str, Any]],
    *,
    profile_entry: dict[str, Any] | None = None,
    min_support: float = 0.34,
) -> dict[str, Any]:
    """Grounding для секций извлечения (medications/investigations/treatment_methods)."""
    prof = profile_entry or {}
    grounded: dict[str, Any] = {}
    field_profile = {
        "medications": prof.get("medications"),
        "investigations": prof.get("diagnostics"),
        "treatment_methods": prof.get("treatment"),
    }
    total = 0
    verified = 0
    supports: list[float] = []
    for field in ("medications", "investigations", "treatment_methods"):
        items = ext.get(field) or []
        if not isinstance(items, list):
            continue
        rows = ground_items(
            [str(x) for x in items],
            chunks,
            profile_items=field_profile.get(field),
            min_support=min_support,
        )
        grounded[field] = rows
        for r in rows:
            total += 1
            supports.append(float(r["support"]))
            if r["verified"]:
                verified += 1
    grounded["summary"] = {
        "items": total,
        "verified": verified,
        "avg_support": round(sum(supports) / len(supports), 3) if supports else 0.0,
        "min_support": min_support,
    }
    return grounded
