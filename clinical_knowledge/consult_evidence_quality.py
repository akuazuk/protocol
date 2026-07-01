"""Фильтры качества для evidence pack и пробелов L2."""
from __future__ import annotations

import re

_KP_LINE_NOISE = re.compile(
    r"клинический протокол диагностики|"
    r"по медицинскому применению|листке-вкладыш|"
    r"учитываются результаты обследований пациента и клиническая картина|"
    r"как можно раньше обеспечить прием препаратов|"
    r"состоянием пациента\s*,?\s*$",
    re.I,
)
_MONTH_ONLY = re.compile(
    r"^(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|"
    r"октября|ноября|декабря)(?:\s*;\s*(?:января|февраля|марта|апреля|мая|июня|"
    r"июля|августа|сентября|октября|ноября|декабря))*\.?$",
    re.I,
)
_TOC_BULLET = re.compile(r"^[\s—\-–•;.,\d]+$")

# Организационно-маршрутный / процедурный текст: не является клинической выдержкой
# (кто выполняет, куда направляют, порядок направления, чем определяется).
_ORG_ROUTING = re.compile(
    r"направля(?:ю|е)т(?:ся)?\s+пациент|"
    r"на\s+консультаци\w*\s+к\s+врач|"
    r"порядок\s+направлени|"
    r"определяется\s+министерством|"
    r"выполняют\s+врачи|"
    r"в\s+организаци\w*\s+здравоохранени\w*\s+в\s+амбулаторных|"
    r"устанавливает\s+общие\s+требования\s+к\s+объ[её]му|"
    r"осуществляется\s+в\s+соответствии\s+с\s+клиническим\s+протоколом",
    re.I,
)
# Признак обрезанного слова в конце коротких выдержек (напр. «динамическое наблюден»).
_TRUNCATED_TAIL = re.compile(
    r"\b(?:наблюден|обследован|консультаци|лечени|диагностик|рекомендаци|"
    r"вмешательств|показани)$",
    re.I,
)

# Ссылочный/нормативный/структурный шум PDF, не являющийся клинической выдержкой:
# колонтитулы («стр. 5»), номера актов («№ 59», «№ 2570-XII»), портал НПА,
# постановления/приказы, приложения, scope-абзацы, разметочные теги «[...]».
_REFERENCE_NOISE: list[tuple[str, re.Pattern[str]]] = [
    ("page_marker", re.compile(r"стр\.?\s*\d", re.I)),
    ("order_number", re.compile(r"№\s*\d")),
    ("portal", re.compile(r"национальн\w+\s+правов|интернет-портал", re.I)),
    ("postanovlenie", re.compile(r"постановлени|пастанова|приказ\w*\s+министерств|утратившим\s+силу", re.I)),
    ("prilozhenie", re.compile(r"приложени\w*\s*\d|к\s+клиническому\s+протоколу|фармакотерапевтическ\w+\s+групп", re.I)),
    ("scope_clause", re.compile(r"настоящий\s+клиническ\w+\s+протокол|устанавливает\s+общие\s+требовани|определяет\s+(?:минимальн\w+\s+)?объ[её]м\s+медицинск", re.I)),
    ("markup_tag", re.compile(r"^\s*\[[a-z_]+\]", re.I)),
    ("blank_line", re.compile(r"_{3,}")),
]


# Нон-клинический/процессный/нормативный текст: описывает порядок, а не суть.
_NON_CLINICAL: list[tuple[str, re.Pattern[str]]] = [
    ("in_accordance", re.compile(r"в\s+соответствии\s+с\s+(?:международн|законодательств|настоящим|классификацией\s+болезн)|утвержд[её]нн?\w*\s|об\s+утверждении", re.I)),
    ("process", re.compile(r"осуществляется\s+в\b|определяется\s+в\s+соответствии|проводится\s+в\s+соответствии|принимается\s+решение\s+о", re.I)),
    ("terms_preamble", re.compile(r"а\s+также\s+(?:следующ|специальн|специфическ)\w+\s+термин|термин\w+\s+и\s+их\s+определени|«о\s+здравоохранении»", re.I)),
    ("who_performs", re.compile(r"выполня\w+\s+врач|(?:осуществляется|проводится)\s+врач|врач\w+-[а-яё]+\w*(?:,?\s*врач\w+\s*[–-]?\s*[а-яё]+\w*){1,}", re.I)),
    ("pharma_boilerplate", re.compile(r"по\s+международн\w+\s+непатентованн|систематическ\w+\s+или\s+заместительн\w+\s+номенклатур|включа\w+\s+основные\s+лекарственн|представлены\s+по\s+международн", re.I)),
    ("conj_fragment", re.compile(r"^(?:и|а|но|или|либо|же|то)\s+[а-яё]", re.I)),
    ("all_caps", re.compile(r"^[А-ЯЁ][А-ЯЁ\s]{7,}$")),
    ("title_only", re.compile(r"^«[^»]{20,}»\.?$")),
]

# Слова лекарственных форм/дозирования - без названия ЛС это пустой фрагмент.
_FORM_WORDS = frozenset(
    {
        "таблетки", "таблетка", "капсулы", "капсула", "раствор", "суспензия",
        "введения", "введение", "применения", "применение", "дозе", "дозы",
        "дозировки", "дозировка", "максимум", "менее", "более", "форме",
        "форма", "формы", "мазь", "гель", "крем", "сироп", "порошок",
    }
)


def reference_noise_types(text: str) -> list[str]:
    """Список типов ссылочного/нормативного шума, найденных в тексте."""
    t = text or ""
    hits = [name for name, pat in _REFERENCE_NOISE if pat.search(t)]
    hits += [name for name, pat in _NON_CLINICAL if pat.search(t.strip())]
    return hits


def is_reference_noise(text: str) -> bool:
    return bool(reference_noise_types(text))


# Выдержка, обрывающаяся на «врач/врача/…», - усечённая оргфраза («…направляют врач…»).
_ENDS_PROFESSION = re.compile(r"\bврач(?:|а|и|ей|ом|ами)$", re.I)


_CONJ_START = re.compile(r"^(?:и|а|но|или|либо|же|то|как)\s+[а-яё]", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|;\s+")

# Служебные слова (предлоги/союзы). Если предложение начинается с такого слова
# в НИЖНЕМ регистре - это обрезок середины фразы («с указанием…», «при объеме…»),
# а не самостоятельная клиническая выдержка. С Заглавной («При постановке…») - ок.
_FUNC_WORDS = frozenset(
    {
        "и", "а", "но", "или", "либо", "же", "то", "как", "с", "со", "из", "изо",
        "во", "в", "на", "по", "к", "ко", "о", "об", "обо", "от", "ото", "до",
        "для", "при", "про", "за", "над", "под", "без", "между", "через", "у",
        "что", "чтобы", "если", "также", "т", "тж",
    }
)


def _is_midsentence_fragment(sentence: str) -> bool:
    s = (sentence or "").strip()
    if not s:
        return True
    first = s[0]
    if not first.isalpha():
        return False
    if not (first.islower() and ("а" <= first.lower() <= "я" or first.lower() == "ё")):
        return False
    token = re.split(r"[\s,]+", s, 1)[0].strip(".,;:()«»-").lower()
    return token in _FUNC_WORDS


def _clean_markup(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^\s*\[[a-z_]+\]\s*", "", t, flags=re.I)  # [treatment]/[classification]
    t = re.sub(r"^\s*\d+(?:\.\d+)*[.)]\s*", "", t)  # ведущая нумерация «4.» / «4.1.»
    t = re.sub(r"\s*\(?\s*стр\.?\s*\d+\s*\)?\s*$", "", t, flags=re.I)
    return t.strip(" ,;")


def clean_clinical_sentences(
    text: str, *, min_tokens: int = 3, max_sentences: int = 2, max_chars: int = 220
) -> str | None:
    """Очищает текст сводки до 1-2 чистых клинических предложений или None.

    Режет разметку/нумерацию/колонтитулы, отбрасывает обрывки-союзы и
    ссылочно-нормативно-процессный шум, сохраняет только клинические фразы.
    """
    cleaned = _clean_markup(text)
    if not cleaned:
        return None
    good: list[str] = []
    for sent in _SENT_SPLIT.split(cleaned):
        s = _clean_markup(sent)
        if not s:
            continue
        tokens = [w for w in re.split(r"\s+", s) if w]
        if len(tokens) < min_tokens:
            continue
        if _CONJ_START.search(s) or _is_midsentence_fragment(s):
            continue
        if not is_usable_summary_excerpt(s):
            continue
        good.append(s)
        if len(good) >= max_sentences:
            break
    if not good:
        return None
    out = "; ".join(good)
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out


def _is_empty_pharma_form(text: str) -> bool:
    """True, если выдержка состоит только из слов лекформ/дозирования (нет названия ЛС)."""
    tokens = [w.strip(".,;:()«»-").lower() for w in re.split(r"[\s;]+", text or "") if w.strip()]
    tokens = [t for t in tokens if t and len(t) > 2]
    if not tokens:
        return True
    clinical = [t for t in tokens if t not in _FORM_WORDS and not t.isdigit()]
    return len(clinical) == 0


def normalize_gap_text(text: str) -> str:
    return re.sub(r"^[—\-–•\s]+", "", (text or "").strip())


def is_kp_checklist_item(text: str) -> bool:
    """Строка похожа на пункт обследования/лечения, а не оглавление PDF."""
    t = normalize_gap_text(text)
    if len(t) < 5 or len(t) > 180:
        return False
    if _TOC_BULLET.match(t):
        return False
    if _MONTH_ONLY.match(t):
        return False
    if _KP_LINE_NOISE.search(t):
        return False
    if t.lower().startswith("кп ") and "диагностика" in t.lower() and len(t) > 80:
        return False
    alpha = sum(1 for c in t if c.isalpha())
    if alpha < 6:
        return False
    return True


def is_usable_evidence_excerpt(text: str) -> bool:
    """Выдержка пригодна для таблицы evidence pack (клиническая, не мусор)."""
    t = (text or "").strip()
    if len(t) < 12:
        return False
    if not is_kp_checklist_item(t) and _KP_LINE_NOISE.search(t):
        return False
    if _MONTH_ONLY.match(t):
        return False
    if re.match(r"^\[treatment\]", t, re.I):
        return False
    if t.count(";") >= 3 and len(t) < 80:
        return False
    # Административный/нормативный текст (утверждение, подпись министра, портал НПА).
    try:
        from clinical_knowledge.chunk_tags import is_administrative_text

        if is_administrative_text(t):
            return False
    except Exception:
        pass
    # Ссылочный/нормативный/структурный шум (колонтитулы, № актов, портал, scope).
    if is_reference_noise(t):
        return False
    # Организационно-маршрутный/процедурный текст - не клиническая выдержка.
    if _ORG_ROUTING.search(t):
        return False
    if _is_empty_pharma_form(t):
        return False
    if _ENDS_PROFESSION.search(t.rstrip(".!?;:,) ")):
        return False
    # Обрывочные короткие выдержки с обрезанным словом в конце.
    tokens = [w for w in re.split(r"\s+", t) if w]
    if len(tokens) <= 2 and len(t) < 25 and not re.search(r"\d", t):
        return False
    stripped = t.rstrip(".!?;:,) ")
    if len(tokens) <= 4 and _TRUNCATED_TAIL.search(stripped):
        return False
    alpha = sum(1 for c in t if c.isalpha())
    return alpha >= 10


def is_usable_summary_excerpt(text: str) -> bool:
    """Проверка для structured-полей сводки (названия обследований, препаратов).

    В отличие от is_usable_evidence_excerpt не требует минимальной длины:
    клинические названия часто короткие («МРТ», «УЗИ», «ОАК»). Но так же
    отсекает административный/организационный/обрывочный шум.
    """
    t = (text or "").strip()
    if len(t) < 2:
        return False
    if _MONTH_ONLY.match(t):
        return False
    if _ORG_ROUTING.search(t):
        return False
    if is_reference_noise(t):
        return False
    if _is_empty_pharma_form(t):
        return False
    stripped = t.rstrip(".!?;:,) ")
    if _ENDS_PROFESSION.search(stripped):
        return False
    try:
        from clinical_knowledge.chunk_tags import is_administrative_text

        if is_administrative_text(t):
            return False
    except Exception:
        pass
    tokens = [w for w in re.split(r"\s+", t) if w]
    if len(tokens) <= 4 and _TRUNCATED_TAIL.search(stripped):
        return False
    return any(c.isalpha() for c in t)


def protocol_title_for_path(path: str) -> str:
    from clinical_knowledge.protocol_links import protocol_display_name

    return protocol_display_name(path or None, fallback="", registry_title=None)
