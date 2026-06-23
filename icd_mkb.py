"""
МКБ-10: извлечение кодов из текста, лексическое сопоставление ТОЛЬКО с русским справочником
(один JSON, собирается из официального Excel - см. scripts/export_icd_ru_from_xlsx.py).
Англ. названия WHO (опционально) - только для подписи к уже известному коду, не для лексикона.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ICD-10: первая буква категории + две цифры + необязательный подуровень (без U).
# Для парсинга допускаем типовые OCR/наборные варианты: пробелы, запятая/дефис/слэш.
ICD10_CODE_RE = re.compile(
    r"\b([A-TV-Z]\s*\d{2}(?:[.,/\-]\s*\d{1,4})?)\b",
    re.IGNORECASE,
)
ICD10_TERMINAL_RU_RE = re.compile(r"^[A-TV-Z]\d{2}(?:\.\d{1,4})?$", re.IGNORECASE)

# Русские PDF/протоколы: коды МКБ нередко содержат кириллические look-alike буквы
# (напр. К60, Н10, Р07) и вариативные разделители.
_ICD_CYR_TO_LAT = {
    "А": "A",
    "В": "B",
    "С": "C",
    "Е": "E",
    "Н": "H",
    "К": "K",
    "М": "M",
    "О": "O",
    "Р": "P",
    "Т": "T",
    "Х": "X",
    "У": "Y",
    "Ј": "J",
}
_ICD_CYR_LEAD = "".join(_ICD_CYR_TO_LAT.keys())
_ICD_CAND_RE = re.compile(
    rf"(?<![A-Za-zА-Яа-яЁё0-9])([A-TV-Z{_ICD_CYR_LEAD}]\s*\d{{2}}(?:\s*[.,/\-]\s*\d{{1,4}})?)(?![A-Za-zА-Яа-яЁё0-9])",
    re.IGNORECASE,
)


def _canonicalize_icd_like_token(token: str) -> str | None:
    """Нормализует похожий на ICD токен к виду K64.9; None если невалидный."""
    if not token:
        return None
    t = token.strip().upper()
    if not t:
        return None
    lead = t[0]
    lead = _ICD_CYR_TO_LAT.get(lead, lead)
    if not re.match(r"[A-TV-Z]", lead, re.IGNORECASE):
        return None
    rest = t[1:]
    rest = re.sub(r"\s+", "", rest)
    rest = rest.replace(",", ".").replace("/", ".").replace("-", ".")
    rest = re.sub(r"\.+", ".", rest).strip(".")
    if not re.match(r"^\d{2}(?:\.\d{1,4})?$", rest):
        return None
    return lead + rest


def normalize_text_for_icd_scan(text: str) -> str:
    """Нормализует типичные формы кодов для стабильного ICD-сканирования."""
    if not text:
        return text
    return _ICD_CAND_RE.sub(
        lambda m: _canonicalize_icd_like_token(m.group(1)) or m.group(1),
        text,
    )


# Слишком общие слова для лексического матчинга по названию МКБ.
# «Хронический/острый» не отфильтровываем: в рубриках J** они различают диагнозы (J01 vs J32).
_RU_STOP = frozenset(
    """
    болезнь болезни заболевание заболевания диагноз код мкб мкб-10 жалоба жалобы
    пациент пациентка симптом симптомы
    для при или без над под про как что это все всех между через
    дифференциальная гипотеза подбора протокола протокол протокола
    после
    """.split()
)

# Сами по себе слабо различают рубрику; в паре с анатомией/носологией полезны.
_RU_WEAK_ADJ = frozenset(
    """
    хронический хроническая хроническое хронические
    острый острая острое острые
    подострый подострая
    """.split()
)

_COUGH_MARKERS = ("кашел", "кашель", "сухой каш")
_FEVER_MARKERS = ("температ", "лихорад", "жар", "озноб", "субфебрил", "гипертерм")
_FUNNEL_CONTEXT_PREFIX = "контекст подбора:"

# Частые коды при ОРВИ/бронхите; выше лексического шума («сухой синдром», экзотические лихорадки A**).
_CLINICAL_ICD_HINTS: dict[str, list[tuple[str, float]]] = {
    "cough_fever": [
        ("J06.9", 22.0),
        ("J20.9", 21.5),
        ("R50.9", 21.0),
        ("R05", 20.5),
        ("J00", 19.5),
        ("J11", 19.0),
    ],
    "cough": [
        ("R05", 18.0),
        ("J06.9", 17.5),
        ("J20.9", 17.0),
    ],
    "fever": [
        ("R50.9", 17.0),
        ("J06.9", 16.5),
    ],
    "rectal_bleeding": [
        ("K64.9", 22.0),
        ("K62.5", 21.5),
        ("K92.2", 21.0),
        ("K92.1", 20.5),
        ("K62.9", 19.5),
        ("K62.6", 19.0),
    ],
    "rectal": [
        ("K64.9", 18.0),
        ("K62.9", 17.5),
        ("K62.5", 17.0),
    ],
    "stool_blood": [
        ("K92.2", 18.0),
        ("K92.1", 17.5),
        ("K92.0", 17.0),
    ],
    "hypertension": [
        ("I10", 22.0),
        ("I11.9", 21.5),
        ("R03.0", 21.0),
        ("I15.9", 20.5),
    ],
    "gastritis_gerd": [
        ("K29.7", 22.0),
        ("K21.9", 21.5),
        ("K29.5", 21.0),
        ("K29.9", 20.5),
    ],
    "dyspepsia": [
        ("K30", 22.0),
        ("K31.9", 21.5),
        ("R14", 21.0),
        ("K59.0", 20.0),
    ],
    "uti": [
        ("N39.0", 22.0),
        ("N30.0", 21.5),
        ("N30.9", 21.0),
        ("N39.9", 20.5),
    ],
    "alcohol_use": [
        ("F10.2", 22.0),
        ("F10.3", 21.5),
        ("F10.0", 21.0),
        ("F10.9", 20.5),
    ],
    "menopause": [
        ("N95.1", 22.0),
        ("N95.9", 21.5),
        ("E28.3", 20.5),
        ("R23.2", 20.0),
    ],
    "urticaria": [
        ("L50.9", 22.0),
        ("L50.0", 21.5),
        ("T78.3", 21.0),
        ("L50.1", 20.5),
    ],
    "anaphylaxis": [
        ("T78.2", 22.0),
        ("T78.4", 21.5),
        ("T78.0", 21.0),
        ("W57", 20.5),
    ],
    "thermal_burn": [
        ("T23.0", 22.0),
        ("T22.0", 21.5),
        ("T31.0", 21.0),
        ("T24.0", 20.5),
    ],
    "sore_throat": [
        ("R07.0", 22.0),
        ("J02.9", 21.5),
        ("J03.9", 21.0),
        ("J06.9", 20.0),
    ],
    "uri_throat": [
        ("J06.9", 22.0),
        ("J00", 21.5),
        ("J02.9", 21.0),
        ("J03.9", 20.5),
        ("R07.0", 20.0),
    ],
    "chest_pain": [
        ("R07.4", 22.0),
        ("I20.9", 21.5),
        ("I21.9", 21.0),
        ("R07", 20.5),
    ],
    "gastroenteritis": [
        ("A09", 22.0),
        ("K59.1", 21.5),
        ("R11", 21.0),
        ("A08.4", 20.0),
    ],
    "parkinson": [
        ("G20", 22.0),
        ("G20.9", 21.5),
        ("R25.1", 20.0),
    ],
    "gout": [
        ("M10", 22.0),
        ("M10.9", 21.5),
        ("M79.6", 20.0),
    ],
    "conjunctivitis": [
        ("H10.9", 22.0),
        ("H10", 21.5),
        ("B30.9", 20.5),
    ],
    "depression": [
        ("F32", 22.0),
        ("F32.9", 21.5),
        ("F33", 21.0),
        ("R45.2", 20.0),
    ],
    "uri_pediatric": [
        ("J06.9", 22.0),
        ("J00", 21.5),
        ("J20.9", 21.0),
        ("R50.9", 20.0),
    ],
    "asthma": [
        ("J45.9", 22.0),
        ("J45", 21.5),
        ("J46", 21.0),
    ],
    "ckd": [
        ("N18.9", 22.0),
        ("N18", 21.5),
        ("N19", 21.0),
        ("N17.9", 20.0),
    ],
    "vulvovaginitis": [
        ("N76.0", 22.0),
        ("B37.3", 21.5),
        ("N89.8", 21.0),
        ("N77.1", 20.0),
    ],
    "herpes_labialis": [
        ("B00.1", 22.0),
        ("B00.9", 21.5),
        ("B00", 21.0),
    ],
    "eye_pain": [
        ("H57.1", 22.0),
        ("H57", 21.5),
        ("H10.9", 20.5),
        ("H44.3", 20.0),
    ],
    "adenoids": [
        ("J35.2", 22.0),
        ("J35", 21.5),
        ("J35.9", 21.0),
        ("R06.5", 20.0),
    ],
    "co_poisoning": [
        ("T58.0", 22.0),
        ("T58", 21.5),
        ("X47", 21.0),
        ("T59.9", 20.0),
    ],
    "foreign_body_airway": [
        ("T17.9", 22.0),
        ("T17", 21.5),
        ("T18.0", 21.0),
        ("W79", 20.0),
    ],
}

# Коды-«последствия» и отдалённые исходы - шум при острых жалобах.
_SEQUELAE_CODE_STEMS = frozenset(
    {
        "B90",
        "B91",
        "B92",
        "B93",
        "B94",
        "E64",
        "E68",
        "G53",
        "H95",
    }
)

_RECTAL_MARKERS = ("задн", "проход", "анус", "прямок", "геморро", "переанал")
_STOOL_MARKERS = ("кал", "стул", "копр", "дефекац")
# Короткие токены (≤4) не ищем подстрокой: «кале» ≠ «раскаленными», «калечащие».
_LEX_SHORT_WORD_MAX = 4


def _norm_icd_code(s: str) -> str:
    s = (s or "").strip().upper().replace(",", ".").replace(" ", "")
    if s.endswith(".-"):
        s = s[:-2]
    return s


def normalize_icd_code(s: str) -> str:
    """Публичная нормализация кода МКБ-10 для сравнения и валидации."""
    return _norm_icd_code(s)


@lru_cache(maxsize=1)
def _load_who_rows() -> list[dict]:
    p = ROOT / "data/icd_reference/icd10_who_2016_terminal_codes.json"
    if not p.is_file():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _who_by_code() -> dict[str, dict]:
    m: dict[str, dict] = {}
    for row in _load_who_rows():
        c = (row.get("code") or "").strip().upper()
        if not c:
            continue
        m[c] = row
        n = _norm_icd_code(c)
        if n and n not in m:
            m[n] = row
    return m


def _ru_json_path() -> Path:
    rel = (os.environ.get("ICD_RU_JSON") or "").strip()
    if rel:
        pp = Path(rel)
        return pp if pp.is_absolute() else (ROOT / pp)
    return ROOT / "data/icd_reference/icd10_ru_mkb10su.json"


@lru_cache(maxsize=1)
def _ru_rows() -> list[dict]:
    p = _ru_json_path()
    if not p.is_file():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _ru_valid_codes() -> frozenset[str]:
    """Коды из единственного русского справочника (Excel → JSON)."""
    return frozenset(
        _norm_icd_code(str(r.get("code") or ""))
        for r in _ru_rows()
        if (r.get("code") or "").strip()
    )


def is_code_in_ru_reference(code: str) -> bool:
    c = _norm_icd_code(code)
    return bool(c) and c in _ru_valid_codes()


def who_title_en(code: str) -> str | None:
    c = _norm_icd_code(code)
    wb = _who_by_code()
    row = wb.get(c) or wb.get(code.upper())
    if not row:
        if len(c) == 3 and c[0].isalpha() and c[1:].isdigit():
            row = wb.get(c + ".-")
    if not row:
        return None
    return (row.get("title_en") or "").strip() or None


def ru_title(code: str) -> str | None:
    c = _norm_icd_code(code)
    for row in _ru_rows():
        if _norm_icd_code(row.get("code") or "") == c:
            t = (row.get("title_ru") or "").strip()
            return t or None
    return None


def extract_icd_codes_raw(text: str) -> list[str]:
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    scan = normalize_text_for_icd_scan(text)
    for m in ICD10_CODE_RE.finditer(scan):
        raw = m.group(1)
        n = _norm_icd_code(raw)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def extract_icd_codes_diagnosis_focused(text: str) -> list[str]:
    """Коды МКБ из строк блока «Диагноз…» до рекомендаций (приоритет для подбора протоколов)."""
    if not text:
        return []
    blob = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = blob.split("\n")
    out: list[str] = []
    seen_codes: set[str] = set()

    diag_touch_re = re.compile(
        r"^\s*(клинический\s+диагноз|диагноз\s+по\s+мкб|диагноз\s*[:\.]?)\s*",
        re.IGNORECASE,
    )
    stop_re = re.compile(
        r"^\s*(рекомендаци|назначени|назначено|назначены|лекарственн|"
        r"повторн[ауы]?.{0,24}(осмотр|осмотре|приём|явк)|заключение\s+врача)\w*\s*[:\.]",
        re.IGNORECASE,
    )

    capturing = False
    for raw_ln in lines:
        ln_st = raw_ln.strip()
        if not ln_st:
            continue
        low = ln_st.lower()

        if "закон " in low and "здравоохранен" in low:
            capturing = False
            continue

        if diag_touch_re.match(raw_ln) or (
            low.startswith("диагноз") and ":" in raw_ln[:56]
        ):
            capturing = True
        elif stop_re.match(raw_ln):
            capturing = False

        if not capturing:
            continue

        scanned = normalize_text_for_icd_scan(ln_st)
        for m in ICD10_CODE_RE.finditer(scanned):
            n = _norm_icd_code(m.group(1))
            if not n or not is_code_in_ru_reference(n):
                continue
            if n in seen_codes:
                continue
            seen_codes.add(n)
            out.append(n)
            if len(out) >= 16:
                return out

    return out


def text_mentions_icd_code(text: str, code_norm: str) -> bool:
    """True, если в тексте есть тот же код МКБ-10 или явное продолжение базы блока."""
    if not text or not code_norm:
        return False
    c = _norm_icd_code(code_norm).upper().replace(" ", "")
    if len(c) < 3:
        return False
    scan_compact = "".join(normalize_text_for_icd_scan(text).upper().split())
    if c in scan_compact:
        return True
    if "." not in c:
        return c in scan_compact
    stem, dot, sub = c.partition(".")
    if not stem:
        return False
    idx = scan_compact.find(stem)
    if idx < 0:
        return stem in scan_compact
    after = scan_compact[idx + len(stem) :]
    if after.startswith("." + sub):
        return True
    if sub and after.startswith(sub):
        return True
    return False


def count_icd_code_mentions(
    text: str,
    *,
    top_n: int = 5,
    focus_codes: list[str] | None = None,
) -> list[dict]:
    """Топ упоминаний МКБ-10 в тексте; только коды из русского справочника.

    Если задан focus_codes (коды из запроса подбора), сначала - по каждому из них
    фактическое число вхождений в текст (включая 0), затем дополнение до top_n
    по частоте остальных кодов в том же тексте.
    """
    if not text or top_n <= 0:
        return []
    counts: Counter[str] = Counter()
    scan = normalize_text_for_icd_scan(text)
    for m in ICD10_CODE_RE.finditer(scan):
        n = _norm_icd_code(m.group(1))
        if not n or not is_code_in_ru_reference(n):
            continue
        counts[n] += 1

    focus_norm: list[str] = []
    seen_fc: set[str] = set()
    for c in focus_codes or []:
        if not isinstance(c, str):
            continue
        nn = _norm_icd_code(c.strip())
        if not nn or nn in seen_fc:
            continue
        if not is_code_in_ru_reference(nn):
            continue
        seen_fc.add(nn)
        focus_norm.append(nn)

    out: list[dict] = []
    if focus_norm:
        used: set[str] = set()
        for code in focus_norm:
            cnt = int(counts.get(code, 0))
            tru = ru_title(code) or ""
            out.append(
                {
                    "code": code,
                    "count": cnt,
                    "title_ru": tru,
                    "from_query": True,
                }
            )
            used.add(code)
            if len(out) >= top_n:
                return out
        for code, cnt in counts.most_common():
            if code in used:
                continue
            tru = ru_title(code) or ""
            out.append({"code": code, "count": int(cnt), "title_ru": tru})
            if len(out) >= top_n:
                break
        return out

    if not counts:
        return []
    for code, cnt in counts.most_common(top_n):
        tru = ru_title(code) or ""
        out.append({"code": code, "count": int(cnt), "title_ru": tru})
    return out


def icd_tokens_for_lex(codes: list[str]) -> set[str]:
    """Токены для лексического RAG (латиница+цифры), чтобы «J20.9» давал j20, j20.9."""
    out: set[str] = set()
    for c in codes:
        n = _norm_icd_code(c)
        if not n:
            continue
        low = n.lower()
        out.add(low)
        if "." in low:
            base, _, rest = low.partition(".")
            out.add(base)
            out.add(base + rest.replace(".", ""))
        else:
            out.add(low)
    return {t for t in out if len(t) >= 2}


def describe_code(code: str) -> dict:
    c = _norm_icd_code(code)
    ten = who_title_en(c)
    tru = ru_title(c)
    return {
        "code": c,
        "title_ru": tru,
        "title_en": ten,
    }


def resolve_extracted_codes(codes: list[str]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for c in codes:
        n = _norm_icd_code(c)
        if not n or n in seen:
            continue
        seen.add(n)
        d = describe_code(n)
        d["match_method"] = "regex_query"
        out.append(d)
    return out


def _complaint_blob(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


def strip_funnel_context_lines(text: str) -> str:
    """Убирает служебные строки воронки («Контекст подбора: …», МКБ из шага воронки) из текста для лексикона МКБ."""
    lines: list[str] = []
    for line in (text or "").splitlines():
        low = line.strip().lower()
        if low.startswith(_FUNNEL_CONTEXT_PREFIX):
            continue
        if low.startswith("мкб-10 для поиска протокола"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def is_drug_external_cause_code(code: str) -> bool:
    """Глава XIX Y - лекарства/побочные эффекты; не диагноз при обычных жалобах."""
    c = _norm_icd_code(code)
    return bool(c) and c[0] == "Y"


def is_external_cause_code(code: str) -> bool:
    """T/X/Y/W/V - внешние причины; для подбора протокола по жалобам обычно не ведут."""
    c = _norm_icd_code(code)
    return bool(c) and c[0] in "TXYWV"


def _has_drug_adverse_context(qlow: str) -> bool:
    return any(
        x in qlow
        for x in (
            "побочн",
            "отравлен",
            "передоз",
            "передозиров",
            "лекарств",
            "препарат",
            "токсич",
            "непереносим",
            "аллерг",
        )
    )


def _has_uri_nasal_complaint(qlow: str) -> bool:
    return any(x in qlow for x in ("насморк", "ринор", "заложен нос", "чих", "орви", "орз", "простуд"))


def _has_uri_throat_complaint(qlow: str) -> bool:
    nasal = _has_uri_nasal_complaint(qlow)
    throat = _has_sore_throat_complaint(qlow) or any(
        x in qlow for x in ("горл", "глот", "фаринг", "перш")
    )
    return nasal and throat


def filter_drug_external_codes_for_complaint(codes: list[str], text: str) -> list[str]:
    """Убирает Y-коды из списка, если в жалобе нет контекста лекарств/отравления."""
    qlow = _complaint_blob(strip_funnel_context_lines(text or ""))
    if _has_drug_adverse_context(qlow):
        return list(codes)
    return [c for c in codes if not is_drug_external_cause_code(c)]


def finalize_icd_analysis_codes(icd_analysis: dict, complaint_text: str) -> None:
    """In-place: приоритет кодов болезни; Y* не ведут подбор без контекста лекарств."""
    from clinical_knowledge.diagnosis_icd import prioritize_codes

    qlow = _complaint_blob(strip_funnel_context_lines(complaint_text or ""))
    raw_codes = list(icd_analysis.get("codes_for_retrieval") or [])
    if icd_analysis.get("explicit_icd_in_query"):
        icd_analysis["codes_for_retrieval"] = prioritize_codes(raw_codes)[:10]
        return
    filtered = filter_drug_external_codes_for_complaint(raw_codes, complaint_text)
    suggested_rows = [
        s for s in (icd_analysis.get("suggested") or []) if isinstance(s, dict) and s.get("code")
    ]
    if not _has_drug_adverse_context(qlow):
        suggested_rows = [
            s for s in suggested_rows if not is_drug_external_cause_code(str(s.get("code") or ""))
        ]
        icd_analysis["suggested"] = suggested_rows
    ordered: list[str] = []
    seen: set[str] = set()
    for row in suggested_rows:
        c = _norm_icd_code(str(row.get("code") or ""))
        if c and c in filtered and c not in seen:
            seen.add(c)
            ordered.append(c)
    for c in prioritize_codes(filtered):
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    icd_analysis["codes_for_retrieval"] = ordered[:10]


def _has_cough_in_complaint(qlow: str) -> bool:
    return any(m in qlow for m in _COUGH_MARKERS)


def _has_fever_in_complaint(qlow: str) -> bool:
    if any(m in qlow for m in _FEVER_MARKERS):
        return True
    return bool(re.search(r"\b3[89]\d?\b", qlow) or re.search(r"\b40\b", qlow))


def _has_cough_fever_complaint(qlow: str) -> bool:
    has_cough = _has_cough_in_complaint(qlow)
    if not has_cough:
        return False
    return _has_fever_in_complaint(qlow) or "сух" in qlow


def _has_rectal_complaint(qlow: str) -> bool:
    if any(m in qlow for m in _RECTAL_MARKERS):
        return True
    return "шишк" in qlow and any(x in qlow for x in ("проход", "задн", "анус"))


def _has_stool_blood_complaint(qlow: str) -> bool:
    if "кров" not in qlow:
        return False
    return any(m in qlow for m in _STOOL_MARKERS) or "мелен" in qlow


def _has_rectal_bleeding_complaint(qlow: str) -> bool:
    if _has_stool_blood_complaint(qlow):
        return True
    if "кров" in qlow and _has_rectal_complaint(qlow):
        return True
    return _has_rectal_complaint(qlow) and (
        "шишк" in qlow or "воспал" in qlow or "бол" in qlow
    )


def _has_sequelae_intent(qlow: str) -> bool:
    return any(
        x in qlow
        for x in (
            "последств",
            "перенес",
            "перенесен",
            "перенесён",
            "ранее болел",
            "в анамнезе заболев",
            "отдаленн",
            "отдалённ",
        )
    )


def _is_sequelae_code(code: str) -> bool:
    cu = _norm_icd_code(code).upper()
    if len(cu) < 3:
        return False
    stem = cu[:3] if cu[1:3].isdigit() else cu[:2]
    if stem in _SEQUELAE_CODE_STEMS:
        return True
    if cu.startswith("B9") and len(cu) >= 3 and cu[1:3].isdigit():
        return True
    return False


def _is_sequelae_title(tlow: str) -> bool:
    if "последств" in tlow or "отдаленн" in tlow or "отдалённ" in tlow:
        return True
    if "после " in tlow and any(
        x in tlow
        for x in (
            "родов",
            "операц",
            "мастоид",
            "туберкул",
            "полиомиел",
            "лепр",
            "трахом",
        )
    ):
        return True
    return False


def _has_hypertension_complaint(qlow: str) -> bool:
    if re.search(r"\b1[2-9]\d/\d{2,3}\b", qlow) or re.search(r"\b20\d/\d{2,3}\b", qlow):
        return True
    if "гипертон" in qlow or "гипертенз" in qlow:
        return True
    return "давлен" in qlow and bool(re.search(r"\b\d{2,3}/\d{2,3}\b", qlow))


def _has_gastritis_gerd_complaint(qlow: str) -> bool:
    return "гастрит" in qlow or "изжог" in qlow or "гэрб" in qlow


def _has_dyspepsia_complaint(qlow: str) -> bool:
    return any(x in qlow for x in ("диспепс", "вздут", "метеоризм", "тяжесть в живот"))


def _has_uti_complaint(qlow: str) -> bool:
    if "цистит" in qlow:
        return True
    if "мочев" in qlow and "инфекц" in qlow:
        return True
    return ("жжен" in qlow or "рез" in qlow) and "моч" in qlow


def _has_pregnancy_ob_context(qlow: str) -> bool:
    return any(
        x in qlow
        for x in (
            "беремен",
            "роды",
            "послерод",
            "новорожд",
            "гестацион",
            "плацент",
            "лактац",
            "грудн",
            "после род",
        )
    )


def _has_alcohol_use_complaint(qlow: str) -> bool:
    if "алкогол" not in qlow:
        return False
    return any(
        x in qlow
        for x in ("зависим", "абстиненц", "злоупотреб", "похмель", "отмен", "синдром отмен")
    )


def _has_menopause_complaint(qlow: str) -> bool:
    return any(x in qlow for x in ("климакс", "менопауз", "прилив"))


def _has_urticaria_complaint(qlow: str) -> bool:
    return "крапивниц" in qlow or ("сып" in qlow and "зуд" in qlow)


def _has_anaphylaxis_complaint(qlow: str) -> bool:
    return "анафилакс" in qlow or ("отек квинке" in qlow.replace("ё", "е"))


def _has_thermal_burn_complaint(qlow: str) -> bool:
    if "ожог" not in qlow:
        return False
    if "солнеч" in qlow:
        return False
    return any(
        x in qlow
        for x in ("рук", "кист", "кипят", "термич", "кипяток", "пар", "пальц")
    )


def _has_sore_throat_complaint(qlow: str) -> bool:
    if "ангин" in qlow:
        return True
    return "горл" in qlow and any(x in qlow for x in ("бол", "глот", "перш", "трудно"))


def _has_chest_pain_complaint(qlow: str) -> bool:
    return ("груд" in qlow or "загрудин" in qlow) and "бол" in qlow


def _has_gastroenteritis_complaint(qlow: str) -> bool:
    if "понос" in qlow or "диаре" in qlow:
        return "рвот" in qlow or "инфекц" in qlow or "кишечн" in qlow
    return "рвот" in qlow and "кишечн" in qlow


def _has_parkinson_complaint(qlow: str) -> bool:
    return "паркинсон" in qlow or ("тремор" in qlow and "ригид" in qlow)


def _has_gout_complaint(qlow: str) -> bool:
    return "подагр" in qlow


def _has_conjunctivitis_complaint(qlow: str) -> bool:
    return "конъюнктивит" in qlow or (
        "глаз" in qlow and any(x in qlow for x in ("покрасн", "конъюнкт", "слезотеч"))
    )


def _has_depression_complaint(qlow: str) -> bool:
    return "депресс" in qlow or ("апат" in qlow and "сонлив" in qlow)


def _has_pediatric_uri_complaint(qlow: str) -> bool:
    if not any(x in qlow for x in ("ребен", "ребён", "дет", "грудн", "младен")):
        return False
    return any(x in qlow for x in ("орви", "насмор", "кашел", "кашель"))


def _has_asthma_complaint(qlow: str) -> bool:
    return "астм" in qlow or ("удуш" in qlow and "кашел" in qlow)


def _has_ckd_complaint(qlow: str) -> bool:
    if "хронич" in qlow and "почек" in qlow:
        return True
    return ("почек" in qlow or "нефропат" in qlow) and "отек" in qlow


def _has_vulvovaginitis_complaint(qlow: str) -> bool:
    if "вульвовагинит" in qlow:
        return True
    return "зуд" in qlow and "выделен" in qlow and any(x in qlow for x in ("вульв", "вагин"))


def _has_herpes_labialis_complaint(qlow: str) -> bool:
    return "герпес" in qlow and any(x in qlow for x in ("губ", "пузырьк", "губе"))


def _has_eye_pain_complaint(qlow: str) -> bool:
    return "глаз" in qlow and any(x in qlow for x in ("бол", "светобоязн", "боль"))


def _has_adenoids_complaint(qlow: str) -> bool:
    return "аденоид" in qlow or ("храп" in qlow and any(x in qlow for x in ("ребен", "дет", "грудн")))


def _has_co_poisoning_complaint(qlow: str) -> bool:
    return "угарн" in qlow or ("угар" in qlow and "газ" in qlow)


def _has_foreign_body_airway_complaint(qlow: str) -> bool:
    return "инородн" in qlow and any(x in qlow for x in ("дыхат", "трахе", "бронх", "гортан"))


def _is_family_history_title(tlow: str) -> bool:
    return any(
        x in tlow
        for x in (
            "семейном анамнезе",
            "в семейном анамнезе",
            "семейный анамнез",
            "наследственн",
        )
    )


def _is_insect_bite_complaint(qlow: str) -> bool:
    return any(x in qlow for x in ("пчел", "ос", "шершен", "комар", "клещ", "насеком"))


def _penalize_sequelae_for_acute(qlow: str, code: str, title: str, score: float) -> float:
    if _has_sequelae_intent(qlow):
        return score
    tlow = title.lower().replace("ё", "е")
    if _is_sequelae_code(code) or _is_sequelae_title(tlow):
        return score * 0.05
    return score


def _penalize_compression_for_bp(qlow: str, code: str, title: str, score: float) -> float:
    if not _has_hypertension_complaint(qlow):
        return score
    tlow = title.lower().replace("ё", "е")
    cu = code.upper()
    if "сдавлен" in tlow or "удушен" in tlow or "удавлен" in tlow:
        return score * 0.04
    if cu.startswith("G93") or cu.startswith("G95"):
        return score * 0.06
    if cu.startswith("W") and "давлен" in tlow:
        return score * 0.05
    if cu.startswith("Y") and ("удуш" in tlow or "удавл" in tlow):
        return score * 0.05
    return score


def _penalize_obstetric_without_context(qlow: str, code: str, score: float) -> float:
    if _has_pregnancy_ob_context(qlow):
        return score
    cu = code.upper()
    if not cu.startswith(("O", "P")):
        return score
    if _has_uti_complaint(qlow) and cu.startswith("O"):
        return score * 0.06
    if cu.startswith("P"):
        return score * 0.08
    return score


def _penalize_family_history_z(qlow: str, code: str, title: str, score: float) -> float:
    cu = code.upper()
    if not cu.startswith("Z"):
        return score
    tlow = title.lower().replace("ё", "е")
    if not _is_family_history_title(tlow):
        return score
    if _has_alcohol_use_complaint(qlow):
        return score * 0.04
    return score * 0.35


def _penalize_exotic_bite_fever(qlow: str, code: str, title: str, score: float) -> float:
    cu = code.upper()
    if not cu.startswith("A"):
        return score
    tlow = title.lower().replace("ё", "е")
    if "укус" not in tlow:
        return score
    if _is_insect_bite_complaint(qlow) and "крыс" not in qlow:
        return score * 0.05
    return score


def _penalize_sunburn_for_thermal(qlow: str, code: str, score: float) -> float:
    if not _has_thermal_burn_complaint(qlow):
        return score
    cu = code.upper()
    if cu.startswith("L55"):
        return score * 0.06
    return score


def _penalize_z_for_acute_complaint(qlow: str, code: str, title: str, score: float) -> float:
    cu = code.upper()
    if not cu.startswith("Z"):
        return score
    if _has_sequelae_intent(qlow) or "скрининг" in qlow:
        return score
    tlow = title.lower().replace("ё", "е")
    if _is_family_history_title(tlow):
        return score * 0.35
    if any(x in tlow for x in ("отлучен", "проблем", "наблюден", "профилакт")):
        return score * 0.06
    return score * 0.4


def _penalize_infant_death_for_uri(qlow: str, code: str, score: float) -> float:
    if code.upper() != "R95":
        return score
    if any(x in qlow for x in ("орви", "насмор", "кашел", "кашель", "ринит")):
        return score * 0.02
    return score


def _penalize_sti_pharynx_without_context(qlow: str, code: str, title: str, score: float) -> float:
    cu = code.upper()
    if not cu.startswith("A"):
        return score
    tlow = title.lower().replace("ё", "е")
    if "фаринг" in qlow and "гонокок" in tlow:
        return score * 0.04
    return score


def _penalize_poisoning_without_intent(qlow: str, code: str, title: str, score: float) -> float:
    cu = code.upper()
    if not cu.startswith("T") or "отравлен" not in title.lower():
        return score
    if "отравлен" in qlow or "яд" in qlow:
        return score
    return score * 0.08


def _penalize_drug_complication_y(qlow: str, code: str, score: float) -> float:
    cu = code.upper()
    if not cu.startswith("Y"):
        return score
    if _has_asthma_complaint(qlow) or _has_cough_fever_complaint(qlow):
        return score * 0.04
    return score * 0.25


def _clinical_hint_profile(qlow: str) -> str | None:
    """Определяет профиль клинической подсказки (приоритет - более специфичные)."""
    if _has_anaphylaxis_complaint(qlow):
        return "anaphylaxis"
    if _has_foreign_body_airway_complaint(qlow):
        return "foreign_body_airway"
    if _has_co_poisoning_complaint(qlow):
        return "co_poisoning"
    if _has_thermal_burn_complaint(qlow):
        return "thermal_burn"
    if _has_adenoids_complaint(qlow):
        return "adenoids"
    if _has_pediatric_uri_complaint(qlow):
        return "uri_pediatric"
    if _has_uri_throat_complaint(qlow):
        return "uri_throat"
    if _has_asthma_complaint(qlow):
        return "asthma"
    if _has_cough_fever_complaint(qlow):
        return "cough_fever"
    if _has_chest_pain_complaint(qlow):
        return "chest_pain"
    if _has_sore_throat_complaint(qlow):
        return "sore_throat"
    if _has_rectal_bleeding_complaint(qlow):
        return "rectal_bleeding"
    if _has_hypertension_complaint(qlow):
        return "hypertension"
    if _has_alcohol_use_complaint(qlow):
        return "alcohol_use"
    if _has_menopause_complaint(qlow):
        return "menopause"
    if _has_uti_complaint(qlow) and not _has_pregnancy_ob_context(qlow):
        return "uti"
    if _has_gastritis_gerd_complaint(qlow):
        return "gastritis_gerd"
    if _has_dyspepsia_complaint(qlow):
        return "dyspepsia"
    if _has_gastroenteritis_complaint(qlow):
        return "gastroenteritis"
    if _has_urticaria_complaint(qlow):
        return "urticaria"
    if _has_conjunctivitis_complaint(qlow):
        return "conjunctivitis"
    if _has_parkinson_complaint(qlow):
        return "parkinson"
    if _has_gout_complaint(qlow):
        return "gout"
    if _has_depression_complaint(qlow):
        return "depression"
    if _has_ckd_complaint(qlow):
        return "ckd"
    if _has_vulvovaginitis_complaint(qlow):
        return "vulvovaginitis"
    if _has_herpes_labialis_complaint(qlow):
        return "herpes_labialis"
    if _has_eye_pain_complaint(qlow):
        return "eye_pain"
    if _has_cough_in_complaint(qlow):
        return "cough"
    if _has_fever_in_complaint(qlow):
        return "fever"
    if _has_rectal_complaint(qlow):
        return "rectal"
    if _has_stool_blood_complaint(qlow):
        return "stool_blood"
    return None


def clinical_hints_confident(text: str, *, min_score: float = 19.0) -> bool:
    """True, если клинические подсказки дают уверенный top-1 без Gemini."""
    qlow = _complaint_blob(strip_funnel_context_lines(text or ""))
    profile = _clinical_hint_profile(qlow)
    if not profile:
        return False
    hints = _CLINICAL_ICD_HINTS.get(profile) or []
    if not hints:
        return False
    return float(hints[0][1]) >= min_score


def filter_icd_pool_for_complaint(scored: list[dict], text: str) -> list[dict]:
    """Предфильтр пула для Gemini: убирает явный лексический шум по типу жалобы."""
    lq = strip_funnel_context_lines(text or "")
    qlow = _complaint_blob(lq)
    if not qlow:
        return list(scored)
    out: list[dict] = []
    for row in scored:
        code = str(row.get("code") or "")
        title = str(row.get("title_ru") or "")
        tlow = title.lower().replace("ё", "е")
        sc = float(row.get("lex_score") or row.get("score") or 0)
        if row.get("match_method") == "clinical_hint":
            out.append(row)
            continue
        if sc <= 0:
            continue
        if not _has_sequelae_intent(qlow) and (
            _is_sequelae_code(code) or _is_sequelae_title(tlow)
        ):
            continue
        if _has_hypertension_complaint(qlow) and (
            "сдавлен" in tlow or (code.upper().startswith("W") and "давлен" in tlow)
        ):
            continue
        if _has_alcohol_use_complaint(qlow) and _is_family_history_title(tlow):
            continue
        if _has_uti_complaint(qlow) and not _has_pregnancy_ob_context(qlow):
            cu = code.upper()
            if cu.startswith("P") or (
                cu.startswith("O") and "мочев" in tlow and "беремен" not in tlow
            ):
                continue
        if _is_insect_bite_complaint(qlow) and code.upper().startswith("A") and "крыс" in tlow:
            continue
        if not _has_drug_adverse_context(qlow) and is_drug_external_cause_code(code):
            continue
        out.append(row)
    return out if out else list(scored)


def _lex_word_in_title(word: str, tlow: str) -> bool:
    """Совпадение слова в названии МКБ; короткие токены - только целым словом."""
    if len(word) > _LEX_SHORT_WORD_MAX:
        return word in tlow
    pat = re.compile(rf"(?<![а-яё]){re.escape(word)}(?![а-яё])", re.IGNORECASE)
    return bool(pat.search(tlow))


def _penalize_external_cause_for_clinical(qlow: str, code: str, score: float) -> float:
    """Снижает T/X/Y/Z при явной клинической жалобе (не травма/осложнение лечения)."""
    cu = code.upper()
    if not cu or cu[0] not in "TXYZ":
        return score
    clinical_acute = (
        _has_rectal_bleeding_complaint(qlow)
        or _has_cough_fever_complaint(qlow)
        or _has_sore_throat_complaint(qlow)
        or _has_uri_throat_complaint(qlow)
        or _has_uri_nasal_complaint(qlow)
        or _has_cough_in_complaint(qlow)
        or _has_fever_in_complaint(qlow)
    )
    if clinical_acute and not _has_drug_adverse_context(qlow):
        return score * 0.05
    if _has_rectal_bleeding_complaint(qlow) or _has_cough_fever_complaint(qlow):
        return score * 0.08
    return score


def _clinical_icd_hint_rows(text: str) -> list[dict]:
    """Приоритетные коды по типовым сочетаниям жалоб."""
    qlow = _complaint_blob(text)
    if len(qlow) < 3:
        return []
    profile = _clinical_hint_profile(qlow)
    if not profile:
        return []
    out: list[dict] = []
    for code, score in _CLINICAL_ICD_HINTS[profile]:
        info = describe_code(code)
        if not info.get("title_ru"):
            continue
        out.append(
            {
                "code": code,
                "title_ru": info.get("title_ru"),
                "title_en": info.get("title_en"),
                "lex_score": score,
                "match_method": "clinical_hint",
            }
        )
    return out


def merge_clinical_icd_hints(scored: list[dict], text: str) -> list[dict]:
    """Подмешивает клинические подсказки в начало лексического пула (для Gemini и suggest)."""
    hints = _clinical_icd_hint_rows(text)
    if not hints:
        return list(scored)
    by_code = {str(x.get("code") or ""): dict(x) for x in scored if x.get("code")}
    merged: list[dict] = []
    seen: set[str] = set()
    for row in hints + scored:
        code = str(row.get("code") or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        base = by_code.get(code, {})
        item = dict(base)
        item.update(row)
        if row.get("match_method") == "clinical_hint":
            item["match_method"] = "clinical_hint"
        merged.append(item)
    return merged


def _icd_extra_roots(text: str) -> list[str]:
    """Подсказки к МКБ: разговорное «гайморит» ↔ формулировки справочника («верхнечелюстной»)."""
    s = text.lower().replace("ё", "е")
    extra: list[str] = []
    if "гаймор" in s or "гайморит" in s:
        extra.extend(["верхнечелюстн", "верхнечелюстной", "гайморова"])
    if "риносинусит" in s or "синусит" in s:
        extra.append("синусит")
    if "кровь" in s and any(m in s for m in _STOOL_MARKERS):
        extra.extend(["мелен", "кровотеч", "желудочно-кишечн"])
    if ("задн" in s and "проход" in s) or "геморро" in s:
        extra.extend(["заднего прохода", "прямой киш", "геморро"])
    if "шишк" in s and ("проход" in s or "задн" in s or "геморро" in s):
        extra.extend(["геморро", "геморроид"])
    if "гипертон" in s or "гипертенз" in s or re.search(r"\d{2,3}/\d{2,3}", s):
        extra.extend(["гипертенз", "артериальн", "эссенциальн"])
    if "климакс" in s or "менопауз" in s or "прилив" in s:
        extra.extend(["менопауз", "климактер", "прилив"])
    if "цистит" in s or ("мочев" in s and "инфекц" in s):
        extra.extend(["мочевых пут", "цистит", "уретрит"])
    if "алкогол" in s and ("зависим" in s or "абстиненц" in s):
        extra.extend(["алкогольн", "зависимост"])
    if "крапивниц" in s:
        extra.append("крапивница")
    if "анафилакс" in s:
        extra.extend(["анафилакт", "аллергическ"])
    if "ожог" in s and "солнеч" not in s:
        extra.extend(["термическ", "ожог"])
    return extra


def _ru_words(text: str) -> list[str]:
    s = text.lower().replace("ё", "е")
    words = [w for w in re.findall(r"[а-яё]{3,}", s) if w not in _RU_STOP]
    for r in _icd_extra_roots(s):
        if r not in words and len(r) >= 3:
            words.append(r)
    return words


def _lexicon_score_one_row(
    words: list[str], qlow: str, code: str, title: str
) -> float:
    tlow = title.lower().replace("ё", "е")
    score = 0.0
    for w in words:
        if not _lex_word_in_title(w, tlow):
            continue
        if w in _RU_WEAK_ADJ:
            score += 1.0
            continue
        if len(w) >= 5:
            score += 3.0
        elif len(w) == 4:
            score += 2.0
        else:
            score += 1.0
    if len(qlow) >= 6 and qlow in tlow:
        score += 8.0
    if words and len(tlow) <= 48:
        hit = sum(1 for w in words if _lex_word_in_title(w, tlow))
        if hit >= 2:
            score += 2.5
    if len(tlow) > 55:
        score -= min(4.0, (len(tlow) - 55) * 0.06)
    if (
        _has_cough_fever_complaint(qlow)
        and code.upper().startswith("A")
        and "лихорад" in tlow
    ):
        score *= 0.12
    if _has_rectal_bleeding_complaint(qlow) and code.upper().startswith("T") and "инородн" in tlow:
        score *= 0.15
    score = _penalize_external_cause_for_clinical(qlow, code, score)
    score = _penalize_sequelae_for_acute(qlow, code, title, score)
    score = _penalize_compression_for_bp(qlow, code, title, score)
    score = _penalize_obstetric_without_context(qlow, code, score)
    score = _penalize_family_history_z(qlow, code, title, score)
    score = _penalize_exotic_bite_fever(qlow, code, title, score)
    score = _penalize_sunburn_for_thermal(qlow, code, score)
    score = _penalize_z_for_acute_complaint(qlow, code, title, score)
    score = _penalize_infant_death_for_uri(qlow, code, score)
    score = _penalize_sti_pharynx_without_context(qlow, code, title, score)
    score = _penalize_poisoning_without_intent(qlow, code, title, score)
    score = _penalize_drug_complication_y(qlow, code, score)
    if score <= 0:
        return 0.0
    # Раньше: +1.8 ко всем *.9 без контекста - тянуло «неуточнённ» коды из-за частицы «для» и т.п.
    if code.endswith(".9") and score >= 4.5 and not any(
        x in qlow
        for x in ("mycoplasma", "микоплазм", "вирус", "бактер", "стрептококк", "гемофил")
    ):
        score += 1.2
    return score


def _ru_lexicon_scored_entries_uncached(words: list[str], qlow: str) -> list[dict]:
    """Внутренний скоринг без кэша (words и qlow уже нормализованы)."""
    best: dict[str, tuple[float, str, str]] = {}
    for row in _ru_rows():
        code = (row.get("code") or "").strip()
        if not ICD10_TERMINAL_RU_RE.match(code):
            continue
        title = (row.get("title_ru") or "").strip()
        if not title:
            continue
        sc = _lexicon_score_one_row(words, qlow, code, title)
        if sc <= 0:
            continue
        n = _norm_icd_code(code)
        prev = best.get(n)
        if prev is None or sc > prev[0]:
            best[n] = (sc, n, title)
    out: list[dict] = []
    for sc, code, title in sorted(best.values(), key=lambda x: -x[0]):
        ten = who_title_en(code)
        out.append(
            {
                "code": code,
                "title_ru": title,
                "title_en": ten,
                "lex_score": round(sc, 2),
            }
        )
    return out


@lru_cache(maxsize=1024)
def _ru_lexicon_cache_key(text: str) -> tuple[tuple[str, ...], str]:
    """Ключ кэша: нормализованные слова + qlow."""
    words = tuple(_ru_words(text))
    qlow = text.lower().replace("ё", "е").strip()
    return words, qlow


def ru_lexicon_scored_entries(text: str) -> list[dict]:
    """
    Все коды с положительным лексическим score, по убыванию score;
    на каждый код - строка с максимальным score (агрегация по коду).
    """
    text = strip_funnel_context_lines(text or "")
    if not text or len(text.strip()) < 3:
        return []
    words, qlow = _ru_lexicon_cache_key(text)
    return _ru_lexicon_scored_entries_uncached(list(words), qlow)


def clear_ru_lexicon_cache() -> None:
    """Сброс LRU-кэша лексикона (тесты / hot reload)."""
    _ru_lexicon_cache_key.cache_clear()


def suggest_icd_from_russian(text: str, max_results: int = 8) -> list[dict]:
    """Лексическое сопоставление запроса с русскими названиями МКБ (без LLM)."""
    text = strip_funnel_context_lines(text or "")
    if not text or len(text.strip()) < 3:
        return []
    scored_rows = ru_lexicon_scored_entries(text)
    scored: list[tuple[float, str, str]] = [
        (float(r.get("lex_score") or 0), str(r["code"]), str(r.get("title_ru") or ""))
        for r in scored_rows
        if r.get("code")
    ]

    def _stem(c: str) -> str:
        c = _norm_icd_code(c)
        if len(c) >= 3 and c[0].isalpha() and c[1:3].isdigit():
            return c[:3].upper()
        return c

    out: list[dict] = []
    seen: set[str] = set()
    stems_used: set[str] = set()

    for hint in _clinical_icd_hint_rows(text):
        n = _norm_icd_code(str(hint.get("code") or ""))
        if not n or n in seen:
            continue
        st = _stem(n)
        if st in stems_used:
            continue
        seen.add(n)
        stems_used.add(st)
        out.append(
            {
                "code": n,
                "title_ru": hint.get("title_ru"),
                "title_en": hint.get("title_en"),
                "match_method": "clinical_hint",
                "score": round(float(hint.get("lex_score") or 0), 2),
            }
        )

    for score, code, title in scored:
        n = _norm_icd_code(code)
        if n in seen:
            continue
        st = _stem(n)
        if st in stems_used:
            continue
        seen.add(n)
        stems_used.add(st)
        ten = who_title_en(n)
        out.append(
            {
                "code": n,
                "title_ru": title,
                "title_en": ten,
                "match_method": "lexicon_ru",
                "score": round(score, 2),
            }
        )
        if len(out) >= max_results:
            break
    return out


def analyze_query_for_icd(
    full_query: str,
    rag_query: str,
    *,
    lexicon_query: str | None = None,
) -> dict:
    """
    Объединяет: коды из полного запроса и RAG-части + лексические гипотезы по русскому тексту.

    Если в тексте явно указаны коды МКБ - используются только они (после проверки по русскому
    справочнику из Excel); лексические гипотезы не подмешиваются.

    lexicon_query: исходная жалоба до LLM-уточнения (чтобы «температура» не превращалась в «лихорадка»
    только для подбора МКБ).
    """
    combined = f"{full_query}\n{rag_query}"
    extracted = extract_icd_codes_raw(combined)
    detected_raw = resolve_extracted_codes(extracted)

    detected_valid: list[dict] = []
    detected_unknown: list[dict] = []
    for d in detected_raw:
        c = normalize_icd_code(str(d.get("code") or ""))
        if not c:
            continue
        if is_code_in_ru_reference(c):
            detected_valid.append(d)
        else:
            detected_unknown.append({**d, "code": c, "match_method": "regex_query_unknown"})

    # Явные коды в тексте, но ни один не из справочника - откатываемся к лексикону.
    explicit_codes_in_query = bool(detected_raw)
    if explicit_codes_in_query and not detected_valid and detected_unknown:
        explicit_codes_in_query = False

    suggested: list[dict] = []
    lq = strip_funnel_context_lines((lexicon_query or rag_query or full_query).strip())
    if lq and not explicit_codes_in_query:
        suggested = suggest_icd_from_russian(lq, max_results=8)

    det_set = {d["code"] for d in detected_valid}
    suggested = [s for s in suggested if s["code"] not in det_set]

    if explicit_codes_in_query:
        codes_for_retrieval = [d["code"] for d in detected_valid][:10]
    else:
        codes_for_retrieval = list(
            dict.fromkeys(
                [d["code"] for d in detected_valid]
                + [s["code"] for s in suggested[:8]]
            )
        )[:10]

    out = {
        "detected": detected_valid,
        "detected_unknown": detected_unknown,
        "suggested": suggested,
        "codes_for_retrieval": codes_for_retrieval,
        "explicit_icd_in_query": explicit_codes_in_query,
    }
    finalize_icd_analysis_codes(out, lq)
    return out
