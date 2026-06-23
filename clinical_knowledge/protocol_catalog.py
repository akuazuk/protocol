"""Единый каталог PDF протоколов: МКБ, аудитория, тип (общий/клинический)."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "protocol_catalog.jsonl"
OVERRIDES_PATH = ROOT / "data" / "protocol_audience_overrides.json"
INDEX_CSV = ROOT / "index.csv"
CARDS_PATH = ROOT / "output" / "registry" / "protocol_cards.jsonl"
CHUNKS_PATH = ROOT / "output" / "chunks" / "chunks.jsonl"
SUMMARY_JSON_DIR = ROOT / "data" / "protocol_summaries" / "json"
ICD_REF_PATH = ROOT / "data" / "icd_reference" / "icd10_who_2016_terminal_codes.json"
PRIMARY_ICD_LIMIT = 12
BODY_TEXT_LIMIT = 80_000

_CLASSIFICATION_CHUNK_TYPES = frozenset(
    {"classification", "diagnostics", "criteria_block"}
)
_DRUG_NOISE_CHUNK_TYPES = frozenset(
    {"drug_list", "pharmacotherapy", "appendix", "table_block"}
)
_EXTERNAL_ICD_LETTERS = frozenset("YWVT")

_SECTION_HEADER_RE = re.compile(
    r"(?m)^(?:ГЛАВА\s+\d+|(?:\d+\.)+\s*[А-ЯЁA-Z]|"
    r"(?:ДИАГНОСТИКА|ЛЕЧЕНИЕ|ПРОФИЛАКТИКА|РЕАБИЛИТАЦИЯ|ДИСПАНСЕРНОЕ|"
    r"МАРШРУТИЗАЦИЯ|ФАРМАКОТЕРАПИЯ|КЛАССИФИКАЦИЯ|ПРИЛОЖЕНИЕ|АЛГОРИТМ"
    r"|ТЕРМИНЫ|ОБЩИЕ\s+ПОЛОЖЕНИЯ|ПОКАЗАНИЯ|КРИТЕРИИ))",
)

_PED_SLUGS = frozenset({"pediatriya"})
_PED_MARKERS = (
    "д-нас", "дет-нас", "дет нас", "дет_нас", "детс", "дет. нас", "детск",
    "детей", " дет", "неонат", "новорожд", "pediatr", "дет возраста",
)
_ADULT_MARKERS = (
    "взросл", "взр ", "взр.", "взр_нас", "в-нас", " в нас", "вз н", "вз_н",
)
_BODY_PED = (
    "детское население", "д-нас", "дет-нас", "детей", "детск", "новорожд",
    "неонатолог", "грудн", "пedi",
)
_BODY_ADULT = (
    "взрослое население", "в-нас", "взр. нас", "взрослых", "взросл",
)
_BODY_PREG = ("беременн", "родильниц", "акушерск", "пrenatal", "пренаталь")


def _norm_path(sp: str) -> str:
    return (sp or "").replace("\\", "/").strip()


def _norm_icd(code: str) -> str:
    return re.sub(r"\s+", "", (code or "").upper().strip())


def _norm_blob(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("_", " ")).strip()


def _infer_audience_from_text(path: str, title: str) -> str | None:
    blob = _norm_blob(f"{path} {title}")
    has_p = any(m in blob for m in _PED_MARKERS)
    has_a = any(m in blob for m in _ADULT_MARKERS)
    if has_p and has_a:
        return "mixed"
    if has_p:
        return "pediatric"
    if has_a:
        return "adult"
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@lru_cache(maxsize=1)
def _valid_icd_codes() -> frozenset[str]:
    if not ICD_REF_PATH.is_file():
        return frozenset()
    try:
        data = json.loads(ICD_REF_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    codes: set[str] = set()
    for row in data if isinstance(data, list) else []:
        c = _norm_icd(str(row.get("code") or ""))
        if c:
            codes.add(c)
    return frozenset(codes)


def normalize_protocol_title(title: str, filename: str = "") -> str:
    """Единое читаемое название без даты, № постановления и префикса «КП»."""
    raw = (title or filename or "").strip()
    s = raw.replace("_", " ").replace("-", " ")
    s = re.sub(r"\.pdf$", "", s, flags=re.I)
    s = re.sub(r"^(?:кп|клинический\s+протокол)\s*[«\"']?", "", s, flags=re.I)
    s = re.sub(r"\s*пост\.?\s*мз.*$", "", s, flags=re.I)
    s = re.sub(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}(?:\s*г\.?)?", " ", s)
    s = re.sub(r"№\s*[\d\-а-яА-ЯёЁ/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" «»\"'")
    return (s[:280] if s else raw[:280])


def compute_icd10_relevance_weights(
    *,
    icd_primary: list[str],
    icd_all: list[str],
    title: str,
    body: str,
    chunk_freq: Counter[str] | None = None,
) -> dict[str, int]:
    """Рейтинг соответствия кода МКБ протоколу, 0-100 % (primary выше secondary)."""
    weights: dict[str, int] = {}
    title_blob = _norm_blob(title)
    body_blob = _norm_blob(body[:40_000])
    primary_set = {_norm_icd(c) for c in icd_primary if _norm_icd(c)}
    freq = chunk_freq or Counter()
    max_freq = max(freq.values()) if freq else 1

    for i, raw in enumerate(icd_primary[:PRIMARY_ICD_LIMIT]):
        code = _norm_icd(raw)
        if not code:
            continue
        pct = 94 - min(22, i * 2)
        if code.lower() in title_blob or code.replace(".", "").lower() in title_blob:
            pct = min(100, pct + 6)
        if code.lower() in body_blob[:12_000]:
            pct = min(100, pct + 4)
        weights[code] = max(weights.get(code, 0), pct)

    for code, cnt in freq.most_common(28):
        nc = _norm_icd(code)
        if not nc or nc in weights:
            continue
        if nc not in {_norm_icd(x) for x in icd_all}:
            continue
        pct = int(32 + 38 * (cnt / max(max_freq, 1)))
        if cnt >= 2:
            pct += 8
        if nc.lower() in title_blob:
            pct = min(88, pct + 10)
        weights[nc] = min(88, pct)

    for raw in icd_all:
        nc = _norm_icd(raw)
        if not nc or nc in weights:
            continue
        weights[nc] = 28

    return dict(sorted(weights.items(), key=lambda x: (-x[1], x[0]))[:36])


def extract_content_tags(
    title: str,
    body: str,
    specialty_slug: str,
    protocol_kind: str,
    audience: str,
) -> list[str]:
    """Тематические метки протокола для поиска и overview-чанка."""
    blob = _norm_blob(f"{title} {body[:12_000]}")
    tags: list[str] = []
    if protocol_kind:
        tags.append(f"kind:{protocol_kind}")
    if specialty_slug:
        tags.append(f"rubric:{specialty_slug}")
    if audience and audience != "any":
        tags.append(f"audience:{audience}")
    markers = (
        ("diagnostics", ("диагност", "обследован", "критери")),
        ("treatment", ("лечен", "терапи", "операц")),
        ("prevention", ("профилакт",)),
        ("rehabilitation", ("реабилит",)),
        ("pharmacotherapy", ("фармако", "препарат")),
        ("routing", ("маршрутизац", "госпитализац")),
        ("classification", ("классификац", "мкб", "шифр")),
        ("criteria", ("показан", "противопоказ")),
    )
    for tag, needles in markers:
        if any(n in blob for n in needles):
            tags.append(tag)
    return tags[:14]


def extract_section_tags_from_text(body: str) -> list[str]:
    """Крупные разделы из текста PDF (заголовки глав/блоков)."""
    tags: list[str] = []
    seen: set[str] = set()
    pat = re.compile(
        r"(?mi)^(?:ГЛАВА\s+\d+|(?:ДИАГНОСТИКА|ЛЕЧЕНИЕ|ПРОФИЛАКТИКА|РЕАБИЛИТАЦИЯ|"
        r"ДИСПАНСЕРНОЕ|МАРШРУТИЗАЦИЯ|ФАРМАКОТЕРАПИЯ|КЛАССИФИКАЦИЯ|ПОКАЗАНИЯ|КРИТЕРИИ))"
    )
    for m in pat.finditer(body or ""):
        s = re.sub(r"\s+", " ", m.group(0)).strip()[:80]
        low = s.lower()
        if s and low not in seen:
            seen.add(low)
            tags.append(s)
        if len(tags) >= 12:
            break
    return tags


def _filter_valid_icd(codes: list[str]) -> list[str]:
    valid = _valid_icd_codes()
    roots = {c[:3] for c in valid if len(c) >= 3}
    out: list[str] = []
    seen: set[str] = set()
    for raw in codes:
        c = _norm_icd(raw)
        if not c or len(c) < 3 or c in seen:
            continue
        if valid:
            if c in valid or c[:3] in roots:
                seen.add(c)
                out.append(c)
            continue
        seen.add(c)
        out.append(c)
    return out


def _load_overrides() -> dict[str, str]:
    if not OVERRIDES_PATH.is_file():
        return {}
    try:
        data = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {_norm_path(k): str(v) for k, v in data.items() if v}


def _cards_by_path() -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for card in _read_jsonl(CARDS_PATH):
        sp = _norm_path(str(card.get("source_path") or ""))
        if not sp:
            continue
        if sp not in merged:
            merged[sp] = dict(card)
            continue
        icd = set(merged[sp].get("icd10_all") or [])
        icd |= {str(x) for x in (card.get("icd10_all") or []) if x}
        merged[sp]["icd10_all"] = sorted(icd)
        if not merged[sp].get("icd10_primary") and card.get("icd10_primary"):
            merged[sp]["icd10_primary"] = card.get("icd10_primary")
    return merged


def _is_external_cause_icd(code: str) -> bool:
    c = _norm_icd(code)
    return bool(c) and c[0] in _EXTERNAL_ICD_LETTERS


def _section_title_is_classification(title: str) -> bool:
    t = _norm_blob(title)
    return any(
        k in t
        for k in (
            "классификац",
            "шифр мкб",
            "код мкб",
            "мкб-10",
            "диагноз",
            "диагностическ критери",
            "номенклатур",
        )
    )


def _section_title_is_drug_noise(title: str) -> bool:
    t = _norm_blob(title)
    return any(
        k in t
        for k in (
            "фармакотерап",
            "лекарствен",
            "перечень лекарств",
            "побочн",
            "приложение",
        )
    )


def _split_body_sections(body: str) -> list[tuple[str, str]]:
    """Разбивка текста PDF по заголовкам разделов: (заголовок, текст)."""
    text = (body or "").strip()
    if not text:
        return []
    matches = list(_SECTION_HEADER_RE.finditer(text))
    if not matches:
        return [("", text)]
    out: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        out.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        title = re.sub(r"\s+", " ", m.group(0)).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            out.append((title, chunk))
    return out


def _icd_from_classification_sections(
    body: str,
    *,
    extract_icd10,
    lookup_disease_icd,
    prioritize_codes,
    is_symptom_code,
) -> tuple[list[str], list[str]]:
    """МКБ только из блоков «Классификация / диагноз / критерии» в тексте PDF."""
    icd_set: set[str] = set()
    sources: list[str] = []
    for title, blob in _split_body_sections(body):
        if _section_title_is_drug_noise(title):
            continue
        is_class = _section_title_is_classification(title) or _norm_blob(title).startswith(
            "диагностик"
        )
        if not is_class and title:
            continue
        part = blob[:16_000]
        for code in lookup_disease_icd(part):
            c = _norm_icd(code)
            if c and not _is_external_cause_icd(c):
                icd_set.add(c)
                sources.append("body_classification_lex")
        for code in extract_icd10(part):
            c = _norm_icd(code)
            if c and not _is_external_cause_icd(c) and not is_symptom_code(c):
                icd_set.add(c)
                sources.append("body_classification_extract")
    return prioritize_codes(sorted(icd_set)), sorted(set(sources))


def _chunk_data_by_path() -> tuple[
    dict[str, list[str]],
    dict[str, Counter[str]],
    dict[str, list[str]],
    dict[str, Counter[str]],
    dict[str, list[str]],
    dict[str, set[str]],
]:
    """path -> texts, icd freq (без drug), icd lists, class freq, class icd, drug-only icd."""
    texts: dict[str, list[str]] = defaultdict(list)
    freq: dict[str, Counter[str]] = defaultdict(Counter)
    field_icd: dict[str, list[str]] = defaultdict(list)
    class_freq: dict[str, Counter[str]] = defaultdict(Counter)
    class_field: dict[str, list[str]] = defaultdict(list)
    drug_icd: dict[str, set[str]] = defaultdict(set)
    if not CHUNKS_PATH.is_file():
        return texts, freq, field_icd, class_freq, class_field, drug_icd
    for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError:
            continue
        sp = _norm_path(str(c.get("source_path") or c.get("path") or ""))
        if not sp:
            continue
        for key in ("text", "excerpt", "content", "normalized"):
            val = c.get(key)
            if isinstance(val, str) and val.strip():
                texts[sp].append(val.strip())
                break
        chunk_type = str(c.get("chunk_type") or "").lower()
        section_title = str(c.get("section_title") or "")
        is_class = chunk_type in _CLASSIFICATION_CHUNK_TYPES or _section_title_is_classification(
            section_title
        )
        is_drug = chunk_type in _DRUG_NOISE_CHUNK_TYPES or _section_title_is_drug_noise(
            section_title
        )
        for code in c.get("icd10_codes") or []:
            cc = _norm_icd(str(code))
            if len(cc) < 3:
                continue
            if is_drug and not is_class:
                drug_icd[sp].add(cc)
                continue
            freq[sp][cc] += 1
            field_icd[sp].append(cc)
            if is_class:
                class_freq[sp][cc] += 1
                class_field[sp].append(cc)
    return texts, freq, field_icd, class_freq, class_field, drug_icd


def _icd_from_summaries() -> dict[str, list[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    if not SUMMARY_JSON_DIR.is_dir():
        return {}
    for path in sorted(SUMMARY_JSON_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        src = data.get("source") if isinstance(data.get("source"), dict) else {}
        lp = _norm_path(str((src or {}).get("local_path") or ""))
        if not lp:
            continue
        for cond in data.get("conditions") or []:
            if not isinstance(cond, dict):
                continue
            for raw in cond.get("icd10_codes") or []:
                c = _norm_icd(str(raw))
                if c:
                    out[lp].add(c)
    return {k: sorted(v) for k, v in out.items()}


def _icd_from_body_text(
    *,
    title: str,
    body: str,
    extract_icd10,
    lookup_disease_icd,
    prioritize_codes,
    is_symptom_code,
) -> tuple[list[str], list[str]]:
    """ICD из текста PDF: title-first, затем частые коды из тела (без шума R/Z)."""
    sources: list[str] = []
    icd_set: set[str] = set()

    for src, blob in (("title_nomenclature", title), ("body_nomenclature", body[:20_000])):
        for code in lookup_disease_icd(blob):
            c = _norm_icd(code)
            if c:
                icd_set.add(c)
                sources.append(src)
        for code in extract_icd10(blob):
            c = _norm_icd(code)
            if c:
                icd_set.add(c)
                sources.append("title_extract" if src.startswith("title") else "body_extract")

    # Частые коды из текста чанков (минимум 2 упоминания), болезни вперёд
    freq: Counter[str] = Counter()
    for part in body[:BODY_TEXT_LIMIT].split("\n"):
        for code in extract_icd10(part):
            c = _norm_icd(code)
            if c:
                freq[c] += 1
    disease_freq = [(c, n) for c, n in freq.items() if not is_symptom_code(c)]
    disease_freq.sort(key=lambda x: (-x[1], x[0]))
    for code, count in disease_freq[:15]:
        if count >= 2 or (count >= 1 and code[:3] in _norm_blob(title)):
            if code not in icd_set:
                icd_set.add(code)
                sources.append("body_freq")

    ordered = prioritize_codes(sorted(icd_set))
    return ordered, sorted(set(sources))


def _infer_audience_from_body(
    title: str,
    body: str,
    specialty_slug: str,
    card_population: str | None,
    override: str | None,
) -> tuple[str, str]:
    if override in ("adult", "pediatric", "mixed", "any"):
        return override, "override"

    from_title = _infer_audience_from_text("", title)
    if from_title:
        return from_title, "filename"

    blob = _norm_blob(f"{title} {body[:25_000]}")
    ped_hits = sum(1 for m in _BODY_PED if m in blob)
    adult_hits = sum(1 for m in _BODY_ADULT if m in blob)
    preg_hits = sum(1 for m in _BODY_PREG if m in blob)

    if ped_hits and adult_hits:
        return "mixed", "body_text"
    if ped_hits >= 2 or ("новорожд" in blob or "неонат" in blob):
        return "pediatric", "body_text"
    if adult_hits >= 2:
        return "adult", "body_text"
    if preg_hits >= 2 or (
        specialty_slug == "akusherstvo-ginekologiya"
        and "женщин" in blob
        and ped_hits == 0
    ):
        return "adult", "body_text_pregnancy"

    pop = (card_population or "").strip().lower()
    if pop == "child":
        return "pediatric", "card_population"
    if pop == "adult":
        return "adult", "card_population"
    if specialty_slug in _PED_SLUGS and "взросл" not in blob:
        return "pediatric", "specialty_slug"
    return "any", "default"


def _classify_protocol_kind(
    title: str,
    icd_all: list[str],
    specialty_slug: str,
    body: str,
) -> tuple[str, str, bool]:
    """protocol_kind, scope_label_ru, general_scope."""
    t = _norm_blob(title)
    b = _norm_blob(body[:12_000])

    if icd_all:
        return "clinical", "Клинический КП (есть коды МКБ)", False

    if any(
        x in t
        for x in (
            "поддержка сексуального",
            "поддержка сексуального и репродуктивного",
            "медицинское наблюдение и оказание медицинской помощи женщинам в акушерстве",
            "организация оказания медицинской",
            "порядок оказания медицинской",
        )
    ):
        return "general_program", "Общий / организационный КП", True

    if "реабилитац" in t:
        return "rehabilitation", "Реабилитация (широкий охват МКБ)", True

    if any(x in t for x in ("экстракорпоральн", " бесплод", "врт", "оплодотворен")):
        return "procedural", "Процедурный КП (ВРТ / бесплодие)", True

    if "искусственн" in t and "прерыван" in t:
        return "procedural", "Процедурный КП (прерывание беременности)", True

    if "алгоритм" in t and ("зно" in t or "онколог" in t or "новообразован" in t):
        return "oncology_algorithm", "Алгоритм ЗНО (мульти-МКБ)", True

    if "диспансер" in t and not icd_all:
        return "screening_dispanser", "Диспансерное наблюдение (общий)", True

    if "оказание медицинской помощи" in t and specialty_slug in (
        "akusherstvo-ginekologiya",
        "terapiya",
    ):
        return "general_care", "Общий КП оказания помощи", True

    if "неонатолог" in t or "неонат" in t:
        return "clinical", "Клинический КП (неонатология)", False

    return "clinical_pending", "Клинический КП (МКБ не извлечён из текста)", True


def build_protocol_catalog(*, write: bool = True) -> list[dict[str, Any]]:
    """Собрать каталог из index.csv, cards, chunks, summaries, текста PDF."""
    import importlib.util

    ent_path = ROOT / "corpus_pipeline" / "entities_extract.py"
    spec = importlib.util.spec_from_file_location("entities_extract", ent_path)
    assert spec and spec.loader
    ent_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ent_mod)
    extract_icd10 = ent_mod.extract_icd10

    diag_path = ROOT / "clinical_knowledge" / "diagnosis_icd.py"
    dspec = importlib.util.spec_from_file_location("diagnosis_icd", diag_path)
    assert dspec and dspec.loader
    diag_mod = importlib.util.module_from_spec(dspec)
    dspec.loader.exec_module(diag_mod)
    lookup_disease_icd = diag_mod.lookup_disease_icd
    prioritize_codes = diag_mod.prioritize_codes
    is_symptom_code = diag_mod.is_symptom_code

    if not INDEX_CSV.is_file():
        raise FileNotFoundError(f"Missing {INDEX_CSV}")

    cards = _cards_by_path()
    chunk_texts, chunk_freq, chunk_field_icd, chunk_class_freq, chunk_class_field, chunk_drug_icd = (
        _chunk_data_by_path()
    )
    summary_icd = _icd_from_summaries()
    overrides = _load_overrides()

    rows: list[dict[str, Any]] = []
    with INDEX_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = _norm_path(row.get("relative_path") or "")
            if not path:
                continue
            filename = row.get("filename") or Path(path).name
            title = row.get("display_title") or re.sub(r"\.pdf$", "", filename, flags=re.I)
            specialty = row.get("category") or ""
            card = cards.get(path) or {}
            body = " ".join(chunk_texts.get(path, []))[:BODY_TEXT_LIMIT]

            sources: set[str] = set()
            icd_set: set[str] = set()
            for src, codes in (
                ("card", card.get("icd10_all") or card.get("icd10_primary") or []),
                ("chunks_field", chunk_field_icd.get(path, [])),
                ("summary", summary_icd.get(path, [])),
                ("filename", extract_icd10(f"{filename} {title}")),
                ("nomenclature", lookup_disease_icd(f"{filename} {title}")),
            ):
                for raw in codes:
                    c = _norm_icd(str(raw))
                    if c:
                        icd_set.add(c)
                        sources.add(src)

            body_icd, body_src = _icd_from_body_text(
                title=title,
                body=body,
                extract_icd10=extract_icd10,
                lookup_disease_icd=lookup_disease_icd,
                prioritize_codes=prioritize_codes,
                is_symptom_code=is_symptom_code,
            )
            class_body_icd, class_body_src = _icd_from_classification_sections(
                body,
                extract_icd10=extract_icd10,
                lookup_disease_icd=lookup_disease_icd,
                prioritize_codes=prioritize_codes,
                is_symptom_code=is_symptom_code,
            )
            for c in body_icd:
                icd_set.add(c)
            sources.update(body_src)
            for c in class_body_icd:
                icd_set.add(c)
            sources.update(class_body_src)

            class_codes_ordered = list(
                dict.fromkeys(
                    [_norm_icd(x) for x in chunk_class_field.get(path, []) if x]
                    + [c for c, _ in chunk_class_freq.get(path, Counter()).most_common(24)]
                    + class_body_icd
                )
            )
            drug_only = chunk_drug_icd.get(path, set())
            class_code_set = set(class_codes_ordered)

            icd_all_raw = _filter_valid_icd(prioritize_codes(sorted(icd_set)))
            icd_all = [
                c
                for c in icd_all_raw
                if not (
                    _is_external_cause_icd(c)
                    and c in drug_only
                    and c not in class_code_set
                )
            ]

            primary_src = (
                class_codes_ordered[:PRIMARY_ICD_LIMIT]
                or [_norm_icd(x) for x in (card.get("icd10_primary") or []) if x]
                or summary_icd.get(path, [])[:PRIMARY_ICD_LIMIT]
                or body_icd[:PRIMARY_ICD_LIMIT]
                or [c for c, _ in chunk_freq.get(path, Counter()).most_common(PRIMARY_ICD_LIMIT)]
                or icd_all[:PRIMARY_ICD_LIMIT]
            )
            icd_primary = _filter_valid_icd(
                [c for c in dict.fromkeys(primary_src) if c and not _is_external_cause_icd(c)]
            )[:PRIMARY_ICD_LIMIT]

            aud_csv = (row.get("audience") or "").strip()
            override = overrides.get(path)
            if aud_csv in ("adult", "pediatric", "mixed"):
                audience, aud_src = aud_csv, "index_csv"
            else:
                audience, aud_src = _infer_audience_from_body(
                    title,
                    body,
                    specialty,
                    str(card.get("population") or ""),
                    override,
                )

            kind, scope_label, general_scope = _classify_protocol_kind(
                title, icd_all, specialty, body
            )
            if icd_all:
                kind, scope_label, general_scope = "clinical", "Клинический КП (есть коды МКБ)", False

            if class_codes_ordered:
                sources.add("chunks_classification")
            if class_body_icd:
                sources.add("body_classification")

            confidence = "high" if icd_primary else ("medium" if icd_all else "low")
            title_norm = normalize_protocol_title(title, filename)
            icd_weights = compute_icd10_relevance_weights(
                icd_primary=icd_primary,
                icd_all=icd_all,
                title=title_norm,
                body=body,
                chunk_freq=chunk_freq.get(path),
            )
            content_tags = extract_content_tags(
                title_norm, body, specialty, kind, audience
            )
            section_tags = extract_section_tags_from_text(body)
            entry = {
                "path": path,
                "specialty_slug": specialty,
                "display_title": title,
                "display_title_normalized": title_norm,
                "audience": audience,
                "audience_source": aud_src,
                "icd10_primary": icd_primary,
                "icd10_all": icd_all,
                "icd10_weights": icd_weights,
                "icd_sources": sorted(sources),
                "icd_count": len(icd_all),
                "confidence": confidence,
                "protocol_kind": kind,
                "scope_label_ru": scope_label,
                "general_scope": general_scope,
                "content_tags": content_tags,
                "section_tags": section_tags,
            }
            rows.append(entry)

    if write:
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CATALOG_PATH.open("w", encoding="utf-8") as out:
            for entry in rows:
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return rows


@lru_cache(maxsize=1)
def load_protocol_catalog() -> dict[str, dict[str, Any]]:
    if not CATALOG_PATH.is_file():
        try:
            build_protocol_catalog(write=True)
        except Exception:
            return {}
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(CATALOG_PATH):
        sp = _norm_path(str(row.get("path") or ""))
        if sp:
            out[sp] = row
    return out


def catalog_stats() -> dict[str, Any]:
    cat = load_protocol_catalog()
    total = len(cat)
    with_icd = sum(1 for r in cat.values() if r.get("icd10_all"))
    with_aud = sum(1 for r in cat.values() if r.get("audience") not in ("", "any", None))
    by_kind: Counter[str] = Counter()
    general = 0
    for r in cat.values():
        by_kind[str(r.get("protocol_kind") or "unknown")] += 1
        if r.get("general_scope"):
            general += 1
    return {
        "total": total,
        "with_icd": with_icd,
        "without_icd": total - with_icd,
        "with_explicit_audience": with_aud,
        "audience_any": sum(1 for r in cat.values() if r.get("audience") == "any"),
        "general_scope_count": general,
        "by_protocol_kind": dict(by_kind),
    }
