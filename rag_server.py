#!/usr/bin/env python3
"""
Локальный RAG: отбор фрагментов из корпусных JSONL (corpus_chunks_parts/*.jsonl) или из chunks.json и ответ по ним.

Запуск: pip install -r requirements-rag.txt, скопировать .env.example в .env и задать ключ API.
Переменные — из .env / .env.local (python-dotenv). См. комментарии в .env.example.

Фронт (index.html) вызывает POST /api/assist; ключ к API не передаётся в браузер.
"""
from __future__ import annotations

import gc
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from env_load import load_project_env

load_project_env(ROOT)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as e:
    raise SystemExit(f"Установите: pip install -r requirements-rag.txt ({e})") from e

CHUNKS_PATH = ROOT / "chunks.json"
CORPUS_CHUNKS_PARTS_GLOB = "corpus_chunks_parts/chunks.part.*.jsonl"
PROTOCOLS_PATH = ROOT / "protocols.json"

_chunks: list[dict] = []
_chunks_by_path: dict[str, list[dict]] = {}
_protocols_by_path: dict[str, dict] = {}
_protocol_meta: dict[str, dict] = {}
_structured_by_path: dict[str, dict] = {}
_routing: dict = {}
_model = None

PROTOCOL_META_PATH = ROOT / "protocol_meta.json"
STRUCTURED_INDEX_PATH = ROOT / "structured_index.json"

ALLOWED_SPECIALTY_SLUGS = frozenset(
    [
        "akusherstvo-ginekologiya",
        "allergologiya-immunologiya",
        "anesteziologiya-reanimatologiya",
        "bolezni-sistemy-krovoobrashcheniya",
        "dermatovenerologiya",
        "endokrinologiya-narusheniya-obmena-veshchestv",
        "gastroenterologiya",
        "gematologiya",
        "infektsionnye-zabolevaniya",
        "khirurgiya",
        "nefrologiya",
        "nevrologiya-neyrokhirurgiya",
        "novoobrazovaniya",
        "oftalmologiya",
        "otorinolaringologiya",
        "palliativnaya-pomoshch",
        "psikhiatriya-narkologiya",
        "pulmonologiya-ftiziatriya",
        "revmatologiya",
        "stomatologiya",
        "transplantatsiya-organov-i-tkaney",
        "travmatologiya-ortopediya",
        "urologiya",
        "zabolevaniya-perinatalnogo-perioda",
    ]
)

# Рубрики каталога Минздрава РБ (slug → подпись для UI и /api/specialties).
SPECIALTY_LABELS_RU: dict[str, str] = {
    "akusherstvo-ginekologiya": "Акушерство и гинекология",
    "allergologiya-immunologiya": "Аллергология и иммунология",
    "anesteziologiya-reanimatologiya": "Анестезиология и реаниматология",
    "bolezni-sistemy-krovoobrashcheniya": "Болезни системы кровообращения",
    "dermatovenerologiya": "Дерматовенерология",
    "endokrinologiya-narusheniya-obmena-veshchestv": "Эндокринология и обмен веществ",
    "gastroenterologiya": "Гастроэнтерология",
    "gematologiya": "Гематология",
    "infektsionnye-zabolevaniya": "Инфекционные заболевания",
    "khirurgiya": "Хирургия",
    "nefrologiya": "Нефрология",
    "nevrologiya-neyrokhirurgiya": "Неврология и нейрохирургия",
    "novoobrazovaniya": "Новообразования",
    "oftalmologiya": "Офтальмология",
    "otorinolaringologiya": "Оториноларингология",
    "palliativnaya-pomoshch": "Паллиативная помощь",
    "psikhiatriya-narkologiya": "Психиатрия и наркология",
    "pulmonologiya-ftiziatriya": "Пульмонология и фтизиатрия",
    "revmatologiya": "Ревматология",
    "stomatologiya": "Стоматология",
    "transplantatsiya-organov-i-tkaney": "Трансплантация органов и тканей",
    "travmatologiya-ortopediya": "Травматология и ортопедия",
    "urologiya": "Урология",
    "zabolevaniya-perinatalnogo-perioda": "Перинатальный период",
}

SYSTEM_JSON = """Ты помощник врача по клиническим протоколам Минздрава Республики Беларусь.
Фрагменты PDF ниже могут быть неполными. Не выдумывай факты вне фрагментов.
Если в запросе есть блок «=== Контекст пациента ===» (возраст, пол и т.д.) и «=== Жалобы и вопрос ===», учитывай контекст при выборе детских vs взрослых протоколов и в формулировке summary.
Если возраст явно взрослый (например «49 лет», ≥18 лет) — не включай в protocols детские КП: в списке должны остаться только path из входных фрагментов; если фрагменты только детские (маловероятно), опирайся на них осторожно и не выдавай детский протокол как основной без пометки.
Клиническая калибровка (обязательно):
- Опирайся на симптомы и формулировки из «Запрос пользователя». Не приписывай пациенту симптомов, которых там нет (в частности: насморк, боль в горле, ангина, ОРВИ, если их не указали в запросе). Не переноси симптомы из фрагментов протоколов в описание жалобы, если их не было в запросе.
- ОРВИ, фарингит, тонзиллит, риносинусит и др. типичные ЛОР-причины — только если они явно следуют из запроса пользователя и/или из приведённых фрагментов; не подставляй их «по умолчанию».
- Редкие неотложные состояния (острый эпиглоттит, ретрофарингеальный абсцесс и т.п.) — только при явных красных флагах в тексте запроса (выраженная одышка, слюнотечение, невозможность глотать слюну, быстрое ухудшение) или если это прямо следует из фрагментов. Если пользователь указал нормальное дыхание без одышки — не ставь эпиглоттит первым в дифференциальный ряд и не формулируй ответ так, будто он наиболее вероятен.
- Не противоречь явным фактам из запроса (например «дыхание нормальное»).
Верни ОДИН JSON-объект (без markdown, без текста до/после).
Схема полей:
{
  "summary": "…",
  "protocols": [{"path":"…","title":"…","match_reason":"…","confidence":"низкая|средняя|высокая","confidence_score":0.0}],
  "differential": ["…","…"],
  "questions_for_patient": [] или ["…","…"],
  "disclaimer": "Информация из протоколов; не замена очной консультации."
}
ЖЁСТКИЕ ЛИМИТЫ (иначе ответ обрежется посередине):
- summary: РОВНО 2 предложения на русском, каждое заканчивается точкой. Вместе НЕ ДЛИННЕЕ 280 символов (с пробелами). Без тире в конце; последний символ — точка.
- match_reason: не длиннее 70 символов, одно короткое предложение или фраза, законченная по смыслу.
- differential: ровно 2 строки, каждая 3–8 слов.
- questions_for_patient: если хотя бы у одного протокола confidence_score равен 1.0 (полное соответствие запросу) — пустой массив []. Иначе ровно 2 коротких вопроса.
- protocols: все уникальные path из входных фрагментов; confidence_score 0.0–1.0.
Если не хватает места — сожми формулировки, но НЕ обрывай слова и НЕ оставляй незаконченное предложение в summary."""

SYSTEM_JSON_RETRY = """Повтори задачу: нужен ОДИН компактный JSON (без markdown).
Не добавляй симптомы носа/горла/ОРВИ, если их не было в запросе пользователя. Эпиглоттит и др. редкие неотложные — только при красных флагах или прямо в фрагментах; при нормальном дыхании не веди с эпиглоттита.
Предыдущая попытка оборвалась по длине. Сделай ещё короче:
- summary: РОВНО 2 коротких предложения, ВМЕСТЕ максимум 220 символов, последний символ — точка.
- match_reason: до 55 символов на протокол.
- differential: 2 коротких пункта; questions_for_patient: [] если есть протокол с confidence_score 1.0, иначе 2 коротких вопроса.
Сохрани все path из фрагментов. Не обрывай слова."""

SYSTEM_EXTRACT = """Ты помощник врача. По фрагментам клинического протокола Минздрава Республики Беларусь извлеки факты, относящиеся к запросу пользователя.
Верни ОДИН JSON-объект (без markdown, без текста до/после).
Схема:
{
  "diagnosis": "диагнозы, состояния, показания протокола по тексту (1–5 предложений)",
  "treatment_methods": ["метод или этап лечения — по тексту протокола"],
  "medications": ["группы препаратов или МНН, если названы во входном тексте — без выдуманных доз"],
  "note": "кратко: чего нет в фрагментах или что требует очной консультации"
}
Не придумывай препараты, дозы и процедуры, которых нет во входном тексте."""

SYSTEM_EXTRACT_FULL = """Ты помощник врача. По ПОЛНОМУ тексту фрагментов клинического протокола Минздрава Республики Беларусь извлеки структурированные сведения, релевантные запросу пользователя.
Запрос считается полностью покрытым протоколом — дай развёрнутый, практичный разбор строго по тексту протокола.
Верни ОДИН JSON-объект (без markdown, без текста до/после).
Схема:
{
  "diagnosis": "диагнозы, состояния, показания (2–8 предложений)",
  "treatment_methods": ["этапы и методы лечения — по тексту протокола"],
  "medications": ["группы препаратов, МНН, режимы — только если есть во входном тексте"],
  "recommendations": ["рекомендации и алгоритм действий для врача/пациента — по тексту"],
  "monitoring_followup": "наблюдение, контроль, когда обращаться — если есть в тексте, иначе кратко что не указано",
  "contraindications": "противопоказания и ограничения — если названы во фрагментах, иначе пустая строка",
  "note": "чего нет в фрагментах; необходимость очной консультации"
}
Не придумывай дозировки, препараты и процедуры, которых нет во входном тексте."""

SYSTEM_CLASSIFY = """По краткому медицинскому запросу пациента выбери до трёх рубрик клинических протоколов (slug), которым соответствует ситуация.
Верни ОДИН JSON: {"categories": ["slug1"], "note": "одно короткое предложение"}
slug ТОЛЬКО из этого списка (копируй точно):
""" + ", ".join(sorted(ALLOWED_SPECIALTY_SLUGS)) + """
Если нельзя уверенно сопоставить — верни "categories": []."""


def _jsonl_chunk_files() -> list[Path]:
    """Порядок: один файл из RAG_CHUNKS_JSONL, либо glob из RAG_CHUNKS_JSONL_GLOB, либо части corpus_chunks_parts."""
    one = (os.environ.get("RAG_CHUNKS_JSONL") or "").strip()
    if one:
        p = Path(one)
        if not p.is_file():
            raise SystemExit(f"RAG_CHUNKS_JSONL: файл не найден: {p}")
        return [p.resolve()]
    gl = (os.environ.get("RAG_CHUNKS_JSONL_GLOB") or "").strip()
    if gl:
        paths = sorted(ROOT.glob(gl))
        if not paths:
            raise SystemExit(f"RAG_CHUNKS_JSONL_GLOB: нет файлов: {ROOT / gl}")
        return paths
    return sorted(ROOT.glob(CORPUS_CHUNKS_PARTS_GLOB))


def _memory_saver_enabled() -> bool:
    """По умолчанию — полный lex (embedding_ready_text при отличии от text).

    На слабом инстансе (например 512Mi) задайте RAG_MEMORY_SAVER=1 — без дубля
    embedding_ready_text, чтобы избежать OOM при старте.
    """
    v = (os.environ.get("RAG_MEMORY_SAVER") or "").strip().lower()
    return v in ("1", "true", "yes")


def _load_chunks_from_jsonl(part_paths: list[Path]) -> list[dict]:
    """Корпусный pipeline: строки JSONL → формат retrieve() / gather_protocol_text.

    Без промежуточного списка «всех сырых строк» — сразу группировка по path и только
    нужные поля (экономия RAM). lex_text хранится только если отличается от text.
    """
    memory_saver = _memory_saver_enabled()
    lex_cap = int(os.environ.get("RAG_LEXICAL_MAX_CHARS", "0") or "0")
    by_path: dict[str, list[dict]] = {}
    for pp in part_paths:
        with pp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                p = (row.get("source_path") or "").strip()
                if not p:
                    continue
                text = (row.get("text") or "").strip()
                slim: dict = {
                    "page_from": int(row.get("page_from") or 0),
                    "page_to": int(row.get("page_to") or 0),
                    "chunk_id": row.get("chunk_id"),
                    "text": text,
                    "chunk_type": (row.get("chunk_type") or "body").strip() or "body",
                }
                if not memory_saver:
                    ert = (row.get("embedding_ready_text") or "").strip()
                    if ert and ert != text:
                        if lex_cap > 0 and len(ert) > lex_cap:
                            ert = ert[:lex_cap]
                        slim["lex_text"] = ert
                elif lex_cap > 0 and len(text) > lex_cap:
                    slim["lex_text"] = text[:lex_cap]
                by_path.setdefault(p, []).append(slim)
    out: list[dict] = []
    for p in sorted(by_path.keys()):
        rows = sorted(
            by_path[p],
            key=lambda r: (
                r["page_from"],
                r["page_to"],
                str(r.get("chunk_id") or ""),
            ),
        )
        for i, row in enumerate(rows):
            text = (row.get("text") or "").strip()
            rec: dict = {
                "path": p,
                "text": text,
                "title": "",
                "category": "",
                "kind": row.get("chunk_type") or "body",
                "chunk_index": i,
                "chunk_id": row.get("chunk_id"),
            }
            if "lex_text" in row:
                rec["lex_text"] = row["lex_text"]
            out.append(rec)
    return out


def _use_jsonl_chunks() -> bool:
    """По умолчанию — JSONL-чанки (корпус), если явно не задан RAG_CHUNKS_SOURCE=json."""
    src = (os.environ.get("RAG_CHUNKS_SOURCE") or "").strip().lower()
    if src in ("json", "legacy", "chunks.json"):
        return False
    if src in ("jsonl", "corpus", "parts", "1", "true", "yes"):
        return True
    # авто: есть части corpus → jsonl; иначе chunks.json
    return bool(_jsonl_chunk_files())


def _enrich_chunks_from_index() -> None:
    """Заголовок и рубрика из protocols.json / protocol_meta для routing и retrieve."""
    for ch in _chunks:
        p = ch.get("path") or ""
        if not p:
            continue
        pr = _protocols_by_path.get(p) or {}
        pm = _protocol_meta.get(p) or {}
        if not (ch.get("title") or "").strip():
            ch["title"] = (pr.get("title") or pm.get("title") or "").strip() or Path(
                p
            ).stem
        if not (ch.get("category") or "").strip():
            ch["category"] = (pr.get("category") or pm.get("category") or "").strip()


def load_data() -> None:
    global _chunks, _chunks_by_path, _protocols_by_path, _protocol_meta, _structured_by_path, _routing
    _protocols_by_path = {}
    if PROTOCOLS_PATH.is_file():
        for row in json.loads(PROTOCOLS_PATH.read_text(encoding="utf-8")):
            _protocols_by_path[row["path"]] = row
    if PROTOCOL_META_PATH.is_file():
        _protocol_meta = json.loads(PROTOCOL_META_PATH.read_text(encoding="utf-8"))
    else:
        _protocol_meta = {}
    if STRUCTURED_INDEX_PATH.is_file():
        _structured_by_path = {
            row["path"]: row
            for row in json.loads(STRUCTURED_INDEX_PATH.read_text(encoding="utf-8"))
            if row.get("path")
        }
    else:
        _structured_by_path = {}
    rp = ROOT / "symptom_routing.json"
    if rp.is_file():
        _routing = json.loads(rp.read_text(encoding="utf-8"))
    else:
        _routing = {}

    if _use_jsonl_chunks():
        parts = _jsonl_chunk_files()
        if not parts:
            raise SystemExit(
                f"Нет JSONL-чанков ({CORPUS_CHUNKS_PARTS_GLOB} или RAG_CHUNKS_JSONL). "
                "Соберите корпус или задайте RAG_CHUNKS_SOURCE=json и наличие chunks.json"
            )
        _chunks = _load_chunks_from_jsonl(parts)
    else:
        if not CHUNKS_PATH.is_file():
            raise SystemExit(
                f"Нет {CHUNKS_PATH}. Запустите: python3 build_chunks.py "
                "или положите corpus_chunks_parts/*.jsonl и уберите RAG_CHUNKS_SOURCE=json"
            )
        _chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    _enrich_chunks_from_index()
    _chunks_by_path = {}
    for ch in _chunks:
        p = ch.get("path") or ""
        if not p:
            continue
        _chunks_by_path.setdefault(p, []).append(ch)
    for plist in _chunks_by_path.values():
        plist.sort(key=lambda x: int(x.get("chunk_index", 0)))
    gc.collect()


def tokenize_ru(s: str) -> list[str]:
    s = s.lower().replace("ё", "е")
    return [t for t in re.findall(r"[а-яa-z]{2,}", s) if len(t) >= 2]


def _norm_query(s: str) -> str:
    return (s or "").lower().replace("ё", "е")


def infer_audience_from_query(q: str, routing: dict) -> str | None:
    """'adult' | 'child' | None — по словам и числам (49 лет, ребёнок …)."""
    nq = _norm_query(q)
    aud = routing.get("audience") or {}
    child_m = aud.get("child_markers") or []
    adult_m = aud.get("adult_markers") or []
    has_ch = any(c in nq for c in child_m)
    has_ad = any(a in nq for a in adult_m)
    if has_ad and not has_ch:
        return "adult"
    if has_ch and not has_ad:
        return "child"

    def age_bucket(age: int) -> str | None:
        if age >= 18:
            return "adult"
        if 0 < age < 18:
            return "child"
        return None

    for m in re.finditer(r"(\d{1,3})\s*лет", nq):
        b = age_bucket(int(m.group(1)))
        if b:
            return b
    for m in re.finditer(r"(\d{1,3})\s*года?\b", nq):
        b = age_bucket(int(m.group(1)))
        if b:
            return b
    for m in re.finditer(r"возраст\s*[:\s]*(\d{1,3})\b", nq):
        b = age_bucket(int(m.group(1)))
        if b:
            return b
    for m in re.finditer(r"пациент(?:у|а)?\s+(\d{1,3})\s*лет", nq):
        b = age_bucket(int(m.group(1)))
        if b:
            return b
    return None


def doc_audience_hint(path: str, title: str, routing: dict) -> str | None:
    """pediatric | adult | mixed | None — по названию файла/заголовка."""
    s = f"{path} {title}".lower()
    ped = routing.get("pediatric_title_markers") or []
    adult_t = routing.get("adult_title_markers") or []
    has_p = any(p in s for p in ped)
    has_a = any(a in s for a in adult_t)
    if has_p and has_a:
        return "mixed"
    if has_p:
        return "pediatric"
    if has_a:
        return "adult"
    return None


def filter_retrieval_by_audience(
    rows: list[dict], rq: str, routing: dict
) -> tuple[list[dict], str | None, bool]:
    """Отбрасывает чанки с явно несовпадающей аудиторией (дет/взросл)."""
    aud = infer_audience_from_query(rq, routing)
    if aud is None or not rows:
        return rows, aud, False

    strict = os.environ.get("RAG_AUDIENCE_FILTER", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not strict:
        return rows, aud, False

    out: list[dict] = []
    for r in rows:
        hint = doc_audience_hint(
            r.get("path") or "",
            r.get("title") or "",
            routing,
        )
        if hint is None or hint == "mixed":
            out.append(r)
            continue
        if aud == "adult" and hint == "pediatric":
            continue
        if aud == "child" and hint == "adult":
            continue
        out.append(r)

    if not out:
        return rows, aud, True
    return out, aud, False


def routing_multiplier(raw_query: str, ch: dict, routing: dict | None) -> float:
    """Усиление/ослабление релевантности по symptom_routing.json (рубрики, аудитория, path)."""
    if not routing:
        return 1.0
    q = _norm_query(raw_query)
    cat = (ch.get("category") or "").strip()
    title_low = ((ch.get("title") or "") + " " + (ch.get("path") or "")).lower()

    m = 1.0

    for br in routing.get("boost_rules", []):
        kws = br.get("match") or []
        if not any(k in q for k in kws):
            continue
        cats = br.get("categories") or []
        if cat and cat in cats:
            m *= float(br.get("factor", 1.0))

    for pr in routing.get("penalty_rules", []):
        when = pr.get("when") or []
        if not any(w in q for w in when):
            continue
        unless = pr.get("unless") or []
        if unless and any(u in q for u in unless):
            continue
        if cat in (pr.get("categories") or []):
            m *= float(pr.get("factor", 1.0))

    aud = routing.get("audience") or {}
    child_m = aud.get("child_markers") or []
    adult_m = aud.get("adult_markers") or []
    ped_title = routing.get("pediatric_title_markers") or []
    adult_title = routing.get("adult_title_markers") or []

    infer = infer_audience_from_query(raw_query, routing)
    if infer == "adult":
        has_child = False
        has_adult = True
    elif infer == "child":
        has_child = True
        has_adult = False
    else:
        has_child = any(c in q for c in child_m)
        has_adult = any(a in q for a in adult_m)
    if has_adult and not has_child:
        if any(p in title_low for p in ped_title):
            m *= float(aud.get("penalty_adult_query_pediatric_doc", 0.35))
    if has_child and not has_adult:
        if any(a in title_low for a in adult_title):
            m *= float(aud.get("penalty_child_query_adult_doc", 0.4))

    for pp in routing.get("path_penalties", []):
        when_q = pp.get("when_query") or []
        if not when_q or not any(w in q for w in when_q):
            continue
        unless = pp.get("unless_query") or []
        if unless and any(u in q for u in unless):
            continue
        pats = pp.get("path_contains") or []
        if any(p.lower() in title_low for p in pats):
            m *= float(pp.get("factor", 0.5))

    for pb in routing.get("path_boosts", []):
        needed = pb.get("when_query") or []
        min_hits = int(pb.get("when_min_hits", 2))
        hits = sum(1 for w in needed if w in q)
        if hits < min_hits:
            continue
        pats = pb.get("path_contains") or []
        if any(p.lower() in title_low for p in pats):
            m *= float(pb.get("factor", 1.5))

    return max(m, 1e-9)


def clinical_query_for_rag(full_query: str) -> str:
    """Текст для лексического RAG: блок «Жалобы и вопрос» без контекста и без ответов на уточняющие вопросы."""
    sep = "=== Жалобы и вопрос ==="
    if sep in full_query:
        part = full_query.split(sep, 1)[1].strip()
    else:
        part = full_query.strip()
    # Блок ответов содержит слова вопросов (напр. «кровотечение») — подстрока «кров» ложно тянет гематологию и размывает отбор.
    mark = "— Ответы на уточняющие вопросы:"
    if mark in part:
        part = part.split(mark, 1)[0].strip()
    return part if part else full_query.strip()


def gather_protocol_text(path: str, max_chars: int) -> str:
    """Склеивает чанки одного PDF по порядку (до max_chars символов)."""
    parts = _chunks_by_path.get(path) or []
    out: list[str] = []
    n = 0
    for ch in parts:
        t = (ch.get("text") or "").strip()
        if not t:
            continue
        if n + len(t) > max_chars:
            rest = max_chars - n
            if rest > 80:
                out.append(t[:rest])
            break
        out.append(t)
        n += len(t)
    return "\n\n".join(out)


def confidence_display_full(score: object) -> bool:
    """Совпадает с отображением 100% в интерфейсе (округление как в index.html)."""
    try:
        x = float(score)
    except (TypeError, ValueError):
        return False
    x = max(0.0, min(1.0, x))
    return round(100 * x) >= 100


def infer_specialties_gemini(q: str, model) -> list[str]:
    """Опционально: первый короткий вызов LLM — к каким рубрикам относится запрос."""
    if os.environ.get("GEMINI_SPECIALTY_CLASSIFY", "0").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return []
    prompt = SYSTEM_CLASSIFY + "\n\nЗапрос пользователя:\n" + (q or "")[:6000]
    try:
        resp = generate_gemini(model, prompt)
        txt = _extract_gemini_text(resp)
        parsed = _try_parse_json(txt)
    except HTTPException:
        return []
    except Exception:
        return []
    if not parsed or not isinstance(parsed, dict):
        return []
    cats = parsed.get("categories") or []
    out = [c for c in cats if isinstance(c, str) and c in ALLOWED_SPECIALTY_SLUGS]
    return out[:3]


def extract_clinical_detail(
    path: str,
    query: str,
    title_hint: str,
    model,
    *,
    detailed: bool = False,
) -> dict | None:
    """Второй вызов LLM: факты по протоколу; при detailed — расширенная схема и больший объём текста."""
    if detailed:
        max_body = int(os.environ.get("RAG_EXTRACT_FULL_MATCH_MAX_CHARS", "32000"))
        idx_lim = 16000
        system = SYSTEM_EXTRACT_FULL
    else:
        max_body = int(os.environ.get("RAG_EXTRACT_MAX_CHARS", "16000"))
        idx_lim = 8000
        system = SYSTEM_EXTRACT
    body = gather_protocol_text(path, max_body)
    struct = _structured_by_path.get(path) or {}
    extra = ""
    if struct.get("diagnosis"):
        extra += "\n\n[Выдержка индекса: диагностика]\n" + str(struct["diagnosis"])[:idx_lim]
    if struct.get("treatment"):
        extra += "\n\n[Выдержка индекса: лечение]\n" + str(struct["treatment"])[:idx_lim]
    if len(body.strip()) < 120 and not extra.strip():
        return None
    meta = _protocol_meta.get(path) or {}
    spec = meta.get("specialty_ru") or ""
    title_line = title_hint or meta.get("title") or path
    prompt = (
        system
        + "\n\n---\n\n"
        + f"Запрос пользователя:\n{query}\n\n"
        + f"Специальность (рубрика каталога): {spec}\n"
        + f"Название протокола: {title_line}\n\n"
        + "Текст протокола (фрагменты PDF):\n"
        + body
        + extra
    )
    plim = int(os.environ.get("GEMINI_PROMPT_MAX_CHARS", "28000"))
    if len(prompt) > plim:
        prompt = prompt[: plim - 80] + "\n…[обрезано]"
    try:
        resp = generate_gemini(model, prompt)
        txt = _extract_gemini_text(resp)
        parsed = _try_parse_json(txt)
    except HTTPException as e:
        return {"error": str(e.detail), "path": path, "title": title_line}
    except Exception as e:
        return {"error": str(e)[:400], "path": path, "title": title_line}
    if not parsed or not isinstance(parsed, dict):
        return None
    ext: dict = {
        "diagnosis": parsed.get("diagnosis") or "",
        "treatment_methods": parsed.get("treatment_methods") or [],
        "medications": parsed.get("medications") or [],
        "note": parsed.get("note") or "",
    }
    if detailed:
        ext["recommendations"] = parsed.get("recommendations") or []
        ext["monitoring_followup"] = parsed.get("monitoring_followup") or ""
        ext["contraindications"] = parsed.get("contraindications") or ""
        ext["detailed"] = True
    return {
        "path": path,
        "title": title_line,
        "specialty_ru": spec or None,
        "category": meta.get("category"),
        "extraction": ext,
    }


def retrieve(
    query: str,
    max_chunks: int | None = None,
    max_per_path: int = 2,
    routing_query: str | None = None,
    category_boost: list[str] | None = None,
    user_category_slugs: list[str] | None = None,
) -> list[dict]:
    """Лексический отбор + множители из symptom_routing.json (если RAG_ROUTING=1).

    query — короткий текст для подсчёта совпадений с чанками (обычно только жалобы).
    routing_query — полный запрос для правил возраста/рубрик; если None, берётся query.
    category_boost — slug рубрик из опционального LLM-классификатора запроса.
    user_category_slugs — рубрики, выбранные пользователем в форме: усиление совпадений и штраф нерелевантных чанков.
    """
    if max_chunks is None:
        max_chunks = int(os.environ.get("RAG_MAX_CHUNKS", "6"))
    use_routing = os.environ.get("RAG_ROUTING", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    boost_set = frozenset(category_boost or [])
    boost_factor = float(os.environ.get("RAG_CATEGORY_BOOST_FACTOR", "1.45"))
    user_slugs = frozenset(
        s for s in (user_category_slugs or []) if s in ALLOWED_SPECIALTY_SLUGS
    )
    user_boost = float(os.environ.get("RAG_USER_CATEGORY_BOOST", "2.05"))
    user_penalty = float(os.environ.get("RAG_USER_CATEGORY_PENALTY", "0.32"))
    user_uncertain = float(os.environ.get("RAG_USER_CATEGORY_UNCERTAIN", "0.78"))
    rq = routing_query if routing_query is not None else query
    qtok = set(tokenize_ru(query))
    if not qtok:
        return []
    scored: list[tuple[float, float, float, dict]] = []
    for ch in _chunks:
        lex_src = (ch.get("lex_text") or ch.get("text") or "") + " " + (
            ch.get("title") or ""
        )
        low = lex_src.lower()
        lex = 0.0
        for t in qtok:
            if t in low:
                lex += 1.0 + min(len(t), 10) * 0.02
        if lex <= 0:
            continue
        mult = (
            routing_multiplier(rq, ch, _routing)
            if use_routing
            else 1.0
        )
        final = lex * mult
        cat = (ch.get("category") or "").strip()
        if boost_set and cat in boost_set:
            final *= boost_factor
        if user_slugs:
            if cat and cat in user_slugs:
                final *= user_boost
            elif cat and cat not in user_slugs:
                final *= user_penalty
            else:
                final *= user_uncertain
        scored.append((final, lex, mult, ch))
    scored.sort(key=lambda x: -x[0])

    per_path: dict[str, int] = {}
    out: list[dict] = []
    for final, lex, mult, ch in scored:
        p = ch.get("path") or ""
        if per_path.get(p, 0) >= max_per_path:
            continue
        per_path[p] = per_path.get(p, 0) + 1
        out.append(
            {
                "path": p,
                "title": ch.get("title") or "",
                "kind": ch.get("kind") or "general",
                "score": round(final, 3),
                "lexical_score": round(lex, 3),
                "routing_multiplier": round(mult, 4),
                "excerpt": (ch.get("text") or "")[
                    : int(os.environ.get("RAG_EXCERPT_CHARS", "700"))
                ],
            }
        )
        if len(out) >= max_chunks:
            break
    return out


# Большой промпт и вызов модели могут занимать 2–3+ мин; клиент в index.html ждёт дольше сервера
GEMINI_CALL_TIMEOUT = float(os.environ.get("GEMINI_CALL_TIMEOUT", "180"))


def get_gemini():
    global _model
    if _model is not None:
        return _model
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=503,
            detail="Задайте переменную окружения GOOGLE_API_KEY",
        )
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail="Установите: pip install google-generativeai",
        ) from e
    genai.configure(api_key=key)
    name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    _model = genai.GenerativeModel(name)
    return _model


def _extract_gemini_text(resp) -> str:
    """Безопасно: при блокировке/пустом ответе свойство .text бросает ValueError."""
    try:
        t = resp.text
        if t:
            return str(t).strip()
    except (ValueError, AttributeError, TypeError):
        pass
    parts: list[str] = []
    cands = getattr(resp, "candidates", None) or []
    for cand in cands:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                parts.append(part.text)
    return "".join(parts).strip()


def _gemini_finish_reason(resp) -> str | None:
    cands = getattr(resp, "candidates", None) or []
    if not cands:
        return None
    fr = getattr(cands[0], "finish_reason", None)
    if fr is None:
        return None
    return str(fr)


def _generate_blocking(model, full_prompt: str):
    import google.generativeai as genai

    max_out = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "16384"))
    use_json = os.environ.get("GEMINI_JSON_MODE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    cfg_kw: dict = {
        "temperature": 0.25,
        "max_output_tokens": max_out,
    }
    if use_json:
        # Снижает обрывы посреди JSON и обрывы «лишнего» текста до/после объекта
        cfg_kw["response_mime_type"] = "application/json"
    return model.generate_content(
        full_prompt,
        generation_config=genai.GenerationConfig(**cfg_kw),
    )


def generate_gemini(model, full_prompt: str):
    """Один поток + таймаут — иначе вызов к API может «висеть» без ответа."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_generate_blocking, model, full_prompt)
        try:
            return fut.result(timeout=GEMINI_CALL_TIMEOUT)
        except FuturesTimeout as e:
            raise HTTPException(
                status_code=504,
                detail=f"Таймаут вызова модели ({int(GEMINI_CALL_TIMEOUT)} с). Проверьте сеть или GEMINI_MODEL.",
            ) from e


def _try_parse_json(t: str) -> dict | None:
    if not t:
        return None
    s = t.strip()
    if "```" in s:
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.M)
        s = re.sub(r"\s*```\s*$", "", s, flags=re.M)
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


def _finish_hits_max(resp) -> bool:
    fr = (_gemini_finish_reason(resp) or "").upper()
    return "MAX" in fr or "LENGTH" in fr


load_data()
app = FastAPI(title="Protocol RAG", version="1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AssistIn(BaseModel):
    query: str = Field(..., min_length=2, max_length=12000)
    category_slugs: list[str] = Field(
        default_factory=list,
        description="Рубрики Минздрава (slug), выбранные пользователем — усиливают отбор",
    )


@app.get("/health")
def health() -> dict:
    has_key = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    return {
        "ok": True,
        "chunks": len(_chunks),
        "protocols": len(_protocols_by_path),
        "protocol_meta": len(_protocol_meta),
        "structured_index": len(_structured_by_path),
        "gemini_configured": has_key,
        "specialties_count": len(SPECIALTY_LABELS_RU),
        "memory_saver": _memory_saver_enabled(),
    }


@app.get("/api/specialties")
def api_specialties() -> dict:
    """Рубрики каталога клинических протоколов (slug + подпись для формы)."""
    return {
        "specialties": [
            {"slug": s, "label": SPECIALTY_LABELS_RU.get(s, s)}
            for s in sorted(SPECIALTY_LABELS_RU.keys())
        ]
    }


try:
    from gemini_verify import verify_gemini_key as _verify_gemini_key
except ImportError:
    _verify_gemini_key = None


@app.get("/api/verify-key")
def verify_key() -> dict:
    """Один тестовый запрос к модели — проверка ключа из .env."""
    if _verify_gemini_key is None:
        raise HTTPException(
            status_code=501,
            detail="Модуль gemini_verify не найден",
        )
    ok, msg = _verify_gemini_key()
    if not ok:
        raise HTTPException(status_code=502, detail=msg)
    return {
        "ok": True,
        "reply_preview": msg,
        "model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    }


@app.post("/api/assist")
def api_assist(body: AssistIn) -> dict:
    q = body.query.strip()
    q_rag = clinical_query_for_rag(q)
    if not q_rag:
        raise HTTPException(status_code=400, detail="Пустой текст жалобы — заполните блок «Жалобы и вопрос»")
    model = get_gemini()
    query_specialties = infer_specialties_gemini(q, model)
    user_slugs = [
        s
        for s in (body.category_slugs or [])
        if isinstance(s, str) and s in ALLOWED_SPECIALTY_SLUGS
    ]
    boost_merged = list(dict.fromkeys((query_specialties or []) + user_slugs))
    retrieved = retrieve(
        q_rag,
        routing_query=q,
        category_boost=boost_merged or None,
        user_category_slugs=user_slugs or None,
    )
    if not retrieved:
        raise HTTPException(status_code=400, detail="Пустой отбор — уточните запрос")

    retrieved, audience_inferred, audience_fallback = filter_retrieval_by_audience(
        retrieved, q, _routing
    )

    lines = []
    meta_specs: list[str] = []
    for i, r in enumerate(retrieved, 1):
        cat = ""
        p = r["path"]
        if p in _protocols_by_path:
            cat = _protocols_by_path[p].get("category") or ""
        pm = _protocol_meta.get(p)
        if pm and pm.get("specialty_ru"):
            meta_specs.append(pm["specialty_ru"])
        lines.append(
            f"[{i}] path={p}\n"
            f"рубрика={cat}\n"
            f"тип_фрагмента={r['kind']}\n"
            f"текст:\n{r['excerpt']}\n"
        )
    context = "\n---\n".join(lines)

    hint_block = ""
    if meta_specs:
        hint_block = (
            "Справочно рубрики отобранных фрагментов: "
            + ", ".join(sorted(set(meta_specs)))
            + "\n\n"
        )
    user_block = hint_block + f"Запрос пользователя:\n{q}\n\nФрагменты протоколов:\n{context}"
    full_prompt = SYSTEM_JSON + "\n\n---\n\n" + user_block
    prompt_limit = int(os.environ.get("GEMINI_PROMPT_MAX_CHARS", "28000"))
    if len(full_prompt) > prompt_limit:
        full_prompt = full_prompt[: prompt_limit - 80] + "\n…[обрезано для лимита контекста]"
    retry_used = False

    def _one_call(prompt: str) -> tuple[object, str, dict | None]:
        try:
            r = generate_gemini(model, prompt)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Модель: {e!s}") from e

        pf = getattr(r, "prompt_feedback", None)
        if pf is not None and getattr(pf, "block_reason", None):
            raise HTTPException(
                status_code=502,
                detail=f"Запрос отклонён моделью: {pf.block_reason}",
            )

        txt = _extract_gemini_text(r)
        if not txt:
            raise HTTPException(
                status_code=502,
                detail="Пустой ответ модели (блокировка контента или сбой). Попробуйте другую формулировку.",
            )
        return r, txt, _try_parse_json(txt)

    try:
        resp, text, parsed = _one_call(full_prompt)
    except HTTPException:
        raise

    finish = _gemini_finish_reason(resp)
    do_retry = os.environ.get("GEMINI_ASSIST_RETRY", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if do_retry and (parsed is None or _finish_hits_max(resp)):
        retry_prompt = SYSTEM_JSON_RETRY + "\n\n---\n\n" + user_block
        if len(retry_prompt) > prompt_limit:
            retry_prompt = retry_prompt[: prompt_limit - 80] + "\n…[обрезано]"
        try:
            resp2, text2, parsed2 = _one_call(retry_prompt)
        except HTTPException:
            pass
        else:
            retry_used = True
            resp, text, parsed = resp2, text2, parsed2
            finish = _gemini_finish_reason(resp)

    clinical_detail = None
    if parsed and os.environ.get("GEMINI_EXTRACT_FULL_MATCH", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        for pr in parsed.get("protocols") or []:
            if not confidence_display_full(pr.get("confidence_score")):
                continue
            pth = pr.get("path") or ""
            if not pth or pth not in _chunks_by_path:
                continue
            clinical_detail = extract_clinical_detail(
                pth,
                q,
                str(pr.get("title") or ""),
                model,
                detailed=True,
            )
            break

    return {
        "query": q,
        "retrieval": retrieved,
        "audience_inferred": audience_inferred,
        "retrieval_audience_fallback": audience_fallback,
        "query_specialties": query_specialties,
        "user_category_slugs": user_slugs,
        "llm_text": text,
        "llm_json": parsed,
        "gemini_finish_reason": finish,
        "gemini_retry_used": retry_used,
        "clinical_detail": clinical_detail,
    }


# Статика (index.html, protocols.json, PDF) — регистрировать после API-маршрутов.
# Иначе GET / даёт 404 «Not Found» на Render при открытии корня в браузере.
if (ROOT / "index.html").is_file():
    app.mount(
        "/",
        StaticFiles(directory=str(ROOT), html=True),
        name="site",
    )
else:

    @app.get("/")
    def root_placeholder() -> dict:
        return {
            "ok": True,
            "service": "Protocol RAG",
            "health": "/health",
            "assist": "POST /api/assist",
            "hint": "В репозитории нет index.html рядом с rag_server.py",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8787)
