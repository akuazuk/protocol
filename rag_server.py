#!/usr/bin/env python3
"""
Локальный RAG: отбор фрагментов из корпусных JSONL (corpus_chunks_parts/*.jsonl) или из chunks.json и ответ по ним.

Запуск: pip install -r requirements-rag.txt, скопировать .env.example в .env и задать ключ API.
Переменные - из .env / .env.local (python-dotenv). См. комментарии в .env.example.

Фронт (index.html) вызывает POST /api/assist; ключ к API не передаётся в браузер.
"""
from __future__ import annotations

import asyncio
import copy
import gc
import hashlib
import io
import json
import logging
import math
import os
import re
import threading
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
import csv
import zipfile
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from html.parser import HTMLParser
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent

# google.generativeai deprecated (→ google.genai); подавляем шум в логах до миграции SDK.
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*[Gg]enerativeai.*")


def _legacy_genai_module():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai

    return genai


def _legacy_genai_types():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from google.generativeai.types import HarmBlockThreshold, HarmCategory

    return HarmBlockThreshold, HarmCategory
from env_load import load_project_env

from icd_mkb import (
    ICD10_CODE_RE,
    analyze_query_for_icd,
    clinical_hints_confident,
    count_icd_code_mentions,
    describe_code,
    extract_icd_codes_diagnosis_focused,
    extract_icd_codes_raw,
    filter_icd_pool_for_complaint,
    icd_tokens_for_lex,
    merge_clinical_icd_hints,
    normalize_text_for_icd_scan,
    normalize_icd_code,
    ru_lexicon_scored_entries,
    strip_funnel_context_lines,
    text_mentions_icd_code,
)

from retrieval_bm25 import build_bm25_index

load_project_env(ROOT)

from typing import Annotated, Any, Iterable

try:
    from fastapi import FastAPI, HTTPException, File, Form, Query, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as e:
    raise SystemExit(f"Установите: pip install -r requirements-rag.txt ({e})") from e

CHUNKS_PATH = ROOT / "chunks.json"
CORPUS_CHUNKS_PARTS_GLOB = "corpus_chunks_parts/chunks.part.*.jsonl"
PROTOCOLS_PATH = ROOT / "protocols.json"


def _chunks_data_root() -> Path:
    """Каталог с JSONL-чанками: по умолчанию рядом с rag_server.py; на Render - смонтированный диск.

    Задаётся RAG_CHUNKS_DIR (например /var/data при Persistent Disk). Глобы RAG_CHUNKS_JSONL_GLOB
    и corpus_chunks_parts/*.jsonl считаются относительно этого каталога.
    """
    raw = (os.environ.get("RAG_CHUNKS_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return ROOT

_chunks: list[dict] = []
_chunks_by_path: dict[str, list[dict]] = {}
_protocols_by_path: dict[str, dict] = {}
_protocol_meta: dict[str, dict] = {}
_structured_by_path: dict[str, dict] = {}
_routing: dict = {}
_model = None
# Метаданные embed-rerank - per-thread, чтобы параллельные запросы не перетирали значения друг друга.
_retrieval_meta_tls = threading.local()
_bm25_index = None
_lex_inverted_index: dict[str, frozenset[int]] | None = None
_lex_index_lock = threading.Lock()
_chunk_global_indices_by_path: dict[str, list[int]] = {}
_lazy_chunk_store = None
_path_manifest = None
_path_lex_index = None


def _set_retrieval_embed_meta(value: dict | None) -> None:
    _retrieval_meta_tls.value = value


def _get_retrieval_embed_meta() -> dict | None:
    return getattr(_retrieval_meta_tls, "value", None)
_chunks_load_done = threading.Event()
_chunks_load_error: str | None = None


def env_int(name: str, default: int) -> int:
    """int из окружения с безопасным fallback (кривое значение не роняет запрос в 500)."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on", "y")


def _render_extended_ram() -> bool:
    """Render Standard+ (2 GiB RAM) - менее агрессивные лимиты, чем Free/Starter 512 MiB."""
    plan = (os.environ.get("RENDER_PLAN") or os.environ.get("RENDER_INSTANCE_TYPE") or "").strip().lower()
    if plan in (
        "standard",
        "pro",
        "pro_plus",
        "pro plus",
        "pro_max",
        "pro max",
        "pro_ultra",
        "pro ultra",
    ):
        return True
    return env_int("RENDER_RAM_MB", 0) >= 1800


def _apply_low_memory_defaults() -> None:
    """Render: дефолты RAM по plan (512 MiB vs Standard 2 GiB). Явные env не перезаписываются."""
    if not env_bool("RENDER", False):
        return
    if _render_extended_ram():
        profile: tuple[tuple[str, str], ...] = (
            ("RAG_MEMORY_SAVER", "1"),
            ("RAG_LEX_BM25_ALPHA", "1.0"),
            ("RAG_EMBED_POOL_MERGE", "0"),
            ("RAG_GEMINI_EMBED_RERANK", "0"),
            ("RAG_LEXICAL_MAX_CHARS", "4096"),
            ("CONSULT_PREWARM_PROTOCOL_ICD_INDEX", "0"),
            ("CONSULT_PREWARM_SUMMARY_ICD_INDEX", "0"),
            ("CONSULT_REVIEW_CACHE_MAX", "48"),
            ("CONSULT_REVIEW_MAX_CHUNKS", "8"),
            ("CONSULT_RENDER_L2_LITE", "1"),
            ("CONSULT_RENDER_L2_SKIP_LLM", "0"),
            ("CONSULT_TYPED_RETRIEVE", "0"),
            ("CONSULT_REVIEW_FORBID_FULL_CORPUS", "1"),
            ("CONSULT_RICH_CHUNKS_MAX_PER_PATH", "12"),
            ("CONSULT_REVIEW_MAX_PROTOCOL_PATHS", "4"),
            ("CONSULT_REVIEW_CACHE", "1"),
            ("CONSULT_RESPONSE_INCLUDE_HTML", "0"),
            ("PROTOCOL_SUMMARY_RAG_MERGE", "1"),
            ("CONSULT_L2_FAST", "1"),
            ("CONSULT_L2_EVIDENCE_MAX_PATHS", "3"),
            ("CONSULT_L2_EVIDENCE_MAX_CHARS", "8000"),
            ("CONSULT_L2_EVIDENCE_MAX_CHUNKS_PER_PATH", "2"),
            ("CONSULT_L2_ALIGN_MAX_CHUNKS_PER_PATH", "4"),
            ("RAG_LEX_MAX_CANDIDATES", "8000"),
            ("RAG_LEX_MAX_UNION", "20000"),
            ("RAG_RETRIEVE_CONCURRENCY", "2"),
            ("CONSULT_ALIGNMENT_ENABLED", "1"),
            ("RAG_LEX_INDEX_DEFER", "1"),
            ("CONSULT_CONCURRENCY", "1"),
            ("RAG_CHUNK_VOTE_RERETRIEVE", "0"),
            ("RAG_SEARCH_REQUIRE_ALLOWLIST_ON_RENDER", "1"),
            ("SEARCH_CONCURRENCY", "2"),
        )
    else:
        profile = (
            ("RAG_MEMORY_SAVER", "1"),
            ("RAG_LEX_BM25_ALPHA", "1.0"),  # без BM25-blend (~50-150 MiB)
            ("RAG_EMBED_POOL_MERGE", "0"),  # иначе BM25-индекс строится даже при alpha=1.0
            ("RAG_GEMINI_EMBED_RERANK", "0"),
            ("RAG_LEXICAL_MAX_CHARS", "4096"),
            ("CONSULT_PREWARM_PROTOCOL_ICD_INDEX", "0"),
            ("CONSULT_PREWARM_SUMMARY_ICD_INDEX", "0"),
            ("CONSULT_REVIEW_CACHE_MAX", "24"),
            ("CONSULT_REVIEW_MAX_CHUNKS", "6"),
            ("CONSULT_RENDER_L2_LITE", "1"),
            ("CONSULT_RENDER_L2_SKIP_LLM", "1"),
            ("CONSULT_TYPED_RETRIEVE", "0"),
            ("CONSULT_REVIEW_CACHE", "0"),
            ("CONSULT_RESPONSE_INCLUDE_HTML", "0"),
            ("PROTOCOL_SUMMARY_RAG_MERGE", "1"),
            ("RAG_LEX_MAX_CANDIDATES", "4000"),
            ("RAG_LEX_MAX_UNION", "12000"),
            ("RAG_RETRIEVE_CONCURRENCY", "1"),
            ("CONSULT_ALIGNMENT_ENABLED", "0"),
            ("RAG_LEX_INDEX_DEFER", "1"),
            ("CONSULT_CONCURRENCY", "1"),
            ("RAG_CHUNK_VOTE_RERETRIEVE", "0"),
            ("RAG_SEARCH_REQUIRE_ALLOWLIST_ON_RENDER", "1"),
            ("SEARCH_CONCURRENCY", "1"),
        )
    for key, val in profile:
        if not (os.environ.get(key) or "").strip():
            os.environ[key] = val


_apply_low_memory_defaults()

_retrieve_sem = threading.Semaphore(
    max(1, env_int("RAG_RETRIEVE_CONCURRENCY", 1 if env_bool("RENDER", False) else 4))
)
_consult_sem = threading.Semaphore(
    max(1, env_int("CONSULT_CONCURRENCY", 1 if env_bool("RENDER", False) else 3))
)
_search_sem = threading.Semaphore(
    max(1, env_int("SEARCH_CONCURRENCY", 2 if env_bool("RENDER", False) else 4))
)
_search_metrics_lock = threading.Lock()
_search_last_metrics: dict[str, Any] = {}


def _embed_protocol_nav_limit() -> int:
    """Сколько protocol_nav встраивать в /api/assist (0 = отключить). На Render по умолчанию 1."""
    return max(0, env_int("RAG_EMBED_PROTOCOL_NAV", 1 if env_bool("RENDER", False) else 3))


async def _run_consult_review_blocking(fn, /, *args, **kwargs):
    """Тяжёлый pipeline КЗ в worker-потоке - event loop свободен для /health."""
    def _job():
        with _consult_sem:
            try:
                return fn(*args, **kwargs)
            finally:
                if env_bool("RENDER", False):
                    gc.collect()

    return await asyncio.to_thread(_job)


def _consult_l2_skip_rag_warm() -> bool:
    """L2 lite/fast на manifest не требует полного lex-индекса (~55k чанков в RAM)."""
    from clinical_knowledge.lazy_rag_config import startup_mode

    if _consult_render_l2_skip_llm():
        return True
    if _consult_l2_fast_enabled() and _consult_render_l2_lite_enabled():
        return True
    if _consult_render_l2_lite_enabled() and startup_mode() == "manifest":
        return True
    return False


def _ensure_consult_rag_ready() -> None:
    """КЗ не стартует пока корпус грузится - иначе двойной пик RAM на Render."""
    if not env_bool("RENDER", False):
        return
    if _consult_l2_skip_rag_warm():
        _require_rag_loaded(
            wait=True,
            max_wait_sec=env_float("RAG_LOAD_WAIT_LITE_SEC", 28.0),
        )
        return
    _require_rag_loaded(wait=True)


def _consult_response_include_html() -> bool:
    return env_bool("CONSULT_RESPONSE_INCLUDE_HTML", not env_bool("RENDER", False))


def _consult_rag_second_pass_enabled() -> bool:
    """Второй RAG-pass: по умолчанию выкл на Render (лимит прокси ~100 с), вкл локально."""
    raw = os.environ.get("CONSULT_REVIEW_RAG_SECOND_PASS")
    if raw is not None and str(raw).strip():
        return env_bool("CONSULT_REVIEW_RAG_SECOND_PASS", True)
    if _consult_review_fast_mode():
        return False
    if env_bool("RENDER", False):
        return False
    return True


def _consult_review_fast_mode() -> bool:
    """Быстрый разбор КЗ: меньше вызовов Gemini, без embed-rerank, компактнее контекст."""
    raw = os.environ.get("CONSULT_REVIEW_FAST")
    if raw is not None and str(raw).strip():
        return env_bool("CONSULT_REVIEW_FAST", False)
    profile = (os.environ.get("CONSULT_REVIEW_PROFILE") or "").strip().lower()
    if profile == "fast":
        return True
    if profile == "full":
        return False
    if env_bool("RENDER", False):
        return True
    return False


def _consult_render_l2_lite_enabled() -> bool:
    """На Render (512Mi) полный retrieve() по корпусу часто даёт OOM; L1+chunks+LLM без RAG."""
    raw = os.environ.get("CONSULT_RENDER_L2_LITE")
    if raw is not None and str(raw).strip():
        return env_bool("CONSULT_RENDER_L2_LITE", True)
    return _consult_review_fast_mode() and env_bool("RENDER", False)


def _consult_render_l2_skip_llm() -> bool:
    """На Render (512Mi) L2 с LLM и загрузкой чанков часто даёт OOM - только L1-разбор."""
    if _consult_l2_fast_enabled():
        return False
    raw = os.environ.get("CONSULT_RENDER_L2_SKIP_LLM")
    if raw is not None and str(raw).strip():
        return env_bool("CONSULT_RENDER_L2_SKIP_LLM", True)
    return env_bool("RENDER", False) and _consult_render_l2_lite_enabled()


def _consult_l2_fast_enabled() -> bool:
    from clinical_knowledge.consult_l2_config import consult_l2_fast_enabled

    return consult_l2_fast_enabled()


def _consult_l2_mode_label() -> str:
    from clinical_knowledge.consult_l2_config import resolve_l2_mode

    return resolve_l2_mode(narrative=False)


def _consult_l2_feedback_latency() -> dict[str, Any]:
    try:
        from clinical_knowledge.feedback_store import feedback_dir as resolve_feedback_dir
        from clinical_knowledge.methodist_stats import iter_feedback_events

        fb = resolve_feedback_dir()
        latencies: list[int] = []
        for e in iter_feedback_events(fb):
            if e.get("event_type") != "kz_analysis":
                continue
            if str(e.get("tier") or "").upper() != "L2":
                continue
            lm = e.get("latency_ms")
            if isinstance(lm, (int, float)) and lm > 0:
                latencies.append(int(lm))
        if not latencies:
            return {"count": 0, "avg_ms": None}
        return {"count": len(latencies), "avg_ms": int(sum(latencies) / len(latencies))}
    except Exception:
        return {"count": 0, "avg_ms": None}


def _annotate_render_l2_limited(result: dict) -> None:
    """Пометка ответа: облачный L2 без языковой модели (экономия RAM)."""
    result["render_l2_limited"] = True
    note = (
        "Облачный L2 на Render: структурный разбор как L1, без оценки языковой модели "
        "(лимит памяти 512 MiB). Для полного L2 с цитатами модели - локальный сервер."
    )
    rev = result.get("review")
    if isinstance(rev, dict):
        prev = (rev.get("limitations_ru") or "").strip()
        rev["limitations_ru"] = (prev + " " + note).strip() if prev else note
    perf = result.get("consult_performance")
    if not isinstance(perf, dict):
        perf = {}
        result["consult_performance"] = perf
    perf["render_l2_skip_llm"] = True


def _consult_retrieve_embed_rerank() -> bool:
    raw = os.environ.get("CONSULT_REVIEW_EMBED_RERANK")
    if raw is not None and str(raw).strip():
        return env_bool("CONSULT_REVIEW_EMBED_RERANK", True)
    return not _consult_review_fast_mode()


def _consult_heuristic_digest_first(min_focus: int, heuristic: str) -> bool:
    """Пропустить Gemini-digest, если эвристика уже вытащила клиническое ядро из КЗ."""
    if _consult_review_fast_mode():
        return True
    if not env_bool("CONSULT_REVIEW_HEURISTIC_DIGEST_FIRST", True):
        return False
    return len((heuristic or "").strip()) >= min_focus


def _consult_env_int(name: str, default_full: int, *, default_fast: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is not None and str(raw).strip():
        try:
            return int(raw)
        except ValueError:
            return default_full
    if _consult_review_fast_mode() and default_fast is not None:
        return default_fast
    return default_full


def public_error_text(err: str | None) -> str | None:
    """Не раскрывать внутренние пути/детали в публичных ответах, если не включён DEBUG_ERRORS."""
    if not err:
        return err
    if env_bool("DEBUG_ERRORS", False):
        return err
    return "Внутренняя ошибка загрузки данных. Обратитесь к администратору."

PROTOCOL_META_PATH = ROOT / "protocol_meta.json"
STRUCTURED_INDEX_PATH = ROOT / "structured_index.json"
INDEX_CSV_PATH = ROOT / "index.csv"
QUALITY_BENCHMARK_PATH = ROOT / "data" / "quality_benchmark.json"
MINZDRAV_PROTOCOLS_INDEX_URL = (
    "https://minzdrav.gov.by/ru/dlya-spetsialistov/standarty-obsledovaniya-i-lecheniya/"
)
TRAINING_CASES_PATH = ROOT / "data" / "training_cases.json"
DEMO_CONSULT_TEXT_PATH = ROOT / "data" / "demo_consult_kz_sample.txt"
_index_csv_by_path: dict[str, dict[str, str]] | None = None


def _load_index_csv_by_path() -> dict[str, dict[str, str]]:
    global _index_csv_by_path
    if _index_csv_by_path is not None:
        return _index_csv_by_path
    out: dict[str, dict[str, str]] = {}
    if INDEX_CSV_PATH.is_file():
        with INDEX_CSV_PATH.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rel = (row.get("relative_path") or "").strip()
                if rel:
                    out[rel] = row
    _index_csv_by_path = out
    return out


def _protocol_display_title(path: str = "", title: str | None = None) -> str:
    """Читаемое название КП для UI и ответов API (без подчёркиваний из имён PDF)."""
    from clinical_knowledge.protocol_links import protocol_display_name

    reg = (title or "").strip() or None
    return protocol_display_name(path or None, fallback=reg or "", registry_title=reg)


def protocol_ui_meta_for_path(path: str) -> dict:
    """Метаданные КП для UI: год, пост МЗ, аудитория (index.csv и имя файла)."""
    from clinical_knowledge.protocol_audience import audience_hint_ru, infer_protocol_audience

    fn = Path(path).name if path else ""
    row = _load_index_csv_by_path().get(path.strip(), {}) if path else {}
    aud_csv = (row.get("audience") or "").strip().lower()
    audience: list[str] = []
    if aud_csv in ("pediatric", "child"):
        audience.append("детское население")
    elif aud_csv == "adult":
        audience.append("взрослое население")
    elif aud_csv == "mixed":
        audience.append("дети и взрослые")
    else:
        hint = infer_protocol_audience(path or "", fn)
        label = audience_hint_ru(hint)
        if label:
            audience.append(label)
    if "беремен" in _norm_query(fn):
        audience.append("беременность")
    nb = _norm_query(fn)
    post_csv = (row.get("has_post_mz") or "").strip().lower() == "yes"
    post_fn = any(x in nb for x in ("пост_мз", "post_mz", "постановление_мз", "постановление мз"))
    year = (row.get("years_in_filename") or "").strip() or None
    if not year:
        ym = re.search(r"(20\d{2})", fn)
        if ym:
            year = ym.group(1)
    mz_m = re.search(r"№\s*(\d+)", fn)
    return {
        "year": year,
        "post_mz": bool(post_csv or post_fn),
        "audience_hint": "; ".join(audience) if audience else None,
        "mz_number": mz_m.group(1) if mz_m else None,
    }


def protocol_ui_meta_bundle(paths: Iterable[str]) -> dict[str, dict]:
    return {p: protocol_ui_meta_for_path(p) for p in paths if p and str(p).strip()}

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
Если возраст явно взрослый (например «49 лет», ≥18 лет) - не включай в protocols детские КП: в списке должны остаться только path из входных фрагментов; если фрагменты только детские (маловероятно), опирайся на них осторожно и не выдавай детский протокол как основной без пометки.
Клиническая калибровка (обязательно):
- Опирайся на симптомы и формулировки из «Запрос пользователя». Не приписывай пациенту симптомов, которых там нет (в частности: насморк, боль в горле, ангина, ОРВИ, если их не указали в запросе). Не переноси симптомы из фрагментов протоколов в описание жалобы, если их не было в запросе.
- Если фрагменты явно про другую нозологию или орган (например вестибулярная патология или деформации позвоночника при жалобе на гайморит/синусы), не используй их как основу ответа и не повышай им confidence; укажи в match_reason несоответствие теме запроса или опусти такие протоколы из верхних позиций.
- ОРВИ, фарингит, тонзиллит, риносинусит и др. типичные ЛОР-причины - только если они явно следуют из запроса пользователя и/или из приведённых фрагментов; не подставляй их «по умолчанию».
- Редкие неотложные состояния (острый эпиглоттит, ретрофарингеальный абсцесс и т.п.) - только при явных красных флагах в тексте запроса (выраженная одышка, слюнотечение, невозможность глотать слюну, быстрое ухудшение) или если это прямо следует из фрагментов. Если пользователь указал нормальное дыхание без одышки - не ставь эпиглоттит первым в дифференциальный ряд и не формулируй ответ так, будто он наиболее вероятен.
- Не противоречь явным фактам из запроса (например «дыхание нормальное»).
- Summary - только краткое сопоставление запроса с отобранными протоколами. Не перечисляй в нём конкретные лекарства, дозы, схемы и перечни анализов/инструментальных исследований (их место - в развёрнутой выдержке по выбранному протоколу); не выдумывай детали вне фрагментов.
- Дифференциальный ряд (differential) в первую очередь из гипотез, согласующихся с тематикой входных фрагментов; расширяй до 3-5 пунктов только если запрос или фрагменты действительно допускают широкий дифференциал.
Верни ОДИН JSON-объект (без markdown, без текста до/после).
Схема полей:
{
  "summary": "…",
  "protocols": [{"path":"…","title":"…","match_reason":"…","confidence":"низкая|средняя|высокая","confidence_score":0.0}],
  "differential": ["…","…"],
  "questions_for_patient": [] или ["…","…"],
  "disclaimer": "Информация из протоколов; не замена очной консультации."
}
Не добавляй в JSON поле icd_codes - список кодов МКБ-10 формирует сервер.
ЖЁСТКИЕ ЛИМИТЫ (иначе ответ обрежется посередине):
- summary: РОВНО 2 предложения на русском, каждое заканчивается точкой. Вместе НЕ ДЛИННЕЕ 280 символов (с пробелами). Без тире в конце; последний символ - точка. Не формулируй как установленный диагноз; это краткое сопоставление запроса с протоколами.
- match_reason: не длиннее 70 символов, одно короткое предложение или фраза, законченная по смыслу.
- differential: только дифференциальные ГИПОТЕЗЫ для обсуждения с врачом; не окончательный диагноз. Не используй формулировки «диагноз:», «установлен», «подтверждён». В приоритете точность. По умолчанию 2 короткие строки (каждая 3-8 слов), порядок по убыванию вероятности. Добавь 3-й-5-й пункт только при явно широком дифференциале; не больше 5 строк.
- questions_for_patient: если хотя бы у одного протокола confidence_score равен 1.0 (полное соответствие запросу) - пустой массив []. Иначе ровно 2 коротких вопроса.
- protocols: все уникальные path из входных фрагментов; confidence_score 0.0-1.0. Каждый path и каждый протокол по названию (title) указывай только один раз - не дублируй одинаковые строки.
Если не хватает места - сожми формулировки, но НЕ обрывай слова и НЕ оставляй незаконченное предложение в summary."""

SYSTEM_JSON_LITE = """Ты помощник врача по клиническим протоколам Минздрава Республики Беларусь.
Фрагменты PDF ниже могут быть неполными. Не выдумывай факты вне фрагментов.
Если в запросе есть блок «=== Контекст пациента ===», учитывай возраст при выборе детских vs взрослых протоколов.
Клиническая калибровка (обязательно):
- Опирайся на «Запрос пользователя». Не приписывай симптомов, которых там нет.
- Если фрагмент явно про другую нозологию - не повышай confidence; укажи несоответствие в match_reason или опусти из верхних позиций.
- Не противоречь явным фактам из запроса.

Верни ОДИН JSON-объект (без markdown, без текста до/после).
Схема (только эти поля):
{
  "protocols": [{"path":"…","title":"…","match_reason":"…","confidence":"низкая|средняя|высокая","confidence_score":0.0}],
  "disclaimer": "Информация из протоколов; не замена очной консультации."
}
НЕ добавляй summary, differential, questions_for_patient, icd_codes.
- match_reason: не длиннее 70 символов, одна короткая фраза.
- protocols: все уникальные path из входных фрагментов; confidence_score 0.0-1.0; без дубликатов path/title."""

SYSTEM_JSON_LITE_RETRY = """Повтори: нужен ОДИН компактный JSON (без markdown).
Только поля protocols и disclaimer. Без summary, differential, questions_for_patient.
match_reason: до 55 символов. Сохрани все path из фрагментов. Не обрывай слова."""

SYSTEM_JSON_RETRY = """Повтори задачу: нужен ОДИН компактный JSON (без markdown).
Не добавляй симптомы носа/горла/ОРВИ, если их не было в запросе пользователя. Эпиглоттит и др. редкие неотложные - только при красных флагах или прямо в фрагментах; при нормальном дыхании не веди с эпиглоттита.
Не дублируй один и тот же протокол в protocols (один path / один title).
Предыдущая попытка оборвалась по длине. Сделай ещё короче:
- summary: РОВНО 2 коротких предложения, ВМЕСТЕ максимум 220 символов, последний символ - точка.
- match_reason: до 55 символов на протокол.
- differential: только гипотезы, не окончательный диагноз; без формулировок «диагноз установлен». 2 коротких пункта (или до 5 только если дифференциал широкий); по убыванию вероятности; questions_for_patient: [] если есть протокол с confidence_score 1.0, иначе 2 коротких вопроса.
Сохрани все path из фрагментов. Не обрывай слова."""

ASSIST_USER_CONTEXT_GUIDE = """Как читать фрагменты выше:
- Они перечислены в порядке отбора; поля score и lexical_score отражают силу совпадения с поисковым запросом (ориентир, не клинический скоринг).
- При противоречии между фрагментами разных протоколов приоритет - согласованность с формулировкой «Запрос пользователя» и с рубрикой фрагмента; не смешивай тактику из явно нерелевантного фрагмента в summary и match_reason.
- Детальные назначения, обследования и режимы лечения не раскрывай в summary JSON - они доступны пользователю при раскрытии протокола (вторая ступень)."""

SYSTEM_EXTRACT = """Ты помощник врача. По фрагментам клинического протокола Минздрава Республики Беларусь извлеки факты, относящиеся к запросу пользователя.
Верни ОДИН JSON-объект (без markdown, без текста до/после).
Схема:
{
  "diagnosis": "диагнозы, состояния, показания протокола по тексту (1-5 предложений)",
  "treatment_methods": ["метод или этап лечения - по тексту протокола"],
  "medications": ["группы препаратов или МНН, если названы во входном тексте - без выдуманных доз"],
  "note": "кратко: чего нет в фрагментах или что требует очной консультации"
}
Не придумывай препараты, дозы и процедуры, которых нет во входном тексте."""

SYSTEM_EXTRACT_FULL = """Ты помощник врача. По ПОЛНОМУ тексту фрагментов клинического протокола Минздрава Республики Беларусь извлеки структурированные сведения, релевантные запросу пользователя.
Запрос хорошо соответствует протоколу (оценка модели обычно ≥80%); это не обязательно «идеальные 100%» - дай развёрнутый практичный разбор строго по тексту протокола.
В списках investigations, medications, treatment_methods и recommendations сначала помещай пункты, прямо относящиеся к формулировке запроса пользователя (симптомы, этап, цель обращения), затем - остальные релевантные пункты протокола по возрастанию общности.

Различай четыре блока (если в тексте нет данных - пустой массив [] или пустая строка ""):
- investigations: только диагностика и обследование (анализы, инструментальные методы, осмотры, критерии до постановки диагноза). Пример: «Общий анализ крови», «УЗИ органов брюшной полости», «рентгенография» - если это в тексте.
- medications: только лекарственные группы, МНН, режимы из текста (не дублируй сюда немедикаментозное лечение).
- treatment_methods: немедикаментозное и медикаментозное лечение как этапы/тактика (операции, режим, физиотерапия, схемы терапии словами протокола). Не копируй сюда дословно длинные таблицы доз - кратко по строкам.
- monitoring_frequency: только кратность и сроки наблюдения (через сколько недель визит, диспансеризация раз в год и т.п.) - одна строка или короткие фразы через «;».

monitoring_followup - отдельно: когда срочно обращаться, реабилитация, прочие формулировки наблюдения без дублирования кратности из monitoring_frequency; если вся наблюдательная информация только про сроки визитов - оставь monitoring_followup пустой строкой "".

Если протокол содержит алгоритмы (ветвления «если/то», пошаговые действия, эскалация помощи), заполни care_algorithms как список структур:
- title: название алгоритма/сценария;
- entry_conditions: короткие условия старта;
- steps: пошаговые действия в порядке выполнения (короткие пункты).
Если алгоритмов нет - верни [].

Если в тексте есть нумерация разделов/подпунктов («п. 2.1», «раздел 3») - по возможности укажи короткую отсылку в пункте списка (только если она есть в OCR).

Верни ОДИН JSON-объект (без markdown, без текста до/после).
Схема:
{
  "diagnosis": "диагнозы, состояния, показания (2-8 предложений)",
  "investigations": ["пункты обследования - по тексту протокола"],
  "medications": ["группы препаратов, МНН, режимы - только если есть во входном тексте"],
  "treatment_methods": ["этапы и методы лечения - по тексту протокола"],
  "monitoring_frequency": "кратность наблюдения одной строкой или пустая строка",
  "recommendations": ["рекомендации и алгоритм действий для врача/пациента - по тексту"],
  "monitoring_followup": "прочие формулировки наблюдения, когда обращаться - если уместно; иначе пустая строка",
  "care_algorithms": [{"title":"название алгоритма","entry_conditions":["условие старта"],"steps":["шаг 1","шаг 2"]}],
  "contraindications": "противопоказания и ограничения - если названы во фрагментах, иначе пустая строка",
  "note": "чего нет в фрагментах; необходимость очной консультации"
}
Не придумывай дозировки, препараты и процедуры, которых нет во входном тексте."""

SYSTEM_EXTRACT_GAP_SCAN = """Ты помощник врача. Ниже - полный текст фрагментов клинического протокола Минздрава Республики Беларусь (и при необходимости - выдержка из индекса).
В первом проходе не были извлечены или остались пустыми разделы: {fields_ru}.
Задача: ещё раз внимательно прочитай ВЕСЬ текст ниже и найди в нём сведения, относящиеся к этим разделам.
Сопоставь с формулировкой запроса пользователя: для investigations и medications в первую очередь извлеки то, что напрямую относится к жалобе/ситуации из запроса, затем прочие пункты из протокола.
Особое внимание: таблицы (каждая осмысленная строка таблицы с обследованием, препаратом или режимом - отдельный короткий пункт списка, если ячейки читаемы), перечни с маркерами, подпункты в скобках.
Каждый пункт - не длиннее ~2 строк текста; при необходимости разбей на несколько пунктов.
Верни ОДИН JSON-объект (без markdown, без текста до/после).
Включай в ответ ТОЛЬКО те ключи из этого списка, которые были пусты: {keys_json}.
- investigations, medications, treatment_methods - массивы строк (короткие пункты);
- monitoring_frequency - одна строка или пустая строка "".
Если в тексте протокола для раздела действительно нет данных - верни [] или "".
Не придумывай препараты, дозы и процедуры, которых нет во входном тексте."""

SYSTEM_EXTRACT_NON_PROTOCOL = """Ты помощник врача. По сути запроса и названию протокола ниже для разделов, которые в тексте протокола не найдены или пусты, дай краткие обобщённые общеклинические ориентиры для врача (не цитата из КП).
Правила:
- Не выдавай это за текст протокола; формулируй осторожно и обобщённо.
- Каждая строка в списках и строка кратности наблюдения ДОЛЖНЫ начинаться с точной пометки «[не из протокола]» (с квадратными скобками).
- Не указывай конкретные дозировки и схемы; не придумывай названия препаратов, которых нет в общих клинических стандартах.
- Если раздел заполнить нельзя безопасно - верни [] или "".
Специальность (рубрика): {spec}
Название протокола: {title}
Запрос пользователя:
{query}
Заполнить только пустые поля: {fields_ru}
Верни ОДИН JSON (без markdown) с ключами ТОЛЬКО из: {keys_json}
Типы: investigations, medications, treatment_methods - массивы строк; monitoring_frequency - одна строка."""

SYSTEM_CLASSIFY = """По краткому медицинскому запросу пациента выбери до трёх рубрик клинических протоколов (slug), которым соответствует ситуация.
Верни ОДИН JSON: {"categories": ["slug1"], "note": "одно короткое предложение"}
slug ТОЛЬКО из этого списка (копируй точно):
""" + ", ".join(sorted(ALLOWED_SPECIALTY_SLUGS)) + """
Если нельзя уверенно сопоставить - верни "categories": []."""

SYSTEM_QUERY_SPELLFIX = """По фрагменту текста жалобы (русский) исправь только орфографию и очевидные опечатки, в том числе в медицинских терминах (например лишние/пропущенные буквы).
Не меняй смысл, не добавляй симптомы и диагнозы, не сокращай и не перефразируй свободно.
Верни ОДИН JSON-объект (без markdown):
{"corrected": "<тот же текст целиком, с исправлениями или без изменений>"}"""

SYSTEM_CLINICAL_QUERY_REFINE = """Ты помощник врача. Ниже - текст жалобы/клинического запроса (русский) для автоматического поиска по клиническим протоколам и справочнику МКБ-10.
Задача: привести формулировки к общепринятой клинической терминологии (как в русскоязычных названиях МКБ-10 и протоколах), сохранив смысл и объём жалобы.
Правила:
- Не добавляй симптомы, жалобы, диагнозы и обстоятельства, которых нет во входном тексте.
- Не приписывай пациенту пол, возраст и сопутствующие болезни, если их нет во входе (если в дополнительном контексте ниже указаны возраст/пол - можно использовать только для согласования формулировок «ребёнок/взрослый», без выдумок).
- Разговорные названия замени на клинические эквиваленты там, где это однозначно (например «гайморит» → можно уточнить «острый/хронический верхнечелюстной синусит» только если степень остроты явно следует из текста; иначе оставь «гайморит» или нейтрально «синусит верхнечелюстной пазухи»).
- Не заменяй «температура»/«жар» на «лихорадка», если в тексте есть другие симптомы (кашель, насморк, боль в горле и т.п.) - оставь исходные слова жалобы.
- Не ставь окончательный клинический диагноз; это подготовка текста к поиску, не заключение.
- Исправь опечатки в медицинских терминах.
- Сохрани структуру: если несколько предложений - не сливай в одно без необходимости; итог не длиннее исходного более чем на ~30% (не раздувай).
Верни ОДИН JSON-объект (без markdown):
{"refined": "<итоговый текст>", "applied": true или false, "note": "<одно короткое предложение или пустая строка: что изменилось; если изменений нет - пусто>"}
Поле applied: false, если текст уже корректен и ты вернул его без существенных правок (допустимы микроисправления - тогда applied: true и кратко в note)."""

SYSTEM_ICD_POOL_SELECT = """Ты помощник врача. По клиническому запросу (жалобы, симптомы) выбери до 5 кодов МКБ-10 ТОЛЬКО из списка allowed ниже.
Запрещено: коды вне списка, выдуманные обозначения, текст вне JSON.
Дублируй поле code ТОЧНО как в списке (латиница и цифры).
Правила отбора:
- При острой жалобе пациента НЕ выбирай коды «Последствия…», «отдалённые последствия», Z-коды семейного анамнеза.
- Глава Y (лекарства, побочные эффекты) - НЕ выбирай при обычных жалобах (насморк, горло, кашель). Y55 «насморк» - это лекарство, не диагноз; при ОРВИ выбирай J00/J06/J02/R07.
- «Давление 140/90» и гипертония - I10/I11/R03, не «сдавление» органов.
- Инфекция мочевых путей у взрослой женщины без беременности - N39/N30, не O86/P39.
- Алкогольная зависимость/абстиненция - F10, не Z81 (семейный анамнез).
- Ожог кипятком/термический - T20-T32, не L55 (солнечный).
- Анафилаксия/укус пчелы - T78/W57, не A25 (крысы).
- Боль в горле и насморк - J06.9 или J00, не Y и не K (желудок).
Верни ОДИН JSON-объект (без markdown):
{"codes":[{"code":"J20.9","rationale":"одно короткое предложение"}]}
Если ни один код из списка не подходит - {"codes":[]}."""

KZ_MATRIX_SECTIONS = (
    "Жалобы и анамнез",
    "Объективный статус",
    "Диагноз и коды МКБ-10",
    "Обследование",
    "Лечение и назначения",
    "Наблюдение и контроль",
    "Направления и консультации специалистов",
)

SYSTEM_KZ_MATRIX = """Ты методист-врач. По тексту клинического протокола Минздрава Республики Беларусь составь структуру «что должно быть отражено в консультативном заключении» для данного случая.

Правила:
- Каждый пункт items должен опираться на текст протокола ниже; protocol_excerpt - короткая дословная вырезка (до 220 символов) или пустая строка, если нельзя процитировать.
- protocol_ref - ссылка на раздел/пункт протокола, если есть в тексте (например «п. 3.2»), иначе пустая строка.
- obligation: required - если протокол прямо требует/рекомендует как обязательное; recommended - желательно; conditional - по показаниям/ветке алгоритма; not_applicable - если для данного запроса раздел КЗ не применим (редко).
- icd_related: true, если пункт прямо связан с кодами МКБ из запроса (перечислены ниже).
- Не выдумывай препараты, дозы и обследования, которых нет в тексте протокола.
- kz_section - ТОЛЬКО из фиксированного списка (скопируй точно):
  """ + "; ".join(KZ_MATRIX_SECTIONS) + """

Верни ОДИН JSON (без markdown):
{
  "icd_codes": ["M32.9"],
  "protocol_title": "…",
  "summary_ru": "2-3 предложения: на что обратить внимание при оформлении КЗ",
  "sections": [
    {
      "kz_section": "Обследование",
      "items": [
        {
          "text": "краткая формулировка для врача",
          "obligation": "required|recommended|conditional|not_applicable",
          "protocol_ref": "",
          "protocol_excerpt": "",
          "icd_related": false
        }
      ]
    }
  ],
  "disclaimer_ru": "Ориентир по протоколу; не замена очного приёма и не юридическая экспертиза."
}
Включи только разделы, по которым в протоколе есть релевантные сведения для запроса; пустые разделы не добавляй."""

SYSTEM_CONSULTATION_TEMPLATE = """Ты помощник врача. По развёрнутой выдержке из клинического протокола Минздрава Республики Беларусь (структура JSON ниже) и по сути запроса пользователя составь текстовый ШАБЛОН консультативного заключения.
Правила:
- Опирайся только на поля выдержки и на запрос; не выдумывай диагнозы, препараты, дозы и процедуры, которых нет во входных данных.
- Если передан блок selected_facts_payload (структурированные выбранные пункты), это ПРИОРИТЕТНЫЙ источник: отрази каждый выбранный пункт в профильном разделе заключения.
- Не пропускай выбранные пункты. Если пункт нельзя включить дословно, включи клинически эквивалентную формулировку без потери смысла.
- Если в «Запрос пользователя» или в блоке «Контекст пациента» указаны возраст и пол - подставь их в разделы «Жалобы» и «Анамнез» в связном тексте (например: «Пациент 49 лет, мужского пола, предъявляет жалобы…»). Не используй плейсхолдер [ФИО, возраст] или аналог, если возраст и пол уже известны из контекста. Фамилию, имя, отчество не выдумывай: если ФИО в данных нет - формулируй без ФИО («Пациент», «Пациентка» + возраст + пол).
- Плейсхолдеры в квадратных скобках должны быть ПОНЯТНЫМИ: не повторяй без разбора общую фразу [уточнить при осмотре]. Вместо этого указывай, ЧТО именно внести, например: [перечислить сопутствующие заболевания, аллергоанамнез, перенесённые операции], [описать объективный статус: общее состояние, местный статус прямой кишки и промежности], [указать сроки и режим наблюдения после лечения], [указать ограничения по выбору метода лечения при отсутствии данных в выдержке]. Если контекст узкий - короткая подсказка в скобках допустима.
- Общую форму [уточнить при осмотре] используй только если нельзя сформулировать конкретнее.
- Акценты разделяй: строки с «ВАЖНО:» - для критичных предупреждений; строки с «Внимание:» - для напоминаний и уточнений (не столь срочных). Отдельный абзац или пункт списка, без markdown.
- Стиль: официально-деловой, медицинский, пригодный для МИС или печати.
- Структура: разделы с заголовками в одну строку с двоеточием в конце, например: Жалобы: / Анамнез: / Объективно: / Диагноз по протоколу Минздрава РБ: / Рекомендации по протоколу: / Наблюдение и контроль: / Дополнительно:
- ОБЯЗАТЕЛЬНО выведи текст ПОЛНОСТЬЮ: не обрывай на середине слова, фразы или раздела; заверши каждый раздел; не используй «…» вместо целых абзацев протокола. Если объём большой - всё равно доведи структуру до конца.
- ЗАПРЕЩЕНО оформление в markdown: не используй звёздочки *, **, подчёркивания для выделения, решётки #, обратные кавычки. Пиши обычным текстом.
- Списки: строки с дефисом и пробелом в начале (- пункт) или нумерация 1. 2.
- Не дублируй заголовок «Консультативное заключение» в тексте - с него начинать не нужно (он будет на экране отдельно).
- В конце кратко: шаблон не заменяет очный осмотр и оформление документации лечащим врачом.
Верни ТОЛЬКО текст шаблона, без вступления «вот шаблон»."""

SYSTEM_CONSULTATION_REFINE = """Ты помощник врача. Ниже - черновик консультативного заключения (часть полей пользователь уже заполнил вместо плейсхолдеров в квадратных скобках). Также даны дополнительные сведения от пользователя (если есть).
Задача: выдай ПОЛНЫЙ итоговый текст заключения, согласованный с развёрнутой выдержкой из протокола Минздрава РБ (JSON ниже) и запросом. Дополни недостающие разделы по протоколу; где данных по-прежнему нет - оставь плейсхолдер с КОНКРЕТНОЙ подсказкой в скобках (что внести), избегая безликого [уточнить при осмотре], если можно уточнить формулировку.
- Если в запросе или в «Контекст пациента» есть возраст и пол - сохрани их в «Жалобы» и «Анамнез»; не возвращай плейсхолдер [ФИО, возраст], если эти данные уже заданы. ФИО не выдумывай.
- Критичное - с префиксом «ВАЖНО:»; напоминания - с «Внимание:».
- Не сокращай и не обрывай текст посередине; заверши все разделы.
- Не выдумывай факты, которых нет в выдержке, запросе или в дополнениях пользователя.
- ЗАПРЕЩЕНО markdown (*, **, #, обратные кавычки).
- Не дублируй заголовок «Консультативное заключение» в теле текста.
Верни ТОЛЬКО полный текст заключения."""

SYSTEM_CONFIDENCE_REFINE = """Ты помощник врача. По запросу и кратким сведениям о протоколе оцени, насколько протокол соответствует сути жалобы (0.0-1.0).
Верни ОДИН JSON без markdown: {"scores":[{"path":"…","confidence_score":0.0}]}
Копируй path точно из списка ниже; не добавляй протоколы вне списка."""

SYSTEM_CONSULT_REVIEW_JSON = """Ты методист-врач. Ниже - 1) текст(ы), извлечённый из одного или нескольких PDF клинических документов после приёма: консультативное заключение (КЗ), медицинский осмотр или консультация (если блоков несколько и они помечены «=== ЗАКЛЮЧЕНИЕ …», это может быть несколько приёмов - учитывай согласованность между ними и возможные временные линии);
2) выдержки из клинических протоколов Минздрава Республики Беларусь (автоматический поисковый отбор по смыслу, неполный и может не охватывать все разделы документа).

Задача: краткое резюме (summary_ru) и ограничения проверки. Детальные критерии с баллами формируются детерминированно отдельно.
Название бланка (КЗ / медосмотр / консультация) не важно - оценивай по содержимому: жалобы, диагноз, рекомендации, обследования, лечение.

ИСТОЧНИКИ ПО БЛОКАМ (строго):
- Жалобы, анамнез, объективный статус: только полнота описания в документе визита. НЕ сравнивать с протоколами.
- Диагноз и МКБ-10: справочник МКБ, не клинический протокол.
- Обследование и лечение: только выдержки КП (диагностика, фармакотерапия).
- Наблюдение: КП (диспансеризация) и НПА.

Строгие правила:
- Не выдавай юридических или МЭЭ-вердиктов.
- Не придумывай цитаты.
- Все пояснения на русском.
- Если ниже передан блок «ДЕТЕРМИНИРОВАННАЯ ПРОВЕРКА ПО ПРАВИЛАМ ПРОТОКОЛА» - учти его в summary_ru и limitations_ru.

Верни ОДИН JSON-объект (без markdown) строго следуя схеме:
{"overall_compliance_pct": <целое 0-100, ориентир>,
 "summary_ru": "<2-4 предложения>",
 "criteria": [],
 "limitations_ru": "<что не удалось проверить>",
 "disclaimer_ru": "Оценка ориентировочная; не замена МЭЭ и очной экспертизы.",
 "protocol_paths_used": [<строки путей протоколов из выдержек, если удалось>]}
Поле criteria оставь пустым массивом []."""

SYSTEM_CONSULT_L2_NARRATIVE = """Ты методист-врач. Ниже - evidence pack (выдержки из клинических протоколов) и список пробелов сверки документа после приёма (КЗ / медосмотр / консультация) с протоколом.

Задача: один абзац (3-5 предложений) для методиста - что проверить вручную, на что обратить внимание. Не дублируй проценты соответствия. Не выдумывай фактов вне выдержек.

Верни ТОЛЬКО текст абзаца на русском, без markdown и без JSON."""

SYSTEM_CONSULT_PDF_FOR_PROTOCOL_SEARCH = """Ты помощник врача. Ниже - текст, машинно извлечённый из PDF клинического документа после приёма: консультативное заключение, медицинский осмотр или консультация (может содержать шапку организации, реквизиты, ФИО). Тип бланка не важен - ориентируйся на клиническое содержимое.

Задача: выделить суть клинического случая для ПОИСКА по текстам клинических протоколов Минздрава Республики Беларусь (подбор фрагментов документов).
- Возьми из текста только клинически значимое: жалобы и анамнез (если есть), объективный статус/осмотр, результаты обследований с короткой формулировкой находок, ключевые диагнозы из заключения, этап ведения (наблюдение, подготовка к операции, послеоперационный период и т.п.), упомянутые в документе коды МКБ-10 и их текстовые формулировки рядом.
- Не включай названия организаций, адреса, телефоны, рекламные блоки, штампы «выдан пациенту», подписи без клинической сути, если они не задают содержание помощи.
- Не добавляй диагнозов, симптомов и назначений, которых НЕТ во входном тексте.
- Если в документе указаны возраст и пол - кратко включи их (важно для детских vs взрослых протоколов).

Итог - один связный текст на русском, плотный по терминам (как в выписках), пригодный как текст запроса к поиску по протоколам: минимум 100 символов, если медицинский смысл в документе вообще есть; максимум 2000 символов.

Верни ОДИН JSON (без markdown, без текста до/после):
{"clinical_search_text": "<…>", "confidence": "high"|"medium"|"low"}
Поле confidence: high если структура заключения ясна и ядро случая восстановимо; medium если часть сведений потеряна при OCR/обрывах; low если текст почти только реквизиты или противоречив. Если клинической сути нет - "clinical_search_text": "" и confidence low."""

SYSTEM_CONSULT_RAG_SECOND_PASS_QUERY = """Ты помощник врача. Первый автоматический поиск по текстам протоколов дал средние баллы совпадения - нужен более прицельный текст запроса для ВТОРОГО поиска по тому же корпусу.

Ниже:
1) клинический текст запроса из документа после приёма - КЗ, медосмотр или консультация (уже без шапки учреждения);
2) краткое напоминание фокуса из PDF (если есть);
3) кандидатные протоколы - название и начало найденного фрагмента (это может быть промах; не принимай всё как верное).

Задача: составить ОДНУ связную русскоязычную строку для полнотекстового поиска по протоколам (плотные термины, диагнозы, этапы, ключевые обследования), без выдувания фактов, которых не было ни в заключении, ни в названиях/фрагментах кандидатов.
- Если кандидаты явно офтальмология, а в заключении пульмонология - не смешивай; опирайся на заключение.
- Если первые найденные протоколы могут быть нерелевантны - переформулируй запрос по тексту заключения и добавь только уместные синонимы из их названий/фрагментов.
- Итог: 120-2200 символов либо пустая строка, если нечего уточнить.

Верни ОДИН JSON без markdown:
{"refined_search_text": "<…>", "draft_note": "<одно короткое предложение почему уточнили или оставили пусто>", "confidence_in_candidates": "high"|"medium"|"low"}
"""


def _jsonl_chunk_files() -> list[Path]:
    """Порядок: RAG_CHUNKS_JSONL → RAG_CHUNKS_JSONL_GLOB → rich_chunks → corpus_chunks_parts."""
    base = _chunks_data_root()
    one = (os.environ.get("RAG_CHUNKS_JSONL") or "").strip()
    if one:
        p = Path(one).expanduser()
        if not p.is_file():
            raise SystemExit(f"RAG_CHUNKS_JSONL: файл не найден: {p}")
        return [p.resolve()]
    gl = (os.environ.get("RAG_CHUNKS_JSONL_GLOB") or "").strip()
    if gl:
        paths = sorted(base.glob(gl))
        if not paths:
            raise SystemExit(f"RAG_CHUNKS_JSONL_GLOB: нет файлов по шаблону {gl!r} в {base}")
        return paths
    if env_bool("RAG_USE_RICH_CHUNKS", True):
        for candidate in (
            base / "output" / "rich_chunks" / "rich_chunks.final.jsonl",
            ROOT / "output" / "rich_chunks" / "rich_chunks.final.jsonl",
            base / "output" / "rich_chunks" / "rich_chunks.v2.jsonl",
            ROOT / "output" / "rich_chunks" / "rich_chunks.v2.jsonl",
            base / "output" / "rich_chunks" / "rich_chunks.jsonl",
            ROOT / "output" / "rich_chunks" / "rich_chunks.jsonl",
        ):
            if candidate.is_file():
                return [candidate.resolve()]
    return sorted(base.glob(CORPUS_CHUNKS_PARTS_GLOB))


def _load_summary_rag_chunks() -> list[dict]:
    """Подмешивает summary_chunks.jsonl в RAG-индекс (Protocol Summary Cards)."""
    if os.environ.get("PROTOCOL_SUMMARY_RAG_MERGE", "1").strip().lower() in ("0", "false", "no"):
        return []
    raw_path = (os.environ.get("PROTOCOL_SUMMARY_RAG_JSONL") or "").strip()
    path = Path(raw_path) if raw_path else ROOT / "data" / "protocol_summaries" / "summary_chunks.jsonl"
    if not path.is_file():
        return []
    out: list[dict] = []
    try:
        from clinical_knowledge.protocol_summary.icd_index import _protocol_id_to_local_path

        id_to_path = _protocol_id_to_local_path()
    except Exception:
        id_to_path = {}
    try:
        with path.open(encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                ch = dict(row)
                pid = ch.get("protocol_id") or "summary"
                cid = ch.get("chunk_id") or ch.get("section_type") or "chunk"
                ch["path"] = f"summary://{pid}/{cid}"
                catalog = id_to_path.get(str(pid), "")
                if catalog:
                    ch["catalog_source_path"] = catalog.replace("\\", "/")
                ch["lex_text"] = ch.get("text") or ""
                ch["title"] = ch.get("condition_name") or ch.get("section_type") or ""
                ch["category"] = ch.get("rubric_slug") or ""
                ch["chunk_source"] = "summary_chunks"
                ch["generated_from_summary"] = True
                ch["chunk_index"] = idx
                out.append(ch)
    except Exception:
        return []
    return out


def _memory_saver_enabled() -> bool:
    """Полный lex (embedding_ready_text при отличии от text) только если явно выключен saver.

    RAG_MEMORY_SAVER=1 - без дубля embedding_ready_text (экономия ~100-200 MiB на rich corpus).
    На Render с Persistent Disk (RAG_CHUNKS_DIR) saver включается по умолчанию, если не задано
    RAG_MEMORY_SAVER=0.
    """
    v = (os.environ.get("RAG_MEMORY_SAVER") or "").strip().lower()
    if v in ("0", "false", "no"):
        return False
    if v in ("1", "true", "yes"):
        return True
    if env_bool("RENDER", False):
        return True
    return bool((os.environ.get("RAG_CHUNKS_DIR") or "").strip())


def _load_chunks_from_jsonl(part_paths: list[Path]) -> list[dict]:
    """Корпусный pipeline: строки JSONL → формат retrieve() / gather_protocol_text.

    Без промежуточного списка «всех сырых строк» - сразу группировка по path и только
    нужные поля (экономия RAM). lex_text хранится только если отличается от text.
    """
    from clinical_knowledge.rich_chunk_search import should_skip_rich_chunk_row

    memory_saver = _memory_saver_enabled()
    keep_struct = env_bool("RAG_KEEP_STRUCT", True)
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
                is_rich = bool(row.get("doc_id") or row.get("protocol_title"))
                if is_rich:
                    row["rich_chunk"] = True
                    if should_skip_rich_chunk_row(row):
                        continue
                text = (row.get("text") or "").strip()
                slim: dict = {
                    "page_from": int(row.get("page_from") or 0),
                    "page_to": int(row.get("page_to") or 0),
                    "chunk_id": row.get("chunk_id"),
                    "text": text,
                    "chunk_type": (row.get("chunk_type") or "body").strip() or "body",
                }
                if is_rich:
                    slim["rich_chunk"] = True
                    spec = (row.get("specialty_slug") or "").strip()
                    if spec:
                        slim["specialty_slug"] = spec
                    pops = row.get("population")
                    if isinstance(pops, list) and pops:
                        slim["chunk_population"] = [str(x) for x in pops][:8]
                if keep_struct:
                    sec_path = row.get("section_path")
                    if isinstance(sec_path, list) and sec_path:
                        slim["section_path"] = [str(x) for x in sec_path][:6]
                    sec_title = (row.get("section_title") or "").strip()
                    if not sec_title and isinstance(sec_path, list) and sec_path:
                        sec_title = str(sec_path[-1]).strip()
                    if sec_title:
                        slim["section_title"] = sec_title[:160]
                    pts = row.get("point_numbers")
                    if isinstance(pts, list) and pts:
                        slim["point_numbers"] = [str(x) for x in pts][:12]
                    icd = row.get("icd10_codes")
                    if isinstance(icd, list) and icd:
                        slim["icd10_codes"] = [str(x).upper() for x in icd][:16]
                    icd_w = row.get("icd10_weights")
                    if isinstance(icd_w, dict) and icd_w:
                        slim["icd10_weights"] = {
                            str(k).upper(): int(v)
                            for k, v in list(icd_w.items())[:16]
                            if k
                        }
                if not memory_saver:
                    ert = (row.get("embedding_ready_text") or "").strip()
                    if ert and ert != text:
                        if lex_cap > 0 and len(ert) > lex_cap:
                            ert = ert[:lex_cap]
                        slim["lex_text"] = ert
                elif lex_cap > 0 and len(text) > lex_cap:
                    slim["lex_text"] = text[:lex_cap]
                if not memory_saver:
                    emb = row.get("embedding")
                    if isinstance(emb, list) and len(emb) >= 8 and isinstance(emb[0], (int, float)):
                        slim["embedding"] = [float(x) for x in emb]
                        em = (row.get("embedding_model") or "").strip()
                        if em:
                            slim["embedding_model"] = em
                        if row.get("embedding_dim"):
                            slim["embedding_dim"] = int(row["embedding_dim"])
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
            for fld in ("section_path", "section_title", "point_numbers", "icd10_codes", "icd10_weights"):
                if fld in row:
                    rec[fld] = row[fld]
            if row.get("page_from"):
                rec["page_from"] = row["page_from"]
            if row.get("page_to"):
                rec["page_to"] = row["page_to"]
            if row.get("rich_chunk"):
                rec["rich_chunk"] = True
                if row.get("specialty_slug"):
                    rec["category"] = row["specialty_slug"]
                if row.get("chunk_population"):
                    rec["chunk_population"] = row["chunk_population"]
            if isinstance(row.get("embedding"), list) and not memory_saver:
                rec["embedding"] = row["embedding"]
                if row.get("embedding_model"):
                    rec["embedding_model"] = row["embedding_model"]
                if row.get("embedding_dim"):
                    rec["embedding_dim"] = row["embedding_dim"]
            out.append(rec)
    return out


def _use_jsonl_chunks() -> bool:
    """По умолчанию - JSONL-чанки (корпус), если явно не задан RAG_CHUNKS_SOURCE=json."""
    src = (os.environ.get("RAG_CHUNKS_SOURCE") or "").strip().lower()
    if src in ("json", "legacy", "chunks.json"):
        return False
    if src in ("jsonl", "corpus", "parts", "1", "true", "yes"):
        return True
    # авто: есть части corpus → jsonl; иначе chunks.json
    return bool(_jsonl_chunk_files())


def _enrich_chunks_from_index() -> None:
    """Заголовок и рубрика из protocols.json / protocol_meta для routing и retrieve."""
    catalog: dict[str, dict] | None = None
    for ch in _chunks:
        p = ch.get("path") or ""
        if not p:
            continue
        pr = _protocols_by_path.get(p) or {}
        pm = _protocol_meta.get(p) or {}
        if not (ch.get("title") or "").strip():
            raw_title = (pr.get("title") or pm.get("title") or "").strip()
            ch["title"] = _protocol_display_title(p, raw_title or Path(p).stem)
        if not (ch.get("category") or "").strip():
            ch["category"] = (pr.get("category") or pm.get("category") or "").strip()
        if ch.get("rich_chunk") and not ch.get("icd10_weights"):
            if catalog is None:
                try:
                    from clinical_knowledge.protocol_catalog import load_protocol_catalog

                    catalog = load_protocol_catalog()
                except Exception:
                    catalog = {}
            norm_p = str(p).replace("\\", "/")
            cat_row = (catalog or {}).get(norm_p) or {}
            weights = cat_row.get("icd10_weights") or {}
            if isinstance(weights, dict) and weights:
                ch["icd10_weights"] = {
                    str(k).upper(): int(v) for k, v in list(weights.items())[:16] if k
                }


def _load_metadata_common() -> None:
    global _protocols_by_path, _protocol_meta, _structured_by_path, _routing
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


def _ensure_lazy_chunk_store():
    global _lazy_chunk_store, _path_manifest
    if _lazy_chunk_store is not None:
        return _lazy_chunk_store
    from clinical_knowledge.chunk_store import LazyChunkStore
    from clinical_knowledge.corpus_path_manifest import CorpusPathManifest
    from clinical_knowledge.lazy_rag_config import manifest_path

    if _path_manifest is None:
        mp = manifest_path()
        _path_manifest = CorpusPathManifest.load(mp) if mp.is_file() else CorpusPathManifest()
    _lazy_chunk_store = LazyChunkStore.from_env(
        protocols_by_path=_protocols_by_path,
        protocol_meta=_protocol_meta,
    )
    if _lazy_chunk_store is None and _path_manifest.entries:
        from clinical_knowledge.lazy_rag_config import chunks_data_root

        corpus = chunks_data_root()
        parts_dir = corpus / "corpus_chunks_parts"
        corpus_dir = parts_dir if parts_dir.is_dir() else corpus
        if corpus_dir.is_dir():
            _lazy_chunk_store = LazyChunkStore(
                manifest=_path_manifest,
                corpus_dir=corpus_dir,
                protocols_by_path=_protocols_by_path,
                protocol_meta=_protocol_meta,
            )
    return _lazy_chunk_store


def _ensure_path_lex_index():
    global _path_lex_index
    if _path_lex_index is not None:
        return _path_lex_index
    from clinical_knowledge.path_lex_index import PathLexIndex

    _path_lex_index = PathLexIndex.from_env()
    return _path_lex_index


def load_data_manifest() -> None:
    """Старт без полного корпуса в RAM: manifest + lazy chunk store."""
    global _chunks, _chunks_by_path, _bm25_index, _lex_inverted_index, _path_manifest, _lazy_chunk_store
    global _chunk_global_indices_by_path
    _load_metadata_common()
    _chunks = []
    _chunks_by_path = {}
    _chunk_global_indices_by_path = {}
    _bm25_index = None
    _lex_inverted_index = None
    from clinical_knowledge.corpus_path_manifest import CorpusPathManifest
    from clinical_knowledge.lazy_rag_config import manifest_path

    mp = manifest_path()
    _path_manifest = CorpusPathManifest.load(mp) if mp.is_file() else CorpusPathManifest()
    _lazy_chunk_store = _ensure_lazy_chunk_store()
    try:
        from clinical_knowledge.vector_index import load_index_from_env

        load_index_from_env(None)
    except Exception:
        pass
    gc.collect()


def load_data_full() -> None:
    global _chunks, _chunks_by_path, _protocols_by_path, _protocol_meta, _structured_by_path, _routing
    _load_metadata_common()

    if _use_jsonl_chunks():
        parts = _jsonl_chunk_files()
        if not parts:
            raise SystemExit(
                f"Нет JSONL-чанков ({CORPUS_CHUNKS_PARTS_GLOB} или RAG_CHUNKS_JSONL) "
                f"в { _chunks_data_root() }. Соберите корпус, задайте RAG_CHUNKS_DIR на диск с данными "
                "или RAG_CHUNKS_SOURCE=json при наличии chunks.json"
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
    summary_rows = _load_summary_rag_chunks()
    if summary_rows:
        _chunks.extend(summary_rows)
    _chunks_by_path = {}
    global _chunk_global_indices_by_path
    _chunk_global_indices_by_path = {}
    for i, ch in enumerate(_chunks):
        p = str(ch.get("path") or "").replace("\\", "/")
        if not p:
            continue
        _chunks_by_path.setdefault(p, []).append(ch)
        _chunk_global_indices_by_path.setdefault(p, []).append(i)
    for plist in _chunks_by_path.values():
        plist.sort(key=lambda x: int(x.get("chunk_index", 0)))
    gc.collect()

    global _bm25_index
    _bm25_alpha_chk = float(os.environ.get("RAG_LEX_BM25_ALPHA", "0.55"))
    _pool_merge_chk = os.environ.get("RAG_EMBED_POOL_MERGE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if _bm25_alpha_chk < 0.999 or _pool_merge_chk:
        _bm25_index = build_bm25_index(_chunks, tokenize_ru)
    else:
        _bm25_index = None

    global _lex_inverted_index
    if env_bool("RAG_LEX_INDEX_DEFER", env_bool("RENDER", False)):
        _lex_inverted_index = None
    else:
        _lex_inverted_index = _build_lex_inverted_index(_chunks)
        gc.collect()

    try:
        from clinical_knowledge.vector_index import load_index_from_env

        load_index_from_env(_chunks)
    except Exception:
        pass


def load_data() -> None:
    from clinical_knowledge.lazy_rag_config import startup_mode

    if startup_mode() == "manifest":
        load_data_manifest()
    else:
        load_data_full()


def _run_load_data_background() -> None:
    """Тяжёлый корпус грузится в фоне - uvicorn успевает открыть порт (Render health check)."""
    global _chunks_load_error
    try:
        load_data()
        if env_bool("CONSULT_RULES_SUMMARY_FALLBACK", True) and env_bool(
            "CONSULT_PREWARM_SUMMARY_ICD_INDEX", True
        ):
            try:
                from clinical_knowledge.protocol_summary.icd_index import prewarm_icd_summary_index

                prewarm_icd_summary_index()
            except Exception:
                pass
        if env_bool("CONSULT_PREWARM_PROTOCOL_SUMMARIES", True):
            try:
                from clinical_knowledge.protocol_summary.loader import prewarm_protocol_summaries

                n = prewarm_protocol_summaries()
                _log.info("Prewarmed %d protocol summaries", n)
            except Exception:
                pass
        if env_bool("CONSULT_PREWARM_PROTOCOL_ICD_INDEX", True):
            try:
                from clinical_knowledge.protocol_icd_index import prewarm_protocol_icd_index

                prewarm_protocol_icd_index()
            except Exception:
                pass
        _chunks_load_error = None
    except SystemExit as e:
        code = e.code
        if isinstance(code, str):
            _chunks_load_error = code
        elif isinstance(code, int):
            _chunks_load_error = f"Ошибка запуска (код {code})"
        else:
            _chunks_load_error = repr(code)
    except Exception as e:
        _chunks_load_error = str(e)
    finally:
        _chunks_load_done.set()


def _require_rag_loaded(*, wait: bool = True, max_wait_sec: float | None = None) -> None:
    """Дождаться фоновой загрузки корпуса (Render health check vs тяжёлый JSONL)."""
    if wait and not _chunks_load_done.is_set():
        if max_wait_sec is not None:
            timeout = max(3.0, float(max_wait_sec))
        else:
            timeout = max(5.0, env_float("RAG_LOAD_WAIT_SEC", 180.0))
        _chunks_load_done.wait(timeout=timeout)
    if not _chunks_load_done.is_set():
        raise HTTPException(
            status_code=503,
            detail="Индекс протоколов загружается. Повторите запрос через минуту.",
        )
    if _chunks_load_error is not None:
        raise HTTPException(
            status_code=503,
            detail=f"Не удалось загрузить корпус: {_chunks_load_error}",
        )


def tokenize_ru(s: str) -> list[str]:
    s = s.lower().replace("ё", "е")
    return [t for t in re.findall(r"[а-яa-z]{2,}", s) if len(t) >= 2]


def _ensure_lex_inverted_index() -> dict[str, frozenset[int]]:
    """Ленивое построение инвертированного индекса (меньше пик RAM при старте на Render)."""
    global _lex_inverted_index
    if _lex_inverted_index is not None:
        return _lex_inverted_index
    with _lex_index_lock:
        if _lex_inverted_index is not None:
            return _lex_inverted_index
        _lex_inverted_index = _build_lex_inverted_index(_chunks)
        import gc as _gc

        _gc.collect()
        return _lex_inverted_index


def _build_lex_inverted_index(chunks: list[dict]) -> dict[str, frozenset[int]]:
    """Токен → индексы чанков (фаза 6: ускорение retrieve без полного прохода)."""
    min_tok = max(2, env_int("RAG_MIN_INDEX_TOKEN_LEN", 2))
    raw: dict[str, set[int]] = {}
    for i, ch in enumerate(chunks):
        lex_src = (ch.get("_lex_search") or ch.get("lex_text") or ch.get("text") or "") + " " + (
            ch.get("title") or ""
        )
        if not lex_src.strip():
            continue
        tokens = set(tokenize_ru(lex_src))
        for code in extract_icd_codes_raw(lex_src):
            tokens.update(icd_tokens_for_lex([code]))
        for t in tokens:
            if len(t) < min_tok:
                continue
            raw.setdefault(t, set()).add(i)
    return {k: frozenset(v) for k, v in raw.items()}


def _chunk_indices_for_path_allowlist(path_set: frozenset[str]) -> set[int]:
    """Индексы чанков только из allowlist - для consult-review, не весь корпус 65k."""
    if not path_set:
        return set()
    out: set[int] = set()
    for p in path_set:
        norm = p.replace("\\", "/").strip()
        if not norm:
            continue
        for idx in _chunk_global_indices_by_path.get(norm, ()):
            out.add(idx)
        if out:
            continue
        for cp, indices in _chunk_global_indices_by_path.items():
            if cp.endswith(norm.split("/")[-1]) or norm.endswith(cp.split("/")[-1]):
                out.update(indices)
    if not out and _chunks:
        for idx, ch in enumerate(_chunks):
            cp = str(ch.get("path") or ch.get("source_path") or "").replace("\\", "/").strip()
            if not cp:
                continue
            if cp in path_set or cp.endswith(norm.split("/")[-1]) or norm.endswith(cp.split("/")[-1]):
                out.add(idx)
    return out


def _cap_lex_candidate_indices(
    candidate_indices: set[int] | None,
    *,
    path_allowlist_set: frozenset[str],
    qtok: set[str] | None = None,
) -> set[int] | None:
    """Ограничить пул кандидатов retrieve (OOM при union 30k+ чанков на consult)."""
    if path_allowlist_set:
        path_only = _chunk_indices_for_path_allowlist(path_allowlist_set)
        if path_only:
            return path_only
    max_cand = env_int("RAG_LEX_MAX_CANDIDATES", 4000 if env_bool("RENDER", False) else 0)
    if max_cand <= 0 or not candidate_indices or len(candidate_indices) <= max_cand:
        return candidate_indices
    if qtok and candidate_indices:
        lex_idx = _lex_inverted_index
        if lex_idx is None:
            try:
                lex_idx = _ensure_lex_inverted_index()
            except Exception:
                lex_idx = None
        if lex_idx:
            rare_first = sorted(
                [t for t in qtok if t in lex_idx and t not in RAG_GENERIC_LEX],
                key=lambda t: len(lex_idx[t]),
            )
            narrowed: set[int] = set()
            for t in rare_first[:12]:
                for idx in lex_idx.get(t, ()):
                    if idx in candidate_indices:
                        narrowed.add(idx)
                        if len(narrowed) >= max_cand:
                            return narrowed
            if len(narrowed) >= max(32, max_cand // 8):
                candidate_indices = narrowed
                if len(candidate_indices) <= max_cand:
                    return candidate_indices
    trimmed: set[int] = set()
    for idx in candidate_indices:
        trimmed.add(idx)
        if len(trimmed) >= max_cand:
            break
    return trimmed


# Слабые модификаторы без смысла диагноза: совпадение только по ним не должно тянуть чужие протоколы.
RAG_GENERIC_LEX: frozenset[str] = frozenset(
    {
        "правосторонний",
        "левосторонний",
        "двусторонний",
        "односторонний",
        "верхний",
        "нижний",
        "передний",
        "задний",
        "средний",
        "острый",
        "хронический",
        "пациент",
        "пациентка",
        "лет",
        "года",
        "году",
        "женский",
        "мужской",
        "возраст",
        "жалуется",
        "жалобы",
        "жалоб",
        "жалоба",
        "предоставлен",
        "предоставленные",
        "отмечает",
        "считает",
        "наличие",
        "дней",
        "недель",
        "месяц",
        "месяцев",
        "год",
    }
)


def _extra_clinical_tokens(q_raw: str) -> set[str]:
    """Доп. токены по подстрокам запроса (ЛОР, вестибулярная тема) - лучше пересечение с корпусом."""
    rq = _norm_query(q_raw)
    extra: set[str] = set()
    if any(
        x in rq
        for x in (
            "гаймор",
            "синусит",
            "пазух",
            "этмоид",
            "лор",
            "носоглот",
            "аденоид",
            "тонзилл",
            "ангин",
            "фаринг",
            "ларинг",
        )
    ):
        extra.update(
            {
                "гаймор",
                "синусит",
                "пазух",
                "придаточн",
                "носоглот",
                "этмоид",
            }
        )
    if any(x in rq for x in ("вертиго", "дппг", "вестибуляр", "нистагм", "дикс", "холлпайк")):
        extra.update({"вестибуляр", "вертиго", "дппг", "нистагм", "позицион"})
    return extra


def _anchor_tokens(qtok: set[str]) -> list[str]:
    """Токены-якоря: не из общего списка модификаторов."""
    a = [t for t in qtok if t not in RAG_GENERIC_LEX]
    return a


def _cosine_vec(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(s / (na * nb))


def _chunk_text_for_embedding(ch: dict) -> str:
    t = (
        (ch.get("embedding_ready_text") or "").strip()
        or (ch.get("lex_text") or "").strip()
        or (ch.get("text") or "").strip()
    )
    if len(t) > 7500:
        t = t[:7500] + "…"
    if not t:
        t = ((ch.get("title") or "") + " " + (ch.get("path") or "")).strip() or "."
    return t


def _norm_minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.5] * len(values)
    return [(float(x) - lo) / (hi - lo) for x in values]


def _gemini_embed_one(
    model: str,
    text: str,
    task_type: str | None,
) -> list[float]:
    genai = _legacy_genai_module()

    embed_fn = getattr(genai, "embed_content", None)
    if embed_fn is None:
        from google.generativeai.embedding import embed_content as embed_fn

    kw: dict = {"model": model, "content": text[:8000]}
    if task_type:
        kw["task_type"] = task_type
    try:
        r = embed_fn(**kw)
    except (TypeError, ValueError, KeyError):
        kw.pop("task_type", None)
        r = embed_fn(**kw)
    emb = r.get("embedding")
    if isinstance(emb, dict) and "values" in emb:
        emb = emb["values"]
    if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
        return [float(x) for x in emb]
    raise RuntimeError("unexpected embedding response")


def _chunk_has_precomputed_embedding(ch: dict) -> bool:
    emb = ch.get("embedding")
    return isinstance(emb, list) and len(emb) >= 8 and isinstance(emb[0], (int, float))


def _precomputed_chunk_embed_rerank_pool(
    query: str,
    pool_rows: list[tuple],
    alpha: float,
    emb_model: str,
) -> list[tuple[float, float, float, dict]] | None:
    """Rerank по embedding в JSONL чанка (один вызов API - только query). RAG_PRECOMPUTED_CHUNK_EMBED=1."""
    if os.environ.get("RAG_PRECOMPUTED_CHUNK_EMBED", "0").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return None
    if not pool_rows:
        return None
    doc_chunks: list[dict] = []
    for row in pool_rows:
        ch = row[4] if len(row) >= 5 else row[3]
        if not _chunk_has_precomputed_embedding(ch):
            return None
        doc_chunks.append(ch)
    q_vec = _gemini_embed_one(emb_model, (query or "").strip()[:8000], "retrieval_query")
    finals = [float(r[0]) for r in pool_rows]
    lex_norm = _norm_minmax(finals)
    out_rows: list[tuple[float, float, float, dict]] = []
    for i, row in enumerate(pool_rows):
        if len(row) >= 5:
            final, lex, mult, ch = row[0], row[1], row[3], row[4]
        else:
            final, lex, mult, ch = row[0], row[1], row[2], row[3]
        doc_vec = [float(x) for x in ch["embedding"]]
        cos = _cosine_vec(q_vec, doc_vec)
        h = alpha * lex_norm[i] + (1.0 - alpha) * cos
        out_rows.append((h, lex, mult, ch))
    out_rows.sort(
        key=lambda x: (
            -x[0],
            str(x[3].get("path", "")),
            str(x[3].get("chunk_index", "")),
        )
    )
    return out_rows


def _gemini_embed_rerank_pool(
    query: str,
    pool_rows: list[tuple[float, float, float, dict]],
    alpha: float,
    model: str,
) -> list[tuple[float, float, float, dict]]:
    """Переранжирование пула чанков: α·lex_norm + (1−α)·cosine(query, chunk)."""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key or not pool_rows:
        return pool_rows
    genai = _legacy_genai_module()

    genai.configure(api_key=key)

    q_text = (query or "").strip()[:8000]
    q_vec = _gemini_embed_one(
        model,
        q_text,
        "retrieval_query",
    )

    finals = [float(r[0]) for r in pool_rows]
    lex_norm = _norm_minmax(finals)

    doc_texts = [
        _chunk_text_for_embedding(r[4] if len(r) >= 5 else r[3]) for r in pool_rows
    ]
    max_workers = min(8, max(1, len(doc_texts)))

    def embed_doc(i: int) -> list[float]:
        return _gemini_embed_one(
            model,
            doc_texts[i],
            "retrieval_document",
        )

    timeout = float(os.environ.get("GEMINI_EMBED_CALL_TIMEOUT", "45"))
    doc_vecs: list[list[float]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(embed_doc, i) for i in range(len(pool_rows))]
        for fut in futures:
            doc_vecs.append(fut.result(timeout=timeout))

    out_rows: list[tuple[float, float, float, dict]] = []
    for i, row in enumerate(pool_rows):
        if len(row) >= 5:
            # (final, lex_raw, bm25_raw, routing_mult, ch)
            final, lex, mult, ch = row[0], row[1], row[3], row[4]
        else:
            final, lex, mult, ch = row
        cos = _cosine_vec(q_vec, doc_vecs[i])
        h = alpha * lex_norm[i] + (1.0 - alpha) * cos
        out_rows.append((h, lex, mult, ch))
    # Детерминированный порядок при равных score: стабильный tie-break по пути и индексу чанка.
    out_rows.sort(
        key=lambda x: (
            -x[0],
            str(x[3].get("path", "")),
            str(x[3].get("chunk_index", "")),
        )
    )
    return out_rows


def _icd_embed_rank_candidates(
    rag_query: str,
    pool: list[dict],
    k: int,
    emb_model: str,
) -> list[dict]:
    """k ближайших по косинусу (эмбеддинг запроса vs «код + название ru»)."""
    if k <= 0 or not pool:
        return []
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        return []
    rq = (rag_query or "").strip()[:8000]
    try:
        q_vec = _gemini_embed_one(emb_model, rq, "retrieval_query")
    except Exception:
        return []
    doc_texts: list[str] = []
    for p in pool:
        t = f"{p.get('code') or ''} {p.get('title_ru') or ''}".strip()[:4000]
        doc_texts.append(t if t else str(p.get("code") or "."))
    max_workers = min(8, max(1, len(doc_texts)))
    timeout = float(os.environ.get("GEMINI_EMBED_CALL_TIMEOUT", "45"))

    def embed_doc(i: int) -> list[float]:
        return _gemini_embed_one(emb_model, doc_texts[i], "retrieval_document")

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(embed_doc, i) for i in range(len(pool))]
            doc_vecs = [f.result(timeout=timeout) for f in futures]
    except Exception:
        return []
    scored_pairs: list[tuple[float, dict]] = []
    for i, row in enumerate(pool):
        cos = _cosine_vec(q_vec, doc_vecs[i])
        copy = dict(row)
        copy["embed_sim"] = round(float(cos), 4)
        scored_pairs.append((float(cos), copy))
    scored_pairs.sort(key=lambda x: -x[0])
    return [d for _, d in scored_pairs[:k]]


def _merge_icd_allowed_for_gemini(
    lex_top: list[dict], embed_top: list[dict]
) -> list[dict]:
    """Объединение лексического топ-N и k-NN по эмбеддингу; один код - одна строка."""
    by_code: dict[str, dict] = {}
    for it in lex_top:
        c = normalize_icd_code(str(it.get("code") or ""))
        if not c:
            continue
        row = dict(it)
        row["code"] = c
        row["pool_source"] = "lex_top"
        by_code[c] = row
    for it in embed_top:
        c = normalize_icd_code(str(it.get("code") or ""))
        if not c:
            continue
        row = dict(it)
        row["code"] = c
        if c not in by_code:
            row["pool_source"] = "embed_knn"
            by_code[c] = row
        else:
            prev = by_code[c]
            if it.get("embed_sim") is not None:
                prev["embed_sim"] = it.get("embed_sim")
            prev["pool_source"] = "lex_top+embed"
    return list(by_code.values())


def _refine_icd_analysis_with_gemini(
    rag_query: str,
    icd_analysis: dict,
    model,
    *,
    lexicon_query: str | None = None,
) -> None:
    """Уточнение suggested и codes_for_retrieval: Gemini выбирает только из лексического топа (без k-NN по умолчанию)."""
    if icd_analysis.get("explicit_icd_in_query"):
        return
    if os.environ.get("RAG_ICD_GEMINI_SELECT", "1").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    if not (rag_query or "").strip():
        return
    lq = (lexicon_query or rag_query).strip()
    scored = merge_clinical_icd_hints(ru_lexicon_scored_entries(lq), lq)
    scored = filter_icd_pool_for_complaint(scored, lq)
    if not scored:
        return
    n_lex = max(1, min(int(os.environ.get("RAG_ICD_LEX_TOP", "12")), 40))
    n_pool = max(n_lex, min(int(os.environ.get("RAG_ICD_EMBED_POOL", "32")), 120))
    k_embed = max(0, min(int(os.environ.get("RAG_ICD_EMBED_K", "0")), 20))

    lex_top = scored[:n_lex]
    pool = scored[:n_pool]
    emb_model = os.environ.get(
        "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2-preview"
    ).strip()
    embed_top: list[dict] = []
    if k_embed > 0 and os.environ.get("RAG_ICD_EMBED_RANK", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        embed_top = _icd_embed_rank_candidates(
            rag_query, pool, k_embed, emb_model
        )
    allowed_list = _merge_icd_allowed_for_gemini(lex_top, embed_top)
    present = {normalize_icd_code(str(x.get("code") or "")) for x in allowed_list}
    present.discard("")
    for d in icd_analysis.get("detected") or []:
        if not isinstance(d, dict):
            continue
        dc = normalize_icd_code(str(d.get("code") or ""))
        if not dc or dc in present:
            continue
        found = next((x for x in scored if x["code"] == dc), None)
        if found is None:
            info = describe_code(dc)
            found = {
                "code": dc,
                "title_ru": info.get("title_ru"),
                "title_en": info.get("title_en"),
                "lex_score": None,
            }
        row = dict(found)
        row["pool_source"] = "regex_query"
        allowed_list.append(row)
        present.add(dc)

    allowed_by_code = {
        normalize_icd_code(str(x.get("code") or "")): x for x in allowed_list
    }
    allowed_by_code.pop("", None)

    payload = json.dumps(
        [
            {
                "code": x["code"],
                "title_ru": x.get("title_ru") or "",
                "title_en": x.get("title_en") or "",
            }
            for x in sorted(allowed_list, key=lambda z: str(z.get("code") or ""))
        ],
        ensure_ascii=False,
    )
    prompt = (
        SYSTEM_ICD_POOL_SELECT
        + "\n\n---\n\nЗапрос пользователя:\n"
        + rag_query.strip()[:4000]
        + "\n\nallowed (единственный источник кодов):\n"
        + payload[:14000]
    )
    parsed = None
    try:
        resp = generate_gemini(model, prompt)
        txt = _extract_gemini_text(resp)
        parsed = _try_parse_json(txt)
    except HTTPException:
        return
    except Exception:
        return
    if not parsed or not isinstance(parsed, dict):
        return
    codes = parsed.get("codes")
    if not isinstance(codes, list):
        return
    selected: list[dict] = []
    for item in codes[:6]:
        if not isinstance(item, dict):
            continue
        raw = normalize_icd_code(str(item.get("code") or ""))
        if not raw or raw not in allowed_by_code:
            continue
        src = allowed_by_code[raw]
        selected.append(
            {
                "code": raw,
                "title_ru": src.get("title_ru"),
                "title_en": src.get("title_en"),
                "match_method": "gemini_from_pool",
                "score": src.get("lex_score"),
                "rationale": (item.get("rationale") or item.get("note") or "")[:320],
            }
        )
    if not selected:
        return
    icd_analysis["suggested"] = selected
    icd_analysis["icd_meta"] = {
        "strategy": (
            "gemini_from_lex_top_and_embed_knn"
            if embed_top
            else "gemini_from_lex_top"
        ),
        "lex_top": n_lex,
        "embed_pool": n_pool,
        "embed_k": k_embed,
        "embedding_used": bool(embed_top),
        "allowed_count": len(allowed_list),
    }
    merged_codes = list(
        dict.fromkeys(
            [normalize_icd_code(str(d.get("code") or "")) for d in icd_analysis.get("detected") or []]
            + [s["code"] for s in selected]
        )
    )
    merged_codes = [c for c in merged_codes if c][:10]
    icd_analysis["codes_for_retrieval"] = merged_codes
    from icd_mkb import finalize_icd_analysis_codes

    finalize_icd_analysis_codes(icd_analysis, rag_query)


def _top_retrieval_score_for_icd_gate(retrieved: list[dict]) -> tuple[float, bool]:
    """После гибридного эмбеддинг-переранжирования чанков поле score ∈ [0,1]."""
    if not retrieved:
        return 0.0, False
    r0 = retrieved[0]
    if not r0.get("embedding_rerank"):
        return 0.0, False
    try:
        sc = float(r0.get("score") or 0)
    except (TypeError, ValueError):
        return 0.0, False
    return max(0.0, min(1.0, sc)), True


def maybe_refine_icd_with_gemini_after_retrieve(
    model,
    rag_query: str,
    icd_analysis: dict,
    retrieved: list[dict],
) -> None:
    """Gemini выбирает коды из пула только при уверенном отборе протоколов (≥ порога) и без явных кодов в тексте."""
    if icd_analysis.get("explicit_icd_in_query"):
        return
    if os.environ.get("RAG_ICD_GEMINI_SELECT", "1").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    if not (rag_query or "").strip():
        return
    require_embed = os.environ.get("RAG_ICD_GEMINI_REQUIRE_EMBED_RANK", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    top, emb_ok = _top_retrieval_score_for_icd_gate(retrieved)
    if require_embed and not emb_ok:
        return
    min_sc = float(os.environ.get("RAG_ICD_GEMINI_MIN_TOP_SCORE", "0.8"))
    if top < min_sc:
        return
    _refine_icd_analysis_with_gemini(rag_query, icd_analysis, model)


def _norm_query(s: str) -> str:
    return (s or "").lower().replace("ё", "е")


def infer_audience_from_query(q: str, routing: dict) -> str | None:
    """'adult' | 'child' | None - по словам и числам (49 лет, ребёнок …)."""
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


def _norm_audience_blob(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("_", " ").replace("-", " ")).strip()


def doc_audience_hint(path: str, title: str, routing: dict) -> str | None:
    """pediatric | adult | mixed | None - по index.csv, названию файла/заголовка."""
    from clinical_knowledge.protocol_audience import infer_protocol_audience

    row = _load_index_csv_by_path().get((path or "").strip(), {})
    aud_csv = (row.get("audience") or "").strip().lower()
    if aud_csv in ("pediatric", "child"):
        return "pediatric"
    if aud_csv == "adult":
        return "adult"
    if aud_csv == "mixed":
        return "mixed"
    hint = infer_protocol_audience(path, title)
    if hint:
        return hint
    s = _norm_audience_blob(f"{path} {title}")
    ped = [_norm_audience_blob(p) for p in routing.get("pediatric_title_markers") or []]
    adult_t = [_norm_audience_blob(a) for a in routing.get("adult_title_markers") or []]
    has_p = any(p in s for p in ped if p)
    has_a = any(a in s for a in adult_t if a)
    if has_p and has_a:
        return "mixed"
    if has_p:
        return "pediatric"
    if has_a:
        return "adult"
    return None


def infer_audience_from_funnel_context(q: str) -> str | None:
    """Аудитория из строк «Контекст подбора: …» (шаг 1 воронки)."""
    nq = _norm_query(q)
    if "контекст подбора: детское" in nq or "детское население" in nq:
        return "child"
    if "контекст подбора: взрослое" in nq or "взрослое население" in nq:
        return "adult"
    if "контекст подбора: беремен" in nq or "беременные" in nq:
        return "pregnant"
    if "контекст подбора: неотлож" in nq or "неотложная помощь" in nq:
        if any(
            x in nq
            for x in ("ребен", "ребён", "детск", "новорожд", "грудн", "младен", "педиатр")
        ):
            return "child"
        return "adult"
    return None


def _chunk_audience_mismatch(aud: str, path: str, title: str, routing: dict) -> bool:
    hint = doc_audience_hint(path, title, routing)
    if hint is None or hint == "mixed":
        return False
    if aud == "adult" and hint == "pediatric":
        return True
    if aud == "child" and hint == "adult":
        return True
    return False


def filter_retrieval_by_audience(
    rows: list[dict], rq: str, routing: dict
) -> tuple[list[dict], str | None, bool]:
    """Отбрасывает чанки с явно несовпадающей аудиторией (дет/взросл)."""
    aud = infer_audience_from_query(rq, routing)
    if aud is None:
        aud = infer_audience_from_funnel_context(rq)
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
        # Никогда не возвращаем чанки явно несовпадающей аудитории (дет/взросл).
        return [], aud, True
    return out, aud, False


def _age_full_years(birth: date, ref: date) -> int:
    a = ref.year - birth.year
    if (ref.month, ref.day) < (birth.month, birth.day):
        a -= 1
    return a


def _consult_extract_date_of_birth(raw: str) -> tuple[date | None, dict]:
    """Ищет дату рождения в форме ДД.ММ.ГГГГ - приоритет строке с Ф.И.О. или маркерами ДР."""
    blob = normalize_text_for_icd_scan(raw or "").replace("\u00a0", " ").strip()
    meta_trace: dict[str, object] = {}
    if not blob:
        return None, meta_trace
    dob_re = re.compile(r"\b(\d{2})\.(\d{2})\.(19\d{2}|20\d{2})\b")
    fio_here = re.compile(
        r"ф\s*\.\s*и\s*\.\s*о\s*", re.I
    )

    cand: list[tuple[int, int, date, str]] = []
    for m in dob_re.finditer(blob):
        try:
            dd = int(m.group(1))
            mo = int(m.group(2))
            yr = int(m.group(3))
            birth_d = date(yr, mo, dd)
        except ValueError:
            continue
        if not (1870 <= birth_d.year <= date.today().year + 1):
            continue

        ls = blob.rfind("\n", 0, m.start())
        le = blob.find("\n", m.end())
        if ls < 0:
            ls = 0
        else:
            ls += 1
        if le < 0:
            le = len(blob)
        line = blob[ls:le]

        prio = 0
        ll = line.lower()
        if fio_here.search(line):
            prio += 6
        if (
            ("дата" in ll and "рожд" in ll)
            or "д р" in ll
            or " д.р" in ll
            or "д/р" in ll
            or ll.strip().startswith("др ")
        ):
            prio += 4
        if "пациент" in ll[:40]:
            prio += 1

        cand.append((prio, -m.start(), birth_d, line.strip()[:220]))

    if not cand:
        return None, meta_trace
    cand.sort(key=lambda x: (-x[0], x[1]))
    best = cand[0]
    meta_trace.update(
        {
            "dob_ddmmyyyy": best[2].strftime("%d.%m.%Y"),
            "confidence_prio": best[0],
            "snippet": best[3],
            "candidate_dates": len(cand),
        }
    )
    return best[2], meta_trace


def consult_demographics_banner_from_kz(full_text_raw: str) -> tuple[str, dict]:
    """Одна-две строки для добавления к routing/full query при известной ДР из КЗ."""
    ref = date.today()
    dob, trace = _consult_extract_date_of_birth(full_text_raw)
    meta: dict[str, object] = {"date_of_birth": None, "age_years": None, "audience": None}
    meta.update(trace)
    if dob is None:
        return "", meta

    yrs = _age_full_years(dob, ref)
    meta["date_of_birth"] = dob.isoformat()
    meta["age_years"] = yrs
    if yrs >= 18:
        meta["audience"] = "adult"
        band = (
            f"Из текста документа (авто): дата рождения пациента {dob.strftime('%d.%m.%Y')}; "
            f"на дату обработки {ref.strftime('%d.%m.%Y')} - {yrs} полных лет; пациент взрослый (≥18 лет)."
        )
    elif yrs >= 0:
        meta["audience"] = "child"
        band = (
            f"Из текста документа (авто): дата рождения пациента {dob.strftime('%d.%m.%Y')}; "
            f"на дату обработки {ref.strftime('%d.%m.%Y')} - {yrs} полных лет; пациент ребёнок (<18 лет)."
        )
    else:
        return "", meta

    return band, meta


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
    # Блок ответов содержит слова вопросов (напр. «кровотечение») - подстрока «кров» ложно тянет гематологию и размывает отбор.
    mark = " - Ответы на уточняющие вопросы:"
    if mark in part:
        part = part.split(mark, 1)[0].strip()
    part = strip_funnel_context_lines(part)
    return part if part else full_query.strip()


def expand_query_for_retrieve(q_rag: str) -> tuple[str, dict | None]:
    """Детерминированное клиническое расширение текста перед retrieve (без LLM)."""
    try:
        from clinical_knowledge.search_query_expand import expand_clinical_query_terms

        expanded, meta = expand_clinical_query_terms(q_rag or "")
        if meta.get("applied"):
            return expanded, meta
        return q_rag, None
    except Exception:
        return q_rag, None


def gather_protocol_text(path: str, max_chars: int) -> str:
    """Склеивает чанки одного PDF по порядку (до max_chars), приоритет клинических типов."""
    parts_raw = _chunks_by_path.get(path) or []
    if not parts_raw:
        return ""

    _type_rank = {
        "diagnostics": 0,
        "criteria_block": 1,
        "treatment": 2,
        "pharmacotherapy": 3,
        "drug_list": 4,
        "dispensary": 5,
        "prevention": 6,
        "algorithm": 7,
        "table": 8,
        "protocol_overview": 9,
        "body": 10,
    }

    def _rank(ch: dict) -> tuple[int, int]:
        ct = (ch.get("chunk_type") or ch.get("kind") or "body").strip().lower()
        pg = int(ch.get("page_from") or ch.get("page") or 0)
        return (_type_rank.get(ct, 10), pg)

    parts = sorted(parts_raw, key=_rank)
    out: list[str] = []
    n = 0
    for ch in parts:
        t = (ch.get("text") or "").strip()
        if not t:
            continue
        if n + len(t) > max_chars:
            rest = max_chars - n
            if rest > 80:
                try:
                    from clinical_knowledge.meaningful_excerpt import meaningful_excerpt

                    tail = meaningful_excerpt(t, limit=rest)
                except Exception:
                    tail = ""
                out.append(tail or t[:rest])
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


def _confidence_numeric(score: object) -> float | None:
    try:
        x = float(score)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, x))


def confidence_for_detailed_extraction(score: object) -> bool:
    """Развёрнутая выдержка (SYSTEM_EXTRACT_FULL) при оценке ≥80%, не только при 100%."""
    x = _confidence_numeric(score)
    if x is None:
        return False
    min_s = float(os.environ.get("RAG_DETAIL_EXTRACT_MIN_SCORE", "0.8"))
    return x >= min_s


def _protocol_catalog_icd_boost(path: str, icd_norms: list[str]) -> float:
    """Усиление по рейтингу МКБ из protocol_catalog (icd10_weights, %)."""
    if not icd_norms:
        return 1.0
    try:
        from clinical_knowledge.protocol_catalog import load_protocol_catalog

        row = load_protocol_catalog().get((path or "").replace("\\", "/"))
    except Exception:
        return 1.0
    if not row:
        return 1.0
    weights = row.get("icd10_weights") or {}
    primary = {
        normalize_icd_code(str(x)).lower()
        for x in (row.get("icd10_primary") or [])
        if normalize_icd_code(str(x))
    }
    best_pct = 0
    for raw in icd_norms:
        c = normalize_icd_code(str(raw).upper())
        if not c:
            continue
        w = weights.get(c)
        if w is not None:
            try:
                best_pct = max(best_pct, int(round(float(w))))
            except (TypeError, ValueError):
                pass
        elif c.lower() in primary:
            best_pct = max(best_pct, 88)
    if best_pct <= 0:
        return 1.0
    scale = float(os.environ.get("RAG_ICD_CATALOG_WEIGHT_SCALE", "0.95"))
    return min(2.25, 1.0 + (best_pct / 100.0) * scale)


def _protocol_meta_icd_boost(path: str, icd_norms: list[str]) -> float:
    """Усиление, если в protocol_meta заданы icd_codes/mkb_codes и они пересекаются с запросом."""
    if not icd_norms:
        return 1.0
    pm = _protocol_meta.get(path) or {}
    raw = pm.get("icd_codes") or pm.get("mkb_codes") or []
    if not isinstance(raw, list):
        return 1.0
    boost = float(os.environ.get("RAG_ICD_META_BOOST", "1.35"))
    want = {x.strip().lower() for x in icd_norms if isinstance(x, str)}
    for c in raw:
        if not isinstance(c, str):
            continue
        n = normalize_icd_code(c).strip().lower()
        if n and n in want:
            return boost
    return 1.0


def _rag_support_map(retrieved: list[dict]) -> tuple[dict[str, float], float]:
    """path → нормализованный score / max в батче; второе значение - max score."""
    max_s = 0.0
    for r in retrieved:
        try:
            s = float(r.get("score") or 0)
        except (TypeError, ValueError):
            s = 0.0
        if s > max_s:
            max_s = s
    if max_s <= 0:
        max_s = 1.0
    m: dict[str, float] = {}
    for r in retrieved:
        p = str(r.get("path") or "")
        if not p:
            continue
        try:
            s = float(r.get("score") or 0)
        except (TypeError, ValueError):
            s = 0.0
        n = s / max_s
        if p not in m or n > m[p]:
            m[p] = n
    return m, max_s


def apply_protocol_confidence_calibration(
    parsed: dict | None, retrieved: list[dict]
) -> dict[str, float]:
    """Смешивает оценку модели с опорой отбора (rag_support); правит confidence_score."""
    rag_map, _ = _rag_support_map(retrieved)
    if not parsed or not isinstance(parsed, dict):
        return rag_map
    w = float(os.environ.get("RAG_LLM_CONF_BLEND_W", "0.62"))
    cap_low = float(os.environ.get("RAG_MIN_RAG_SUPPORT_CAP", "0.74"))
    min_rag_high = float(os.environ.get("RAG_MIN_RAG_SUPPORT_FOR_HIGH_CONF", "0.2"))
    protos = parsed.get("protocols")
    if not isinstance(protos, list):
        return rag_map
    for pr in protos:
        if not isinstance(pr, dict):
            continue
        p = str(pr.get("path") or "")
        rag_sup = float(rag_map.get(p, 0.0))
        pr["rag_support"] = round(rag_sup, 4)
        llm_c = _confidence_numeric(pr.get("confidence_score"))
        if llm_c is None:
            llm_c = 0.55
        pr["confidence_score_llm"] = pr.get("confidence_score")
        blended = w * llm_c + (1.0 - w) * rag_sup
        if rag_sup < min_rag_high:
            blended = min(blended, cap_low)
            pr["low_retrieval_support"] = True
        pr["confidence_score"] = round(max(0.0, min(1.0, blended)), 4)
    return rag_map


def _majority_category_from_retrieval(retrieved: list[dict]) -> str | None:
    """Рубрика (slug), чаще всего встречающаяся среди отобранных чанков."""
    cats: list[str] = []
    for r in retrieved:
        c = (r.get("category") or "").strip()
        if c:
            cats.append(c)
            continue
        p = r.get("path") or ""
        pr = _protocols_by_path.get(p) or {}
        pm = _protocol_meta.get(p) or {}
        x = (pr.get("category") or pm.get("category") or "").strip()
        if x:
            cats.append(x)
    if not cats:
        return None
    return Counter(cats).most_common(1)[0][0]


def refine_protocol_confidences_gemini(
    model,
    q: str,
    parsed: dict,
    retrieved: list[dict],
) -> bool:
    """Второй короткий вызов модели для калибровки confidence_score (опционально)."""
    protos = parsed.get("protocols") or []
    if not protos:
        return False
    ex_by_path: dict[str, str] = {}
    for r in retrieved:
        p = r.get("path") or ""
        if p and p not in ex_by_path:
            ex_by_path[p] = str(r.get("excerpt") or "")[:500]
    lines: list[str] = []
    for pr in protos[:8]:
        if not isinstance(pr, dict):
            continue
        p = str(pr.get("path") or "")
        ex = ex_by_path.get(p, "")
        lines.append(f"path={p}\ntitle={pr.get('title')}\nфрагмент: {ex}\n")
    if not lines:
        return False
    prompt = (
        SYSTEM_CONFIDENCE_REFINE
        + "\n\nЗапрос:\n"
        + (q or "")[:6000]
        + "\n\nПротоколы:\n"
        + "\n---\n".join(lines)
    )
    try:
        resp = generate_gemini(model, prompt)
        txt = _extract_gemini_text(resp)
        pj = _try_parse_json(txt)
    except Exception:
        return False
    if not pj or isinstance(pj, bool) or not isinstance(pj, dict):
        return False
    scores = pj.get("scores") or []
    by_path: dict[str, float] = {}
    for s in scores:
        if not isinstance(s, dict):
            continue
        p = str(s.get("path") or "")
        try:
            c = float(s.get("confidence_score"))
        except (TypeError, ValueError):
            continue
        if p:
            by_path[p] = max(0.0, min(1.0, c))
    if not by_path:
        return False
    mix = float(os.environ.get("RAG_CONFIDENCE_SECOND_BLEND", "0.55"))
    touched = False
    for pr in protos:
        if not isinstance(pr, dict):
            continue
        p = str(pr.get("path") or "")
        if p not in by_path:
            continue
        cur = _confidence_numeric(pr.get("confidence_score"))
        if cur is None:
            cur = 0.55
        new = mix * by_path[p] + (1.0 - mix) * cur
        pr["confidence_score"] = round(max(0.0, min(1.0, new)), 4)
        pr["confidence_second_pass"] = True
        touched = True
    return touched


def _merge_embed_pool_rows(
    scored: list[tuple],
    pool_n: int,
    merge_on: bool,
) -> list[tuple]:
    """Топ по score + доп. кандидаты с высоким BM25, чтобы не терять чанки вне первых N."""
    if not scored:
        return []
    pool_n = min(int(pool_n), len(scored))
    if not merge_on or len(scored) <= pool_n:
        return scored[:pool_n]
    primary = scored[:pool_n]
    primary_ids: set[int] = set()
    for row in primary:
        ch = row[4] if len(row) >= 5 else row[3]
        primary_ids.add(id(ch))
    bm25_i = 2
    by_bm25 = sorted(scored, key=lambda x: -float(x[bm25_i]))
    cap = min(len(scored), max(pool_n * 2, pool_n + 24))
    out = list(primary)
    seen = set(primary_ids)
    for row in by_bm25:
        if len(out) >= cap:
            break
        ch = row[4] if len(row) >= 5 else row[3]
        if id(ch) in seen:
            continue
        seen.add(id(ch))
        out.append(row)
    out.sort(key=lambda x: -float(x[0]))
    return out


_GAP_FIELD_LABELS_RU: dict[str, str] = {
    "investigations": "обследование (диагностика)",
    "medications": "препараты и группы лекарственных средств",
    "treatment_methods": "лечение и методы",
    "monitoring_frequency": "кратность наблюдения",
}

_NON_PROTOCOL_MARK = "[не из протокола]"


def _norm_str_list_ext(val: object) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    return []


def _append_note_field(existing: object, addition: str) -> str:
    e = str(existing or "").strip()
    a = str(addition or "").strip()
    if not a:
        return e
    if not e:
        return a
    return e + " " + a


def _merge_str_lists_unique(a: list[str], b: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in a + b:
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _detailed_block_missing_keys(ext: dict) -> list[str]:
    missing: list[str] = []
    if not _norm_str_list_ext(ext.get("investigations")):
        missing.append("investigations")
    if not _norm_str_list_ext(ext.get("medications")):
        missing.append("medications")
    if not _norm_str_list_ext(ext.get("treatment_methods")):
        missing.append("treatment_methods")
    mf = ext.get("monitoring_frequency")
    if not (str(mf).strip() if mf is not None else ""):
        missing.append("monitoring_frequency")
    return missing


def _merge_gap_into_ext(ext: dict, gap: dict, allowed_keys: list[str]) -> None:
    if "investigations" in allowed_keys:
        g = _norm_str_list_ext(gap.get("investigations"))
        if g:
            ext["investigations"] = _merge_str_lists_unique(
                _norm_str_list_ext(ext.get("investigations")), g
            )
    if "medications" in allowed_keys:
        g = _norm_str_list_ext(gap.get("medications"))
        if g:
            ext["medications"] = _merge_str_lists_unique(
                _norm_str_list_ext(ext.get("medications")), g
            )
    if "treatment_methods" in allowed_keys:
        g = _norm_str_list_ext(gap.get("treatment_methods"))
        if g:
            ext["treatment_methods"] = _merge_str_lists_unique(
                _norm_str_list_ext(ext.get("treatment_methods")), g
            )
    if "monitoring_frequency" in allowed_keys:
        g = gap.get("monitoring_frequency")
        s = str(g).strip() if g is not None else ""
        if s and not str(ext.get("monitoring_frequency") or "").strip():
            ext["monitoring_frequency"] = s


def _ensure_non_protocol_prefix(s: str) -> str:
    t = str(s).strip()
    if not t:
        return ""
    if t.startswith(_NON_PROTOCOL_MARK):
        return t
    return _NON_PROTOCOL_MARK + " " + t


def _merge_non_protocol_into_ext(ext: dict, raw: dict, allowed_keys: list[str]) -> None:
    if "investigations" in allowed_keys:
        items = _norm_str_list_ext(raw.get("investigations"))
        if items:
            fixed = [_ensure_non_protocol_prefix(x) for x in items]
            ext["investigations"] = _merge_str_lists_unique(
                _norm_str_list_ext(ext.get("investigations")), fixed
            )
    if "medications" in allowed_keys:
        items = _norm_str_list_ext(raw.get("medications"))
        if items:
            fixed = [_ensure_non_protocol_prefix(x) for x in items]
            ext["medications"] = _merge_str_lists_unique(
                _norm_str_list_ext(ext.get("medications")), fixed
            )
    if "treatment_methods" in allowed_keys:
        items = _norm_str_list_ext(raw.get("treatment_methods"))
        if items:
            fixed = [_ensure_non_protocol_prefix(x) for x in items]
            ext["treatment_methods"] = _merge_str_lists_unique(
                _norm_str_list_ext(ext.get("treatment_methods")), fixed
            )
    if "monitoring_frequency" in allowed_keys:
        if not str(ext.get("monitoring_frequency") or "").strip():
            g = raw.get("monitoring_frequency")
            s = str(g).strip() if g is not None else ""
            if s:
                ext["monitoring_frequency"] = _ensure_non_protocol_prefix(s)


def _run_gap_fill_scan(
    *,
    model,
    query: str,
    title_line: str,
    spec: str,
    body: str,
    extra: str,
    missing_keys: list[str],
    plim: int,
) -> dict | None:
    labels = [_GAP_FIELD_LABELS_RU[k] for k in missing_keys if k in _GAP_FIELD_LABELS_RU]
    fields_ru = "; ".join(labels)
    keys_json = json.dumps(missing_keys, ensure_ascii=False)
    head = SYSTEM_EXTRACT_GAP_SCAN.format(
        fields_ru=fields_ru,
        keys_json=keys_json,
    )
    prompt = (
        head
        + "\n\n---\n\n"
        + f"Запрос пользователя:\n{query}\n\n"
        + f"Специальность (рубрика каталога): {spec}\n"
        + f"Название протокола: {title_line}\n\n"
        + "Текст протокола (фрагменты PDF):\n"
        + body
        + extra
    )
    if len(prompt) > plim:
        prompt = prompt[: plim - 80] + "\n…[обрезано]"
    try:
        resp = generate_gemini(model, prompt)
        txt = _extract_gemini_text(resp)
        parsed = _try_parse_json(txt)
    except (HTTPException, Exception):
        return None
    if not parsed or not isinstance(parsed, dict):
        return None
    return parsed


def _run_non_protocol_fill(
    *,
    model,
    query: str,
    title_line: str,
    spec: str,
    missing_keys: list[str],
) -> dict | None:
    labels = [_GAP_FIELD_LABELS_RU[k] for k in missing_keys if k in _GAP_FIELD_LABELS_RU]
    fields_ru = "; ".join(labels)
    keys_json = json.dumps(missing_keys, ensure_ascii=False)
    prompt = SYSTEM_EXTRACT_NON_PROTOCOL.format(
        spec=spec,
        title=title_line,
        query=query[:8000],
        fields_ru=fields_ru,
        keys_json=keys_json,
    )
    try:
        resp = generate_gemini(model, prompt)
        txt = _extract_gemini_text(resp)
        parsed = _try_parse_json(txt)
    except (HTTPException, Exception):
        return None
    if not parsed or not isinstance(parsed, dict):
        return None
    return parsed


def _extract_prompt_char_limit() -> int:
    """Лимит символов промпта для извлечения по протоколу (отдельно от общего чата)."""
    v = (os.environ.get("RAG_EXTRACT_PROMPT_MAX_CHARS") or "").strip()
    if v.isdigit():
        return max(4000, int(v))
    return int(os.environ.get("GEMINI_PROMPT_MAX_CHARS", "28000"))


def _clamp_detail_string_list(vals: list[str], max_item: int) -> list[str]:
    out: list[str] = []
    for x in vals:
        s = str(x).strip()
        if not s:
            continue
        if len(s) > max_item:
            s = s[: max(1, max_item - 1)] + "…"
        out.append(s)
    return out


def _clamp_detail_ext_lists(ext: dict) -> None:
    mic = int(os.environ.get("RAG_EXTRACT_ITEM_MAX_CHARS", "420"))
    mic = max(120, mic)
    for k in ("investigations", "medications", "treatment_methods"):
        if isinstance(ext.get(k), list):
            ext[k] = _clamp_detail_string_list(
                [str(x) for x in ext[k] if str(x).strip()],
                mic,
            )


_DETAIL_FOCUS_EXTRA: dict[str, str] = {
    "investigations": (
        "Приоритет извлечения: максимально полно заполни investigations и при необходимости diagnosis; "
        "medications и treatment_methods - только если явно следуют из текста протокола по запросу."
    ),
    "medications": (
        "Приоритет извлечения: максимально полно medications; не включай сюда пункты обследования (investigations)."
    ),
    "treatment_methods": (
        "Приоритет извлечения: максимально полно treatment_methods (этапы лечения, операции, режим); "
        "отделяй от диагностики."
    ),
    "monitoring_frequency": (
        "Приоритет извлечения: monitoring_frequency (сроки и частота визитов, диспансеризация) "
        "и при необходимости monitoring_followup (срочные ситуации); не дублируй кратность в recommendations."
    ),
    "care_algorithms": (
        "Приоритет извлечения: care_algorithms - пошаговые алгоритмы ведения/неотложной помощи, "
        "ветвления «если/то», критерии эскалации и госпитализации; верни структурированно."
    ),
}


def _normalize_detail_extract_focus(raw: str | None) -> str | None:
    if not raw or not isinstance(raw, str):
        return None
    x = raw.strip().lower()
    if x == "monitoring":
        x = "monitoring_frequency"
    if x == "algorithms":
        x = "care_algorithms"
    if x in _DETAIL_FOCUS_EXTRA:
        return x
    return None


def _algo_marker_score(text: str) -> float:
    if not text:
        return 0.0
    t = text.lower()
    pats = [
        r"\bалгоритм\w*",
        r"\bэтап\w*",
        r"\bпошаг\w*",
        r"\bесли\b.{0,80}\bто\b",
        r"\bпри\s+отсутствии\s+эффект",
        r"\bпоказания?\b.{0,70}\bгоспитал",
        r"\bнеотложн\w+\s+помощ",
    ]
    score = 0.0
    for p in pats:
        if re.search(p, t, re.IGNORECASE | re.DOTALL):
            score += 0.14
    return max(0.0, min(1.0, score))


def _normalize_algorithm_rows(raw: object) -> list[dict]:
    out: list[dict] = []
    if not raw:
        return out
    rows = raw if isinstance(raw, list) else [raw]
    idx = 0
    for row in rows:
        idx += 1
        if isinstance(row, str):
            s = row.strip()
            if not s:
                continue
            out.append(
                {
                    "id": f"alg_{idx}",
                    "title": f"Алгоритм {idx}",
                    "entry_conditions": [],
                    "steps": [s],
                }
            )
            continue
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip() or f"Алгоритм {idx}"
        ent = row.get("entry_conditions")
        steps = row.get("steps")
        entry_conditions = _norm_str_list_ext(ent)
        steps_norm = _norm_str_list_ext(steps)
        if not steps_norm:
            fallback = _norm_str_list_ext(row.get("actions"))
            if fallback:
                steps_norm = fallback
        if not steps_norm:
            continue
        out.append(
            {
                "id": f"alg_{idx}",
                "title": title[:220],
                "entry_conditions": entry_conditions[:8],
                "steps": steps_norm[:24],
            }
        )
    return out[:12]


def _fallback_algorithms_from_ext(ext: dict) -> list[dict]:
    recs = _norm_str_list_ext(ext.get("recommendations"))
    tms = _norm_str_list_ext(ext.get("treatment_methods"))
    pool = recs + tms
    if not pool:
        return []
    picked: list[str] = []
    for x in pool:
        t = x.strip()
        if not t:
            continue
        if re.search(r"\b(если|то|этап|показан|неотлож|госпитал|алгоритм)\b", t, re.I):
            picked.append(t)
        if len(picked) >= 10:
            break
    if len(picked) < 3:
        return []
    return [
        {
            "id": "alg_1",
            "title": "Алгоритм ведения по протоколу",
            "entry_conditions": [],
            "steps": picked,
        }
    ]


def infer_specialties_gemini(q: str, model) -> list[str]:
    """Опционально: первый короткий вызов LLM - к каким рубрикам относится запрос."""
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
    protocol_confidence: float | None = None,
    extract_focus: str | None = None,
    client_rag_support: float | None = None,
) -> dict | None:
    """Второй вызов LLM: факты по протоколу; при detailed - расширенная схема и больший объём текста."""
    focus_key = _normalize_detail_extract_focus(extract_focus)
    detail_prompt_truncated = False
    if detailed:
        max_body = int(os.environ.get("RAG_EXTRACT_FULL_MATCH_MAX_CHARS", "32000"))
        idx_lim = 16000
        summary_lim = min(4096, idx_lim)
        system = SYSTEM_EXTRACT_FULL
    else:
        max_body = int(os.environ.get("RAG_EXTRACT_MAX_CHARS", "16000"))
        idx_lim = 8000
        summary_lim = 4000
        system = SYSTEM_EXTRACT
    body = gather_protocol_text(path, max_body)
    struct = _structured_by_path.get(path) or {}
    extra = ""
    if struct.get("summary"):
        extra += (
            "\n\n[Выдержка индекса: краткое содержание]\n"
            + format_structured_index_text(str(struct["summary"]), summary_lim)
        )
    if struct.get("diagnosis"):
        extra += (
            "\n\n[Выдержка индекса: диагностика]\n"
            + format_structured_index_text(str(struct["diagnosis"]), idx_lim)
        )
    if struct.get("treatment"):
        extra += (
            "\n\n[Выдержка индекса: лечение]\n"
            + format_structured_index_text(str(struct["treatment"]), idx_lim)
        )
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
    if focus_key:
        prompt += "\n\n---\n" + _DETAIL_FOCUS_EXTRA[focus_key]
    plim = _extract_prompt_char_limit()
    if len(prompt) > plim:
        prompt = prompt[: plim - 80] + "\n…[обрезано для лимита контекста]"
        detail_prompt_truncated = True
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

    def _norm_str_list(val: object) -> list[str]:
        if val is None:
            return []
        if isinstance(val, str):
            s = val.strip()
            return [s] if s else []
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        return []

    ext: dict = {
        "diagnosis": parsed.get("diagnosis") or "",
        "treatment_methods": parsed.get("treatment_methods") or [],
        "medications": parsed.get("medications") or [],
        "note": parsed.get("note") or "",
    }
    if detailed:
        ext["investigations"] = _norm_str_list(parsed.get("investigations"))
        mf = parsed.get("monitoring_frequency")
        ext["monitoring_frequency"] = (
            str(mf).strip() if mf is not None and str(mf).strip() else ""
        )
        ext["recommendations"] = parsed.get("recommendations") or []
        ext["monitoring_followup"] = parsed.get("monitoring_followup") or ""
        ext["care_algorithms"] = _normalize_algorithm_rows(parsed.get("care_algorithms"))
        ext["contraindications"] = parsed.get("contraindications") or ""
        ext["detailed"] = True
        if not ext["care_algorithms"]:
            ext["care_algorithms"] = _fallback_algorithms_from_ext(ext)
        _clamp_detail_ext_lists(ext)
        gap_on = os.environ.get("RAG_EXTRACT_GAP_RETRY", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if gap_on:
            missing = _detailed_block_missing_keys(ext)
            if missing:
                gap_parsed = _run_gap_fill_scan(
                    model=model,
                    query=query,
                    title_line=title_line,
                    spec=spec,
                    body=body,
                    extra=extra,
                    missing_keys=missing,
                    plim=plim,
                )
                if gap_parsed:
                    n_before = len(_detailed_block_missing_keys(ext))
                    _merge_gap_into_ext(ext, gap_parsed, missing)
                    _clamp_detail_ext_lists(ext)
                    if len(_detailed_block_missing_keys(ext)) < n_before:
                        ext["note"] = _append_note_field(
                            ext.get("note"),
                            "Выполнен повторный поиск по полному тексту протокола для пустых разделов (обследование, препараты, лечение, кратность наблюдения).",
                        )
            missing2 = _detailed_block_missing_keys(ext)
            np_on = os.environ.get(
                "RAG_EXTRACT_NON_PROTOCOL_FALLBACK", "0"
            ).strip().lower() in (
                "1",
                "true",
                "yes",
            )
            np_mon_only = os.environ.get(
                "RAG_EXTRACT_NON_PROTOCOL_MONITORING_ONLY", "0"
            ).strip().lower() in (
                "1",
                "true",
                "yes",
            )
            np_keys = list(missing2)
            if np_on and np_keys:
                if np_mon_only:
                    np_keys = [k for k in np_keys if k == "monitoring_frequency"]
                if np_keys:
                    np_parsed = _run_non_protocol_fill(
                        model=model,
                        query=query,
                        title_line=title_line,
                        spec=spec,
                        missing_keys=np_keys,
                    )
                    if np_parsed:
                        n_before_np = len(_detailed_block_missing_keys(ext))
                        _merge_non_protocol_into_ext(ext, np_parsed, np_keys)
                        _clamp_detail_ext_lists(ext)
                        if len(_detailed_block_missing_keys(ext)) < n_before_np:
                            ext["note"] = _append_note_field(
                                ext.get("note"),
                                "Формулировки с пометкой «[не из протокола]» - общеклинические ориентиры, не цитата из клинического протокола.",
                            )
    if os.environ.get("RAG_EXTRACT_GROUNDING", "1").strip().lower() in ("1", "true", "yes"):
        try:
            from clinical_knowledge.item_grounding import build_extraction_grounding
            from clinical_knowledge.protocol_icd_profile_index import (
                _load_profile_index_by_path,
            )

            _min_sup = float(os.environ.get("RAG_EXTRACT_GROUNDING_MIN_SUPPORT", "0.34"))
            _prof = _load_profile_index_by_path().get(path.replace("\\", "/"))
            _chunks_for_ground = _chunks_by_path.get(path) or []
            if _chunks_for_ground:
                ext["grounding"] = build_extraction_grounding(
                    ext, _chunks_for_ground, profile_entry=_prof, min_support=_min_sup
                )
                if os.environ.get("RAG_EXTRACT_GROUNDING_DROP", "0").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    for _f in ("medications", "investigations", "treatment_methods"):
                        rows = ext["grounding"].get(_f) or []
                        if rows:
                            ext[_f] = [r["text"] for r in rows if r.get("verified")]
        except Exception:
            pass
    warn_thr = float(os.environ.get("RAG_DETAIL_WARN_RAG_SUPPORT", "0.22"))
    low_sup = False
    if client_rag_support is not None:
        try:
            low_sup = float(client_rag_support) < warn_thr
        except (TypeError, ValueError):
            low_sup = False
    out: dict = {
        "path": path,
        "title": title_line,
        "specialty_ru": spec or None,
        "category": meta.get("category"),
        "extraction": ext,
        "detail_prompt_truncated": detail_prompt_truncated,
        "extract_focus_applied": focus_key,
        "low_protocol_match_support": low_sup,
    }
    algo_src = "\n".join(
        [
            str(ext.get("diagnosis") or ""),
            "\n".join(_norm_str_list_ext(ext.get("recommendations"))),
            "\n".join(_norm_str_list_ext(ext.get("treatment_methods"))),
            body[:12000],
        ]
    )
    algo_conf = _algo_marker_score(algo_src)
    has_algos = bool(_normalize_algorithm_rows(ext.get("care_algorithms")))
    if has_algos:
        algo_conf = max(algo_conf, 0.62)
    out["algorithm_confidence"] = round(float(max(0.0, min(1.0, algo_conf))), 4)
    out["is_algorithmic_protocol"] = bool(has_algos or algo_conf >= 0.42)
    if out["is_algorithmic_protocol"] and not has_algos:
        out["algorithm_warnings"] = [
            "В тексте есть признаки алгоритма, но явная структура шагов извлечена частично."
        ]
    if client_rag_support is not None:
        try:
            out["client_rag_support"] = float(
                max(0.0, min(1.0, float(client_rag_support)))
            )
        except (TypeError, ValueError):
            pass
    if protocol_confidence is not None:
        out["detail_match_score"] = protocol_confidence
    return out


def retrieve(
    query: str,
    max_chunks: int | None = None,
    max_per_path: int = 2,
    routing_query: str | None = None,
    category_boost: list[str] | None = None,
    user_category_slugs: list[str] | None = None,
    icd_codes_for_lex: list[str] | None = None,
    path_boost: list[str] | None = None,
    path_allowlist: list[str] | None = None,
    catalog_path_extra: list[str] | None = None,
    embed_rerank: bool | None = None,
    audience_hint: str | None = None,
) -> list[dict]:
    """Сериализация retrieve на Render (512Mi) - один активный lexical-pass."""
    with _retrieve_sem:
        return _retrieve_core(
            query,
            max_chunks=max_chunks,
            max_per_path=max_per_path,
            routing_query=routing_query,
            category_boost=category_boost,
            user_category_slugs=user_category_slugs,
            icd_codes_for_lex=icd_codes_for_lex,
            path_boost=path_boost,
            path_allowlist=path_allowlist,
            catalog_path_extra=catalog_path_extra,
            embed_rerank=embed_rerank,
            audience_hint=audience_hint,
        )


def _retrieve_core(
    query: str,
    max_chunks: int | None = None,
    max_per_path: int = 2,
    routing_query: str | None = None,
    category_boost: list[str] | None = None,
    user_category_slugs: list[str] | None = None,
    icd_codes_for_lex: list[str] | None = None,
    path_boost: list[str] | None = None,
    path_allowlist: list[str] | None = None,
    catalog_path_extra: list[str] | None = None,
    embed_rerank: bool | None = None,
    audience_hint: str | None = None,
) -> list[dict]:
    """Лексический отбор + множители из symptom_routing.json (если RAG_ROUTING=1).

    query - короткий текст для подсчёта совпадений с чанками (обычно только жалобы).
    routing_query - полный запрос для правил возраста/рубрик; если None, берётся query.
    category_boost - slug рубрик из опционального LLM-классификатора запроса.
    user_category_slugs - рубрики, выбранные пользователем в форме: усиление совпадений и штраф нерелевантных чанков.
    icd_codes_for_lex - нормализованные коды МКБ-10: дополнительные лексические токены и усиление чанков, где встречается код.
    path_boost - пути PDF протоколов (source_path): усиление чанков из matched protocol cards.
    path_allowlist - если задан, учитываются только чанки с path из этого списка (строгий режим КЗ).
    audience_hint - 'adult' | 'child' из воронки; фильтр несовместимых PDF до embed rerank.
    """
    from clinical_knowledge.rich_chunk_search import (
        build_chunk_match_reason,
        chunk_population_penalty,
        chunk_type_multiplier,
        enrich_lex_source,
    )

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
    aud_filter: str | None = None
    if use_routing:
        aud_filter = (audience_hint or "").strip().lower() or None
        if aud_filter not in ("adult", "child"):
            aud_filter = infer_audience_from_query(rq, _routing)
        if aud_filter is None:
            aud_filter = infer_audience_from_funnel_context(rq)
    aud_strict = os.environ.get("RAG_AUDIENCE_FILTER", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    icd_lex = icd_tokens_for_lex(icd_codes_for_lex or [])
    # Коды вида J20.9 в самом запросе не попадают в tokenize_ru - извлекаем отдельно.
    icd_from_query = icd_tokens_for_lex(extract_icd_codes_raw(query))
    qtok = (
        set(tokenize_ru(query))
        | icd_lex
        | icd_from_query
        | _extra_clinical_tokens(rq)
    )
    if not qtok:
        return []
    anchor_list = _anchor_tokens(qtok)
    anchor_set = frozenset(anchor_list)
    generic_w = float(os.environ.get("RAG_GENERIC_LEX_WEIGHT", "0.22"))
    anchor_miss_penalty = float(os.environ.get("RAG_ANCHOR_MISS_PENALTY", "0.045"))
    icd_chunk_boost = float(os.environ.get("RAG_ICD_CHUNK_BOOST", "1.65"))
    icd_norms = [
        c.strip().lower()
        for c in (icd_codes_for_lex or [])
        if isinstance(c, str) and len(c.strip()) >= 3
    ]
    path_boost_set = frozenset(
        p.strip() for p in (path_boost or []) if isinstance(p, str) and p.strip()
    )
    path_allowlist_set = frozenset(
        p.replace("\\", "/").strip()
        for p in (path_allowlist or [])
        if isinstance(p, str) and p.strip()
    )
    from clinical_knowledge.lazy_rag_config import (
        forbid_full_corpus_retrieve,
        lazy_retrieve_enabled,
        path_lex_shards_enabled,
        startup_mode,
    )

    lazy_active = lazy_retrieve_enabled() or startup_mode() == "manifest"
    lazy_chunks_pool: list[dict] | None = None
    if lazy_active:
        if not path_allowlist_set and forbid_full_corpus_retrieve():
            return []
        if path_allowlist_set:
            store = _ensure_lazy_chunk_store()
            if store:
                per_path = max(4, int(max_per_path) * 4)
                lazy_chunks_pool = store.get_chunks_for_paths(
                    list(path_allowlist_set),
                    max_chunks_per_path=per_path,
                    max_total=max(256, int(max_chunks or 6) * 32),
                )
                if path_lex_shards_enabled() and lazy_chunks_pool:
                    pli = _ensure_path_lex_index()
                    if pli:
                        chunk_ids = {
                            str(c.get("chunk_id"))
                            for c in lazy_chunks_pool
                            if c.get("chunk_id")
                        }
                        rubrics: set[str] = set()
                        if _path_manifest:
                            for p in path_allowlist_set:
                                ent = _path_manifest.get(p)
                                if ent and ent.rubric:
                                    rubrics.add(ent.rubric)
                        matched = pli.query_chunk_ids(
                            query,
                            rubrics=sorted(rubrics),
                            chunk_id_allowlist=chunk_ids,
                        )
                        if matched:
                            filtered = [
                                c
                                for c in lazy_chunks_pool
                                if str(c.get("chunk_id") or "") in matched
                            ]
                            if filtered:
                                lazy_chunks_pool = filtered
            if not lazy_chunks_pool and not _chunks:
                return []
    path_boost_factor = float(os.environ.get("RAG_MATCHED_PROTOCOL_PATH_BOOST", "1.85"))
    bm25_alpha = float(os.environ.get("RAG_LEX_BM25_ALPHA", "0.55"))
    use_bm25_blend = _bm25_index is not None and bm25_alpha < 0.999
    pool_merge = os.environ.get("RAG_EMBED_POOL_MERGE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    raw_rows: list[tuple[float, float, float, dict, float]] = []
    use_inverted = os.environ.get("RAG_LEX_INVERTED_INDEX", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    candidate_indices: set[int] | None = None
    if lazy_chunks_pool is not None:
        candidate_indices = None
    elif path_allowlist_set:
        path_cand = _chunk_indices_for_path_allowlist(path_allowlist_set)
        if path_cand:
            candidate_indices = path_cand
    elif use_inverted:
        lex_idx = _ensure_lex_inverted_index()
        max_union = env_int("RAG_LEX_MAX_UNION", 12000 if env_bool("RENDER", False) else 0)
        tokens_ordered = sorted(
            qtok,
            key=lambda t: len(lex_idx.get(t, ())),
        )
        cand: set[int] = set()
        for t in tokens_ordered:
            posts = lex_idx.get(t)
            if posts:
                cand |= set(posts)
            if max_union > 0 and len(cand) >= max_union:
                break
        if cand:
            candidate_indices = cand
    try:
        from clinical_knowledge.vector_index import ensure_index_loaded, index_stats, search, vector_index_enabled

        if vector_index_enabled():
            ensure_index_loaded()
        if vector_index_enabled() and index_stats().get("loaded"):
            v_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if v_key:
                q_embed_vec = (query + "\n" + " ".join(icd_codes_for_lex or [])).strip()[:8000]
                ex_emb_v = _extra_clinical_tokens(rq)
                if ex_emb_v:
                    q_embed_vec = (q_embed_vec + " " + " ".join(sorted(ex_emb_v))).strip()[:8000]
                v_model = os.environ.get(
                    "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2-preview"
                ).strip()
                q_vec = _gemini_embed_one(v_model, q_embed_vec, "retrieval_query")
                v_top = int(os.environ.get("RAG_VECTOR_TOP_K", "200"))
                vector_hits = search(q_vec, top_k=v_top)
                if vector_hits:
                    if candidate_indices is not None:
                        candidate_indices |= vector_hits
                    else:
                        candidate_indices = set(vector_hits)
    except Exception:
        pass
    candidate_indices = _cap_lex_candidate_indices(
        candidate_indices, path_allowlist_set=path_allowlist_set, qtok=qtok
    )
    if lazy_chunks_pool is not None:
        chunk_source = iter(lazy_chunks_pool)
    elif candidate_indices is not None:
        chunk_source = (_chunks[i] for i in sorted(candidate_indices))
    else:
        chunk_source = iter(_chunks)
    for ch in chunk_source:
        pth = ch.get("path") or ""
        if path_allowlist_set and not _path_matches_allowlist(ch, path_allowlist_set):
            continue
        if aud_filter and aud_strict and use_routing:
            aud_path = _retrieval_path_for_chunk(ch)
            if _chunk_audience_mismatch(
                aud_filter, aud_path, str(ch.get("title") or ""), _routing
            ):
                continue
        lex_src = enrich_lex_source(ch)
        low = lex_src.lower()
        lex_icd = normalize_text_for_icd_scan(lex_src).lower()
        lex = 0.0
        for t in qtok:
            if t not in low:
                continue
            wt = 1.0 + min(len(t), 10) * 0.02
            if t in RAG_GENERIC_LEX:
                wt *= generic_w
            lex += wt
        if lex <= 0:
            continue
        if anchor_set:
            if not any(t in low for t in anchor_set):
                lex *= anchor_miss_penalty
        bm25_s = 0.0
        if _bm25_index is not None:
            bm25_s = _bm25_index.score_doc(qtok, ch)
        mult = (
            routing_multiplier(rq, ch, _routing)
            if use_routing
            else 1.0
        )
        post = 1.0
        if icd_norms:
            if any(code in lex_icd for code in icd_norms):
                post *= icd_chunk_boost
            else:
                post *= float(
                    os.environ.get("RAG_ICD_QUERY_MISS_CHUNK_MULT", "0.62")
                )
        pth = ch.get("path") or ""
        catalog_pth = _retrieval_path_for_chunk(ch)
        if path_boost_set and catalog_pth in path_boost_set:
            post *= path_boost_factor
        post *= _protocol_meta_icd_boost(catalog_pth, icd_norms)
        post *= _protocol_catalog_icd_boost(catalog_pth, icd_norms)
        cat = (ch.get("category") or "").strip()
        if boost_set and cat in boost_set:
            post *= boost_factor
        if user_slugs:
            if cat and cat in user_slugs:
                post *= user_boost
            elif cat and cat not in user_slugs:
                post *= user_penalty
            else:
                post *= user_uncertain
        if (ch.get("kind") or "").strip() in ("table_block", "table"):
            ql = (query or "").lower()
            if (
                any(c.isdigit() for c in query)
                or "таблиц" in ql
                or "доз" in ql
                or "мг" in ql
                or "мкг" in ql
                or "мл" in ql
                or "сут" in ql
            ):
                post *= float(os.environ.get("RAG_TABLE_BLOCK_BOOST", "1.14"))
        post *= chunk_type_multiplier(query, ch, icd_codes=icd_norms)
        post *= chunk_population_penalty(aud_filter, ch)
        if ch.get("generated_from_summary") or ch.get("chunk_source") == "summary_chunks":
            ps_mode = (os.environ.get("PROTOCOL_SUMMARY_MODE") or "legacy").strip().lower()
            summary_first = env_bool("SEARCH_RAG_SUMMARY_FIRST", True)
            if summary_first and icd_norms and ps_mode == "legacy":
                ps_mode = "hybrid"
            summary_boost = 1.0
            if ps_mode == "hybrid":
                summary_boost = float(os.environ.get("RAG_SUMMARY_CHUNK_BOOST", "1.55"))
            elif ps_mode == "summary":
                summary_boost = float(os.environ.get("RAG_SUMMARY_CHUNK_BOOST", "1.75"))
            elif icd_norms:
                summary_boost = float(os.environ.get("RAG_SUMMARY_ICD_FIRST_BOOST", "1.85"))
            if summary_boost > 1.0:
                post *= summary_boost
            if icd_norms:
                chunk_icd = ch.get("icd10_codes") or []
                for raw_icd in chunk_icd:
                    c = normalize_icd_code(str(raw_icd)).strip().lower()
                    if not c:
                        continue
                    if any(c == n or c.startswith(n) or n.startswith(c) for n in icd_norms):
                        post *= float(os.environ.get("RAG_SUMMARY_ICD_MATCH_BOOST", "1.25"))
                        break
        raw_rows.append((lex, bm25_s, mult, ch, post))
    if not raw_rows:
        return []

    lex_vals = [r[0] for r in raw_rows]
    bm25_vals = [r[1] for r in raw_rows]
    lex_n = _norm_minmax(lex_vals)
    bm25_n = _norm_minmax(bm25_vals)
    scored: list[tuple[float, float, float, float, dict]] = []
    for i, row in enumerate(raw_rows):
        lex, bm25_s, mult, ch, post = row
        ln = lex_n[i]
        bn = bm25_n[i]
        if use_bm25_blend:
            blend = bm25_alpha * ln + (1.0 - bm25_alpha) * bn
        else:
            blend = ln
        final = blend * mult * post
        scored.append((final, lex, bm25_s, mult, ch))
    # Детерминированный порядок при равных score: стабильный tie-break по пути и индексу чанка.
    scored.sort(
        key=lambda x: (
            -x[0],
            str(x[4].get("path", "")),
            str(x[4].get("chunk_index", "")),
        )
    )

    embed_meta: dict = {"used": False}
    prefilter_meta: dict[str, Any] = {"used": False}

    prefilter_on = os.environ.get("RAG_PREFILTER_BEFORE_EMBED", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if prefilter_on and scored and (icd_norms or user_slugs or boost_set):
        from clinical_knowledge.protocol_summary.icd_index import (
            build_retrieval_prefilter_context,
            chunk_matches_retrieval_prefilter,
        )

        slug_union = frozenset(user_slugs | boost_set)
        pf_ctx = build_retrieval_prefilter_context(
            icd_codes_for_lex,
            sorted(slug_union),
            extra_catalog_paths=list(catalog_path_extra or []) + list(path_boost or []),
        )
        if pf_ctx.get("active"):
            catalog_paths = pf_ctx["paths"]  # type: ignore[index]
            cat_slugs = pf_ctx["slugs"]  # type: ignore[index]
            pool_before = len(scored)
            filtered_rows = [
                row
                for row in scored
                if chunk_matches_retrieval_prefilter(
                    row[4],
                    catalog_paths=catalog_paths,  # type: ignore[arg-type]
                    category_slugs=cat_slugs,  # type: ignore[arg-type]
                    icd_norms=icd_norms,
                )
            ]
            min_ratio = float(os.environ.get("RAG_PREFILTER_MIN_KEEP_RATIO", "0.35"))
            min_abs = int(os.environ.get("RAG_PREFILTER_MIN_KEEP", "12"))
            min_keep = max(min_abs, int(pool_before * min_ratio))
            if len(filtered_rows) >= min_keep:
                scored = filtered_rows
                prefilter_meta = {
                    "used": True,
                    "pool_before": pool_before,
                    "pool_after": len(scored),
                    "reduction_pct": round(100.0 * (1.0 - len(scored) / pool_before), 1),
                    "routing_version": int(_routing.get("version", 1)) if _routing else 1,
                    "icd_catalog_paths": len(catalog_paths),  # type: ignore[arg-type]
                    "category_slugs": sorted(cat_slugs)[:8],  # type: ignore[arg-type]
                }

    embed_on = (
        embed_rerank
        if embed_rerank is not None
        else os.environ.get("RAG_GEMINI_EMBED_RERANK", "1").strip().lower()
        in ("1", "true", "yes")
    )
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    pool_n = env_int("RAG_EMBED_POOL", 44)
    alpha = env_float("RAG_HYBRID_ALPHA", 0.46)
    emb_model = os.environ.get(
        "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2-preview"
    ).strip()

    work_rows: list[tuple] = scored
    q_embed = query
    if icd_codes_for_lex:
        q_embed = (query + "\n" + " ".join(icd_codes_for_lex)).strip()[:8000]
    ex_emb = _extra_clinical_tokens(rq)
    if ex_emb:
        q_embed = (q_embed + " " + " ".join(sorted(ex_emb))).strip()[:8000]
    if embed_on and api_key and scored:
        pool_n = min(pool_n, len(scored))
        pool_rows = _merge_embed_pool_rows(scored, pool_n, pool_merge)
        try:
            precomputed = _precomputed_chunk_embed_rerank_pool(
                q_embed, pool_rows, alpha, emb_model
            )
            if precomputed is not None:
                work_rows = precomputed
                embed_meta = {
                    "used": True,
                    "model": emb_model,
                    "alpha": alpha,
                    "pool": len(pool_rows),
                    "precomputed_docs": True,
                }
            else:
                work_rows = _gemini_embed_rerank_pool(q_embed, pool_rows, alpha, emb_model)
                embed_meta = {
                    "used": True,
                    "model": emb_model,
                    "alpha": alpha,
                    "pool": len(pool_rows),
                }
        except Exception as e:
            work_rows = scored
            embed_meta = {"used": False, "error": str(e)[:240]}

    embed_meta["prefilter"] = prefilter_meta
    _set_retrieval_embed_meta(embed_meta)
    per_path: dict[str, int] = {}
    per_basename: dict[str, int] = {}
    max_per_basename = int(os.environ.get("RAG_MAX_CHUNKS_PER_BASENAME", "1"))
    out: list[dict] = []
    rerank_used = bool(embed_meta.get("used"))
    for row in work_rows:
        if len(row) >= 5:
            final, lex, _bm25_s, mult, ch = (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
            )
        else:
            final, lex, mult, ch = row[0], row[1], row[2], row[3]
        p_raw = ch.get("path") or ""
        p = _retrieval_path_for_chunk(ch)
        if per_path.get(p, 0) >= max_per_path:
            continue
        bk = _protocol_basename_key(p)
        if bk and per_basename.get(bk, 0) >= max(1, max_per_basename):
            continue
        per_path[p] = per_path.get(p, 0) + 1
        if bk:
            per_basename[bk] = per_basename.get(bk, 0) + 1
        ex_lim = int(os.environ.get("RAG_EXCERPT_CHARS", "700"))
        cat_out = (ch.get("category") or "").strip()
        row_out: dict = {
            "path": p,
            "title": ch.get("title") or "",
            "kind": ch.get("kind") or "general",
            "score": round(final, 3),
            "lexical_score": round(lex, 3),
            "routing_multiplier": round(mult, 4),
            "excerpt": format_excerpt_for_display(ch.get("text") or "", ex_lim),
        }
        if p_raw.startswith("summary://"):
            row_out["chunk_source"] = "summary_chunks"
            row_out["generated_from_summary"] = True
        catalog = str(ch.get("catalog_source_path") or "").strip()
        if catalog:
            row_out["catalog_source_path"] = catalog.replace("\\", "/")
        if cat_out:
            row_out["category"] = cat_out
        # Структурная привязка (если корпус её содержит и RAG_KEEP_STRUCT включён).
        sec_title = (ch.get("section_title") or "").strip()
        if not sec_title:
            sp = ch.get("section_path")
            if isinstance(sp, list) and sp:
                sec_title = str(sp[-1]).strip()
        if sec_title:
            row_out["section_title"] = sec_title
        chunk_icd = ch.get("icd10_codes")
        if isinstance(chunk_icd, list) and chunk_icd:
            row_out["icd10_codes"] = chunk_icd[:12]
        pf = int(ch.get("page_from") or 0)
        pt = int(ch.get("page_to") or 0)
        if pf:
            row_out["page_from"] = pf
            row_out["page_to"] = pt or pf
        pts = ch.get("point_numbers")
        if isinstance(pts, list) and pts:
            row_out["point_numbers"] = pts[:12]
        if rerank_used:
            row_out["embedding_rerank"] = True
        row_out["match_reason"] = build_chunk_match_reason(row_out, icd_norms or None)
        row_out["chunk_type"] = ch.get("kind") or ch.get("chunk_type") or "body"
        out.append(row_out)
        if len(out) >= max_chunks:
            break
    if env_bool("RENDER", False):
        import gc as _gc

        _gc.collect()
    return out


# Большой промпт и вызов модели могут занимать 2-3+ мин; клиент в index.html ждёт дольше сервера
GEMINI_CALL_TIMEOUT = float(os.environ.get("GEMINI_CALL_TIMEOUT", "180"))
GEMINI_SPELLFIX_TIMEOUT = float(os.environ.get("GEMINI_SPELLFIX_TIMEOUT", "45"))
GEMINI_QUERY_REFINE_TIMEOUT = float(os.environ.get("RAG_QUERY_REFINE_TIMEOUT", "45"))
GEMINI_CONSULT_DIGEST_TIMEOUT = float(
    os.environ.get("GEMINI_CONSULT_DIGEST_TIMEOUT", str(GEMINI_QUERY_REFINE_TIMEOUT))
)
GEMINI_CONSULT_RAG_REFINE_TIMEOUT = float(
    os.environ.get("GEMINI_CONSULT_RAG_REFINE_TIMEOUT", "42")
)
GEMINI_CONSULT_REVIEW_SYNTH_TIMEOUT = float(
    os.environ.get("GEMINI_CONSULT_REVIEW_SYNTH_TIMEOUT", "120")
)
GEMINI_METHODIST_AI_REVIEW_TIMEOUT = float(
    os.environ.get("GEMINI_METHODIST_AI_REVIEW_TIMEOUT", "90")
)

_methodist_model = None
_methodist_model_name: str | None = None
_methodist_model_warn: str | None = None


def reset_methodist_gemini_cache() -> None:
    global _methodist_model, _methodist_model_name, _methodist_model_warn
    _methodist_model = None
    _methodist_model_name = None
    _methodist_model_warn = None


def get_gemini():
    global _model
    if _model is not None:
        return _model
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=503,
            detail="На сервере не настроен ключ API для обработки текста.",
        )
    try:
        genai = _legacy_genai_module()
        HarmBlockThreshold, HarmCategory = _legacy_genai_types()
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail="На сервере не установлены зависимости для обработки текста (requirements-rag.txt).",
        ) from e
    genai.configure(api_key=key)
    from clinical_knowledge.gemini_model_config import main_gemini_model_name

    name, _warn = main_gemini_model_name()
    # Единые safety-настройки (как в gemini_verify) - иначе медицинский текст чаще даёт пустой ответ.
    safety = [
        {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
        {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
        {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
        {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
    ]
    _model = genai.GenerativeModel(name, safety_settings=safety)
    return _model


def get_methodist_gemini():
    """Отдельная модель для AI-оценки методиста (GEMINI_METHODIST_MODEL или gemini-2.5-pro)."""
    global _methodist_model, _methodist_model_name, _methodist_model_warn
    if _methodist_model is not None:
        return _methodist_model
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=503,
            detail="На сервере не настроен ключ API для обработки текста.",
        )
    try:
        genai = _legacy_genai_module()
        HarmBlockThreshold, HarmCategory = _legacy_genai_types()
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail="На сервере не установлены зависимости для обработки текста (requirements-rag.txt).",
        ) from e
    genai.configure(api_key=key)
    from clinical_knowledge.gemini_model_config import methodist_gemini_model_name

    name, warn = methodist_gemini_model_name()
    _methodist_model_name = name
    _methodist_model_warn = warn
    safety = [
        {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
        {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
        {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
        {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
    ]
    _methodist_model = genai.GenerativeModel(name, safety_settings=safety)
    return _methodist_model


def _is_quota_error(e: Exception) -> bool:
    s = str(e).lower()
    return (
        "429" in s
        or "quota" in s
        or "resource_exhausted" in s
        or "rate limit" in s
        or "ratelimit" in s
    )


def _run_model_with_retry(fn, model, full_prompt: str, timeout: float):
    """Вызов модели в отдельном потоке с таймаутом и retry при 429/quota (экспоненциальный backoff)."""
    attempts = max(0, env_int("GEMINI_QUOTA_RETRY", 2))
    base_delay = env_float("GEMINI_QUOTA_RETRY_DELAY", 2.0)
    last_quota_err: Exception | None = None
    for i in range(attempts + 1):
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn, model, full_prompt)
            try:
                return fut.result(timeout=timeout)
            except FuturesTimeout as e:
                raise HTTPException(
                    status_code=504,
                    detail=f"Таймаут вызова модели ({int(timeout)} с). Проверьте сеть или настройки модели на сервере.",
                ) from e
            except Exception as e:
                if _is_quota_error(e) and i < attempts:
                    last_quota_err = e
                    time.sleep(base_delay * (i + 1))
                    continue
                raise
    raise HTTPException(
        status_code=503,
        detail="Лимит запросов к модели (quota). Повторите попытку через минуту.",
    ) from last_quota_err


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


def _make_generation_config(genai, *, max_output_tokens: int, json_mode: bool):
    """Единый конфиг генерации с детерминированными настройками.

    Воспроизводимость важна для медицинского инструмента: одинаковый вход должен давать одинаковый
    результат. По умолчанию temperature=0 (жадное декодирование) и candidate_count=1.
    Переопределяется через GEMINI_TEMPERATURE / GEMINI_TOP_P / GEMINI_TOP_K / GEMINI_SEED.
    """
    kw: dict = {
        "temperature": env_float("GEMINI_TEMPERATURE", 0.0),
        "max_output_tokens": max_output_tokens,
        "candidate_count": 1,
    }
    if (os.environ.get("GEMINI_TOP_P") or "").strip():
        kw["top_p"] = env_float("GEMINI_TOP_P", 1.0)
    if (os.environ.get("GEMINI_TOP_K") or "").strip():
        kw["top_k"] = env_int("GEMINI_TOP_K", 1)
    if json_mode:
        # Снижает обрывы посреди JSON и обрывы «лишнего» текста до/после объекта
        kw["response_mime_type"] = "application/json"
    seed_raw = (os.environ.get("GEMINI_SEED") or "").strip()
    if seed_raw:
        try:
            return genai.GenerationConfig(seed=int(seed_raw), **kw)
        except (TypeError, ValueError):
            # Старая версия SDK без поддержки seed - детерминизм обеспечивает temperature=0.
            pass
    return genai.GenerationConfig(**kw)


def _generate_blocking(model, full_prompt: str):
    genai = _legacy_genai_module()

    max_out = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "16384"))
    use_json = os.environ.get("GEMINI_JSON_MODE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    return model.generate_content(
        full_prompt,
        generation_config=_make_generation_config(genai, max_output_tokens=max_out, json_mode=use_json),
    )


def generate_gemini(model, full_prompt: str):
    """Один поток + таймаут + retry при 429/quota - иначе вызов к API может «висеть» или падать по лимиту."""
    return _run_model_with_retry(_generate_blocking, model, full_prompt, GEMINI_CALL_TIMEOUT)


def _generate_blocking_plain(model, full_prompt: str):
    """Текст без JSON mode - шаблоны заключений и т.п."""
    genai = _legacy_genai_module()

    max_out = int(os.environ.get("GEMINI_TEMPLATE_MAX_TOKENS", "8192"))
    return model.generate_content(
        full_prompt,
        generation_config=_make_generation_config(genai, max_output_tokens=max_out, json_mode=False),
    )


def generate_gemini_plain(model, full_prompt: str):
    return _run_model_with_retry(_generate_blocking_plain, model, full_prompt, GEMINI_CALL_TIMEOUT)


def _generate_blocking_spellfix(model, full_prompt: str):
    """Короткий JSON-ответ: исправление опечаток в запросе."""
    genai = _legacy_genai_module()

    return model.generate_content(
        full_prompt,
        generation_config=_make_generation_config(genai, max_output_tokens=1024, json_mode=True),
    )


def generate_gemini_spellfix(model, full_prompt: str):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_generate_blocking_spellfix, model, full_prompt)
        try:
            return fut.result(timeout=GEMINI_SPELLFIX_TIMEOUT)
        except FuturesTimeout:
            return None


def _generate_blocking_query_refine(model, full_prompt: str):
    """JSON: нормализация жалобы под МКБ/протоколы."""
    genai = _legacy_genai_module()

    return model.generate_content(
        full_prompt,
        generation_config=_make_generation_config(genai, max_output_tokens=3072, json_mode=True),
    )


def generate_gemini_query_refine(model, full_prompt: str):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_generate_blocking_query_refine, model, full_prompt)
        try:
            return fut.result(timeout=GEMINI_QUERY_REFINE_TIMEOUT)
        except FuturesTimeout:
            return None


def _generate_blocking_consult_pdf_digest(model, full_prompt: str):
    """JSON: извлечение клинического ядра из текста PDF КЗ под RAG."""
    genai = _legacy_genai_module()

    return model.generate_content(
        full_prompt,
        generation_config=_make_generation_config(genai, max_output_tokens=2048, json_mode=True),
    )


def generate_gemini_consult_pdf_digest(model, full_prompt: str):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_generate_blocking_consult_pdf_digest, model, full_prompt)
        try:
            return fut.result(timeout=GEMINI_CONSULT_DIGEST_TIMEOUT)
        except FuturesTimeout:
            return None


def _generate_blocking_consult_rag_second_pass(model, full_prompt: str):
    """JSON: уточнение строки запроса RAG между проходами на проверке КЗ."""
    genai = _legacy_genai_module()

    return model.generate_content(
        full_prompt,
        generation_config=_make_generation_config(genai, max_output_tokens=1024, json_mode=True),
    )


def generate_gemini_consult_rag_second_pass(model, full_prompt: str):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_generate_blocking_consult_rag_second_pass, model, full_prompt)
        try:
            return fut.result(timeout=GEMINI_CONSULT_RAG_REFINE_TIMEOUT)
        except FuturesTimeout:
            return None


def _consult_review_synth_max_tokens() -> int:
    return int(os.environ.get("GEMINI_CONSULT_REVIEW_MAX_TOKENS", "16384"))


def _make_blocking_consult_review_synth(max_out: int):
    def _fn(model, full_prompt: str):
        genai = _legacy_genai_module()

        return model.generate_content(
            full_prompt,
            generation_config=_make_generation_config(
                genai, max_output_tokens=max_out, json_mode=True
            ),
        )

    return _fn


def generate_gemini_consult_review_synthesize(model, full_prompt: str, *, max_out: int | None = None):
    """Финальная оценка КЗ - отдельный таймаут для надёжности SSE."""
    mo = max_out or _consult_review_synth_max_tokens()
    return _run_model_with_retry(
        _make_blocking_consult_review_synth(mo),
        model,
        full_prompt,
        GEMINI_CONSULT_REVIEW_SYNTH_TIMEOUT,
    )


def generate_gemini_methodist_ai_review(model, full_prompt: str):
    """AI-оценка для кабинета методиста (этап 2 после детерминированного анализа)."""
    mo = env_int("GEMINI_METHODIST_AI_MAX_TOKENS", 4096)

    def _call(m, prompt, *, json_mode: bool):
        cfg: dict = {"temperature": 0.2, "max_output_tokens": mo}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        return m.generate_content(prompt, generation_config=cfg)

    def _fn(m, prompt):
        try:
            return _call(m, prompt, json_mode=True)
        except Exception as exc:
            if "response_mime_type" in str(exc).lower() or "json" in str(exc).lower():
                return _call(m, prompt, json_mode=False)
            raise

    try:
        return _run_model_with_retry(_fn, model, full_prompt, GEMINI_METHODIST_AI_REVIEW_TIMEOUT)
    except Exception as exc:
        err = str(exc).lower()
        if "not found" in err or "is not supported" in err:
            reset_methodist_gemini_cache()
            return _run_model_with_retry(_fn, get_methodist_gemini(), full_prompt, GEMINI_METHODIST_AI_REVIEW_TIMEOUT)
        raise


def refine_clinical_query_gemini(
    complaint_rag: str, full_query: str, model
) -> tuple[str, dict | None]:
    """Уточнение формулировки жалобы через Gemini для лучшего совпадения с МКБ и RAG."""
    sq = (complaint_rag or "").strip()
    if len(sq) < 3 or len(sq) > 8000:
        return complaint_rag, None
    ctx = (full_query or "").strip()[:4500]
    prompt = (
        SYSTEM_CLINICAL_QUERY_REFINE
        + "\n\n---\n\nТекст жалобы (основной):\n"
        + sq[:6000]
        + "\n\nДополнительный контекст запроса (если есть возраст/пол/шапка - только для согласования формулировок, не выдумывай факты):\n"
        + ctx
    )
    try:
        resp = generate_gemini_query_refine(model, prompt)
        if resp is None:
            return complaint_rag, None
        txt = _extract_gemini_text(resp)
        parsed = _try_parse_json(txt)
        if not parsed or not isinstance(parsed, dict):
            return complaint_rag, None
        refined = (parsed.get("refined") or "").strip()
        if not refined or len(refined) > 12000:
            return complaint_rag, None
        # Защита от чрезмерного сжатия/обнуления
        if len(sq) >= 40 and len(refined) < max(12, int(len(sq) * 0.15)):
            return complaint_rag, None
        applied = bool(parsed.get("applied"))
        note = (parsed.get("note") or "").strip()
        if refined == sq and not applied:
            return complaint_rag, None
        if refined == sq:
            applied = False
        meta: dict = {
            "applied": applied,
            "before": sq,
            "after": refined,
        }
        if note:
            meta["note"] = note
        return refined, meta
    except (HTTPException, Exception):
        return complaint_rag, None


def apply_clinical_correction(full_q: str, corrected_rag: str) -> str:
    """Подставляет исправленный клинический текст в полный запрос (контекст + ответы на уточнения сохраняются)."""
    cq = (corrected_rag or "").strip()
    sep = "=== Жалобы и вопрос ==="
    if sep in full_q:
        head = full_q.split(sep, 1)[0] + sep + "\n"
        tail = full_q.split(sep, 1)[1]
        mark = " - Ответы на уточняющие вопросы:"
        if mark in tail:
            return head + cq + "\n\n" + mark + tail.split(mark, 1)[1]
        return head + cq
    return cq


def fix_query_spelling_medical(short_query: str, model) -> tuple[str, bool]:
    """Исправление опечаток для лексического поиска. При сбое API - исходный текст, changed=False."""
    sq = (short_query or "").strip()
    if len(sq) < 2 or len(sq) > 8000:
        return short_query, False
    prompt = SYSTEM_QUERY_SPELLFIX + "\n\nТекст:\n" + sq[:6000]
    try:
        resp = generate_gemini_spellfix(model, prompt)
        if resp is None:
            return short_query, False
        txt = _extract_gemini_text(resp)
        parsed = _try_parse_json(txt)
        if not parsed or not isinstance(parsed, dict):
            return short_query, False
        corrected = (parsed.get("corrected") or "").strip()
        if not corrected:
            return short_query, False
        if corrected == sq:
            return short_query, False
        return corrected, True
    except (HTTPException, Exception):
        return short_query, False


def _repair_truncated_json(s: str) -> dict | None:
    """Попытка восстановить усечённый JSON-объект (обрыв по лимиту токенов).

    Закрывает незавершённую строку, убирает «висящую» запятую/двоеточие и достраивает
    недостающие закрывающие скобки. Возвращает dict или None.
    """
    if not s:
        return None
    start = s.find("{")
    if start < 0:
        return None
    s = s[start:]
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack:
                stack.pop()
    repaired = s
    if in_str:
        repaired += '"'
    repaired = repaired.rstrip()
    # убрать висящие разделители в конце усечённого фрагмента
    repaired = re.sub(r"[:,]\s*$", "", repaired)
    repaired += "".join(reversed(stack))
    try:
        out = json.loads(repaired)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


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
        return _repair_truncated_json(s)


_DIAG_LINE_HINTS_RU: tuple[str, ...] = (
    "диагноз",
    "заключени",
    "рекомендац",
    "мкб-10",
    "мкб 10",
    "международн",
    "код по мкб",
    "клиническ",
    "основной",
    "сопутствующ",
    "жалоб",
    "анамнез",
    "объективно",
    "локальный статус",
    "осмотр",
    "пальпац",
    "аускульт",
    "узи ",
    " узд ",
    " кт ",
    " мрт ",
    "эхокг",
    "эндоскоп",
    "операци",
    "послеоперац",
    "госпитал",
    "объективный статус",
)


def _consult_heuristic_focus_text(pdf_plain: str, *, max_chars: int = 4500) -> str:
    """Выдёргивает из КЗ строки с клиническими маркерами и блоки с кодами МКБ (без LLM)."""
    if not pdf_plain.strip():
        return ""
    norm_full = normalize_text_for_icd_scan(pdf_plain.replace("\u00a0", " "))
    lines = norm_full.replace("\r", "\n").split("\n")
    picked: list[str] = []
    seen_norm: set[str] = set()
    for ln in lines:
        s = (ln or "").strip()
        if not s or len(s) > 480:
            continue
        low = s.lower()
        hit = False
        for hint in _DIAG_LINE_HINTS_RU:
            if hint.strip() and hint.lower() in low:
                hit = True
                break
        if not hit and ICD10_CODE_RE.search(normalize_text_for_icd_scan(s)):
            hit = True
        if hit:
            key = low[:280]
            if key in seen_norm:
                continue
            seen_norm.add(key)
            picked.append(s)
    out = "\n".join(picked).strip()
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out


def _consult_gemini_clinical_focus_text(model, pdf_digest_input: str) -> tuple[str | None, dict]:
    """Один короткий вызов модели - клиническое «ядро» для поиска по протоколам."""
    meta: dict = {"ok": False}
    pdf_digest_input = (pdf_digest_input or "").strip()
    if not pdf_digest_input:
        return None, meta
    use = os.environ.get("CONSULT_REVIEW_GEMINI_DIGEST", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not use:
        meta["reason"] = "disabled_via_env"
        return None, meta
    cap = max(3500, int(os.environ.get("CONSULT_REVIEW_DIGEST_PROMPT_CHARS", "12000")))
    body = pdf_digest_input[:cap]
    prompt = (
        SYSTEM_CONSULT_PDF_FOR_PROTOCOL_SEARCH
        + "\n\n--- ТЕКСТ ИЗ PDF (может быть с обрывами) ---\n\n"
        + body
    )
    try:
        resp = generate_gemini_consult_pdf_digest(model, prompt)
        if resp is None:
            meta["reason"] = "timeout_or_empty_response"
            return None, meta
        txt = _extract_gemini_text(resp)
        parsed = _try_parse_json(txt)
        if not parsed or not isinstance(parsed, dict):
            meta["reason"] = "bad_json"
            return None, meta
        clinic = (parsed.get("clinical_search_text") or "").strip()
        conf = str(parsed.get("confidence") or "").strip().lower()
        meta["confidence"] = conf or "unknown"
        meta["ok"] = bool(clinic)
        if not clinic:
            meta["reason"] = "empty_clinical_search_text"
            return None, meta
        return clinic, meta
    except HTTPException:
        meta["reason"] = "http_exception"
        return None, meta
    except Exception as e:
        meta["reason"] = "error"
        meta["detail"] = str(e)[:200]
        return None, meta


def _merge_icd_codes_for_consult_retrieval(
    icd_analysis: dict, full_pdf_text: str
) -> tuple[list[str], dict[str, object]]:
    """Коды МКБ для RAG: сначала блок «Диагноз», затем остальной PDF, затем pipeline."""
    max_n = max(5, min(16, int(os.environ.get("CONSULT_REVIEW_ICD_MERGE_CAP", "12"))))
    from_diag = extract_icd_codes_diagnosis_focused(full_pdf_text or "")
    from_pdf_raw = extract_icd_codes_raw(full_pdf_text or "")
    from_pipe = icd_analysis.get("codes_for_retrieval") if isinstance(icd_analysis, dict) else None
    if not isinstance(from_pipe, list):
        from_pipe = []
    seen: set[str] = set()
    merged: list[str] = []

    def push(code: object) -> None:
        if not isinstance(code, str):
            return
        n = normalize_icd_code(code.strip())
        if not n or n in seen:
            return
        seen.add(n)
        merged.append(n)

    for c in from_diag:
        push(c)
    for c in from_pdf_raw:
        push(c)
    for c in from_pipe:
        push(str(c))
    trimmed = merged[:max_n]
    dn = {normalize_icd_code(x) for x in from_diag}
    meta: dict[str, object] = {
        "diag_block_icd_codes": list(from_diag),
        "codes_for_merge_order": trimmed,
        "cap_applied": max_n,
        "codes_outside_diag_block_pdf": [
            normalize_icd_code(x) for x in from_pdf_raw if normalize_icd_code(x) not in dn
        ],
    }
    return trimmed, meta


def _consult_row_text_for_icd_scan(row: dict) -> str:
    chunks: list[str] = []
    for key in ("title", "path"):
        raw = row.get(key)
        if isinstance(raw, str) and raw.strip():
            chunks.append(raw.strip())
    for key in ("excerpt", "text"):
        raw = row.get(key)
        if isinstance(raw, str) and raw.strip():
            chunks.append(raw.strip())
        elif isinstance(raw, dict):
            t = raw.get("text") or raw.get("excerpt")
            if isinstance(t, str) and t.strip():
                chunks.append(t.strip())
    return "\n".join(chunks)


def _consult_needles_icd_fragments_consult_review(
    diag_block_icd: list[str],
    merged_icd: list[str],
) -> list[str]:
    if diag_block_icd:
        base = diag_block_icd
    elif merged_icd:
        base = merged_icd
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in base:
        if not isinstance(raw, str):
            continue
        n = normalize_icd_code(raw.strip())
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
        if len(out) >= 12:
            break
    return out


def _consult_path_icd_match_meta(
    retrieved: list[dict], icd_needles: list[str]
) -> dict[str, tuple[bool, float, list[str]]]:
    """path → есть ли совпадение кода МКБ в хотя бы одном фрагменте; лучший score; коды."""
    paths_blob: dict[str, list[tuple[float, dict]]] = {}
    if not retrieved or not icd_needles:
        return {}

    def row_score(rd: dict) -> float:
        try:
            return float(rd.get("score") or 0)
        except (TypeError, ValueError):
            return 0.0

    for row in retrieved:
        if not isinstance(row, dict):
            continue
        pth = str(row.get("path") or "").strip()
        if not pth:
            continue
        paths_blob.setdefault(pth, []).append((row_score(row), row))

    out: dict[str, tuple[bool, float, list[str]]] = {}
    for pth, pairs in paths_blob.items():
        best_s = max((sc for sc, _ in pairs), default=0.0)
        hits: list[str] = []
        matched = False
        for _, row in pairs:
            blob = _consult_row_text_for_icd_scan(row)
            if not blob:
                continue
            for code in icd_needles:
                if text_mentions_icd_code(blob, code):
                    n = normalize_icd_code(code)
                    if n and n not in hits:
                        hits.append(n)
            if hits:
                matched = True
        out[pth] = (matched, best_s, hits)
    return out


def _consult_sort_retrieval_by_icd_fragments_first(
    retrieved: list[dict], icd_needles: list[str]
) -> list[dict]:
    if not retrieved or not icd_needles:
        return retrieved
    meta = _consult_path_icd_match_meta(retrieved, icd_needles)

    def key_row(row: dict) -> tuple[int, float]:
        if not isinstance(row, dict):
            return 1, 0.0
        pth = str(row.get("path") or "").strip()
        m = meta.get(pth)
        hit = bool(m and m[0])
        try:
            sc = float(row.get("score") or 0)
        except (TypeError, ValueError):
            sc = 0.0
        return (0 if hit else 1, -sc)

    return sorted(retrieved, key=key_row)


def _consult_precise_links_for_icd_in_fragments(
    retrieved: list[dict],
    *,
    diag_block_icd: list[str],
    merged_icd: list[str],
) -> tuple[list[dict], str]:
    """Протоколы из отбора с явным упоминанием кода диагноза в тексте переданной выдержки."""
    icd_needles = _consult_needles_icd_fragments_consult_review(diag_block_icd, merged_icd)
    if not icd_needles:
        return [], (
            "В блоке диагноза консультативного заключения не найдено однозначных кодов МКБ‑10 для "
            "сопоставления с текстом фрагментов протоколов. Ниже - полный результат автоматического отбора."
        )

    paths_meta = _consult_path_icd_match_meta(retrieved, icd_needles)
    rows_list: list[tuple[float, dict]] = []
    for path, tup in paths_meta.items():
        matched, best_s, hit_codes = tup
        if not matched:
            continue
        pr = _protocols_by_path.get(path) or {}
        ttl = _protocol_display_title(path, str(pr.get("title") or "").strip() or None)
        rows_list.append(
            (
                -best_s,
                {
                    "path": path,
                    "title": ttl,
                    "matched_icd_codes": hit_codes,
                },
            )
        )
    rows_list.sort(key=lambda x: x[0])
    from clinical_knowledge.protocol_links import dedupe_protocol_rows, protocol_link_payload

    slim: list[dict] = []
    for r in rows_list:
        row = r[1]
        payload = protocol_link_payload(
            row.get("path"),
            title=row.get("title"),
            matched_icd_codes=row.get("matched_icd_codes"),
            icd_verified=True,
        )
        if payload:
            slim.append(payload)
        else:
            slim.append(row)
    slim = dedupe_protocol_rows(slim)
    if slim:
        return slim, ""

    quoted = ", ".join(icd_needles[:8])
    note = (
        f"Для кодов ({quoted}) в переданных выдержках из протоколов не найдено явного упоминания МКБ - "
        "многие КП задают содержание по рубрике без кодов в оглавлении. Ниже - полный автоматический отбор."
    )
    return [], note


def _consult_icd_banner_for_retrieval(icd_diag: list[str], icd_ordered: list[str]) -> str:
    lines: list[str] = []
    di = [
        normalize_icd_code(str(c))
        for c in (icd_diag or [])
        if isinstance(c, str) and normalize_icd_code(str(c))
    ]
    if di:
        lines.append(
            "КОДЫ МКБ ИЗ ФОРМУЛИРОВОК ДИАГНОЗА В ЗАКЛЮЧЕНИИ: " + ", ".join(di[:14])
        )
    oc = []
    seen: set[str] = set()
    for raw in icd_ordered or []:
        if not isinstance(raw, str):
            continue
        n = normalize_icd_code(raw.strip())
        if n and n not in seen:
            seen.add(n)
            oc.append(n)
        if len(oc) >= 14:
            break
    if oc:
        lines.append("УПОРЯДОЧЕННЫЙ СПИСОК КОДОВ МКБ ДЛЯ ПОИСКА: " + ", ".join(oc))
    return "\n".join(lines).strip()


def _consult_review_paths_hint(
    paths_used_hint: list[str],
    *,
    retrieved: list[dict],
    icd_needles: list[str],
) -> list[str]:
    if not paths_used_hint:
        return paths_used_hint
    pm = _consult_path_icd_match_meta(retrieved, icd_needles) if icd_needles else {}
    prio = [p for p in paths_used_hint if pm.get(p, (False, 0.0, []))[0]]
    rest = [p for p in paths_used_hint if p not in prio]
    return prio + rest


def _consult_retrieval_quality_metrics(retrieved: list[dict]) -> dict[str, float | int]:
    """Компактные метрики первого отбора для решения о второй проходке retrieve."""
    if not retrieved:
        return {
            "max_score": 0.0,
            "top3_lex_avg": 0.0,
            "n_chunks": 0,
            "uniq_paths": 0,
        }
    paths: set[str] = set()
    scores: list[float] = []
    lexs: list[float] = []
    for r in retrieved:
        if not isinstance(r, dict):
            continue
        pth = str(r.get("path") or "").strip()
        if pth:
            paths.add(pth)
        try:
            scores.append(float(r.get("score") or 0))
        except (TypeError, ValueError):
            scores.append(0.0)
        try:
            lexs.append(float(r.get("lexical_score") or 0))
        except (TypeError, ValueError):
            lexs.append(0.0)
    top_lex = sorted(lexs, reverse=True)[:3]
    return {
        "max_score": max(scores) if scores else 0.0,
        "top3_lex_avg": (sum(top_lex) / len(top_lex)) if top_lex else 0.0,
        "n_chunks": len(retrieved),
        "uniq_paths": len(paths),
    }


def _merge_chunk_retrieval_lists(
    buckets: list[list[dict]],
    *,
    max_chunks: int,
    max_per_path: int,
) -> list[dict]:
    """Объединяет несколько списков чанков retrieve(): по убыванию score, разнообразие по path."""
    pool: list[dict] = []
    for lst in buckets:
        if lst:
            pool.extend(lst)
    if not pool:
        return []
    pool.sort(key=lambda r: float((r.get("score") if isinstance(r, dict) else 0) or 0), reverse=True)
    per_path: dict[str, int] = {}
    out: list[dict] = []
    for row in pool:
        if not isinstance(row, dict):
            continue
        pth = str(row.get("path") or "").strip()
        if not pth:
            continue
        if per_path.get(pth, 0) >= max_per_path:
            continue
        per_path[pth] = per_path.get(pth, 0) + 1
        out.append(row)
        if len(out) >= max_chunks:
            break
    return out


def _consult_candidates_blob_for_second_pass(retrieved: list[dict]) -> tuple[str, int]:
    """Текстовый блок кандидатов для промпта второй проходки."""
    lines: list[str] = []
    lim = max(4, min(14, int(os.environ.get("CONSULT_REVIEW_RAG_SECOND_PASS_CANDIDATES", "8"))))
    n = 0
    seen_path: set[str] = set()
    for row in retrieved:
        if not isinstance(row, dict):
            continue
        p = str(row.get("path") or "").strip()
        if not p or p in seen_path:
            continue
        seen_path.add(p)
        ttl = _protocol_display_title(p, str(row.get("title") or "").strip() or None)
        body = _retrieval_fragment_body(row).replace("\n", " ").strip()
        if len(body) > 420:
            body = body[:419].rstrip() + "…"
        lines.append(f"- path: {p}\n  title: {ttl}\n  fragment_start: {body}")
        n += 1
        if n >= lim:
            break
    return "\n".join(lines), n


def _consult_titles_fallback_augment(
    q_rag: str, retrieved: list[dict], *, cap: int = 2600
) -> tuple[str, dict]:
    """Добавляет к запросу названия к протоколов из первого отбора без вызова LLM."""
    base = (q_rag or "").strip()
    titles: list[str] = []
    seen: set[str] = set()
    for row in retrieved:
        if not isinstance(row, dict):
            continue
        tt = str(row.get("title") or "").strip()
        if tt and tt not in seen:
            seen.add(tt)
            titles.append(tt)
        if len(titles) >= 14:
            break
    extra = ""
    if titles:
        extra = "\nНазвания протоколов-кандидатов (первый отбор): " + "; ".join(titles)
    merged = (base + extra).strip()[:cap]
    return merged, {"source": "titles_append", "titles_used": len(titles)}


def _consult_gemini_second_pass_augment(
    model,
    *,
    q_rag: str,
    rq_trim: str,
    focus_preview: str,
    candidates_blob: str,
) -> tuple[str | None, dict]:
    meta: dict = {"ok": False}
    use = os.environ.get("CONSULT_REVIEW_RAG_SECOND_PASS_GEMINI", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not use:
        meta["reason"] = "disabled_via_env"
        return None, meta
    if not (q_rag or "").strip():
        meta["reason"] = "empty_q_rag"
        return None, meta
    max_out = max(480, min(3200, int(os.environ.get("CONSULT_REVIEW_RAG_SECOND_PASS_MAX_CHARS", "2200"))))
    qr = (q_rag or "").strip()[:7600]
    rq_s = (rq_trim or "").strip()[:5200]
    fp = (focus_preview or "").strip()[:900]
    cb = (candidates_blob or "").strip()
    if not cb:
        meta["reason"] = "no_candidates_blob"
        return None, meta
    prompt = (
        SYSTEM_CONSULT_RAG_SECOND_PASS_QUERY
        + "\n\n--- ТЕКУЩИЙ ЗАПРОС ДЛЯ RAG (из заключения) ---\n"
        + qr
        + "\n\n--- КОРОТКИЙ ФОКУС ИЗ PDF (если был) ---\n"
        + (fp if fp else "(нет)")
        + "\n\n--- МАРШРУТИЗАЦИЯ (фрагмент полного синтетического запроса) ---\n"
        + (rq_s if rq_s else "(нет)")
        + "\n\n--- КАНДИДАТЫ ПЕРВОГО ОТБОРА ---\n"
        + cb
    )
    try:
        resp = generate_gemini_consult_rag_second_pass(model, prompt)
        if resp is None:
            meta["reason"] = "timeout_or_empty_response"
            return None, meta
        txt = _extract_gemini_text(resp)
        parsed = _try_parse_json(txt)
        if not parsed or not isinstance(parsed, dict):
            meta["reason"] = "bad_json"
            return None, meta
        refined = (parsed.get("refined_search_text") or "").strip()
        note = (parsed.get("draft_note") or "").strip()
        conf_c = str(parsed.get("confidence_in_candidates") or "").strip().lower()
        meta["draft_note"] = note[:420]
        meta["confidence_in_candidates"] = conf_c or ""
        meta["ok"] = bool(refined)
        meta["parsed_len"] = len(refined)
        if len(refined) > max_out:
            refined = refined[: max_out - 1].rstrip() + "…"
            meta["truncated_to"] = max_out
        if not refined or len(refined) < 72:
            meta["reason"] = "too_short_or_empty_refined"
            return None, meta
        meta["source"] = "gemini_second_pass"
        return refined, meta
    except HTTPException:
        meta["reason"] = "http_exception"
        return None, meta
    except Exception as e:
        meta["reason"] = "error"
        meta["detail"] = str(e)[:200]
        return None, meta


def _consult_second_pass_build_query(
    model,
    q_rag: str,
    rq: str,
    focus_meta: dict,
    retrieved: list[dict],
) -> tuple[str, dict]:
    """Строка для второго retrieve: черновой анализ кандидатов + Gemini или заголовочный fallback."""
    out_meta: dict = {
        "candidates_summarized": 0,
        "chosen_source": "",
    }
    blob, n_seen = _consult_candidates_blob_for_second_pass(retrieved)
    out_meta["candidates_summarized"] = n_seen
    preview = ""
    if isinstance(focus_meta, dict):
        preview = str(focus_meta.get("focus_preview") or "").strip()

    rq_cap = rq.strip()[:6000]

    refined, gm = _consult_gemini_second_pass_augment(
        model,
        q_rag=q_rag,
        rq_trim=rq_cap,
        focus_preview=preview,
        candidates_blob=blob,
    )
    out_meta["gemini"] = gm

    if refined and len(refined.strip()) >= 40:
        out_meta["chosen_source"] = "gemini_second_pass"
        max_total = max(800, int(os.environ.get("CONSULT_REVIEW_RAG_SECOND_PASS_JOIN_MAX", "8000")))
        return refined.strip()[:max_total], out_meta

    fb, fb_meta = _consult_titles_fallback_augment(q_rag, retrieved)
    out_meta["fallback"] = fb_meta
    out_meta["chosen_source"] = fb_meta.get("source", "titles_append")
    return fb, out_meta


def _consult_should_second_pass(metrics: dict) -> tuple[bool, str]:
    """Низкая «уверенность» по скорингам первого батча - повод запустить второй retrieve."""
    if not metrics:
        return False, "no_metrics"
    if not _consult_rag_second_pass_enabled():
        return False, "disabled_env"

    max_sc = float(metrics.get("max_score") or 0)
    thr = float(os.environ.get("CONSULT_REVIEW_RAG_LOW_MAX_SCORE", "0.32"))
    lex3 = float(metrics.get("top3_lex_avg") or 0)
    lex_thr = float(os.environ.get("CONSULT_REVIEW_RAG_LOW_LEX_AVG", "0"))
    uniq = int(metrics.get("uniq_paths") or 0)
    min_paths = int(os.environ.get("CONSULT_REVIEW_RAG_SECOND_PASS_MIN_UNIQUE_PATHS", "3"))

    if max_sc < thr:
        return True, f"max_score_below_{thr}"
    if lex_thr > 0 and lex3 > 0 and lex3 < lex_thr:
        return True, f"weak_top3_lex_below_{lex_thr}"
    if 0 < uniq < min_paths:
        return True, f"few_protocols_{uniq}_lt_{min_paths}"
    return False, "confidence_ok"


def _build_consult_review_pipeline_query(model, full_text: str) -> tuple[str, dict]:
    """Синтетический запрос под цепочку МКБ+refine+RAG без «шумной» шапки PDF."""
    sep = "=== Жалобы и вопрос ==="
    meta: dict = {
        "focus_source": "",
        "focus_chars": 0,
        "gemini_digest": None,
    }
    q_slice = max(2000, int(os.environ.get("CONSULT_REVIEW_RAG_QUERY_CHARS", "9000")))
    dig_in_lim = max(3500, int(os.environ.get("CONSULT_REVIEW_DIGEST_INPUT_CHARS", "14000")))
    min_focus = max(50, int(os.environ.get("CONSULT_REVIEW_DIGEST_MIN_CHARS", "100")))
    max_focus = max(min_focus + 80, int(os.environ.get("CONSULT_REVIEW_DIGEST_MAX_CHARS", "2000")))

    excerpt = full_text[: min(len(full_text), dig_in_lim)]

    heuristic = _consult_heuristic_focus_text(excerpt)
    gemini_focus = None
    gmeta: dict = {"ok": False}
    if not _consult_heuristic_digest_first(min_focus, heuristic):
        gemini_focus, gmeta = _consult_gemini_clinical_focus_text(model, excerpt)
    else:
        gmeta["reason"] = "fast_mode" if _consult_review_fast_mode() else "heuristic_sufficient"
    meta["gemini_digest"] = gmeta

    focus_plain = ""
    if gemini_focus:
        gf = gemini_focus.strip()
        cf = str(gmeta.get("confidence") or "").lower()
        too_short = len(gf) < min_focus
        low_conf_short = cf == "low" and len(gf) < max(240, min_focus * 2)
        if not too_short and not low_conf_short:
            focus_plain = gf
            meta["focus_source"] = "gemini_digest"

    if not focus_plain.strip():
        h = heuristic.strip()
        if len(h) >= min_focus:
            focus_plain = h
            meta["focus_source"] = "heuristic_sections"
        else:
            focus_plain = excerpt[: min(len(excerpt), q_slice)].strip()
            meta["focus_source"] = "truncated_pdf"

    if len(focus_plain) > max_focus:
        focus_plain = focus_plain[: max_focus - 1].rstrip() + "…"

    meta["focus_chars"] = len(focus_plain)
    meta["focus_preview"] = (focus_plain[:420] + "…") if len(focus_plain) > 420 else focus_plain
    synthetic = sep + "\n\n" + focus_plain
    return synthetic, meta


def extract_pdf_text_from_bytes(data: bytes) -> tuple[str, list[str]]:
    """Извлечение текстового слоя PDF (без OCR). pypdf → PyMuPDF."""
    from clinical_knowledge.text_extract import extract_pdf_text_bytes

    max_pages = env_int("CONSULT_REVIEW_MAX_PAGES", 200)
    txt, warnings, err = extract_pdf_text_bytes(data, max_pages=max_pages)
    if err == "encrypted":
        raise HTTPException(
            status_code=400,
            detail="PDF защищён паролем - загрузите незашифрованную копию",
        )
    if err == "unreadable":
        raise HTTPException(
            status_code=400,
            detail="Файл не читается как PDF",
        )
    return txt, warnings


# Расширения файлов КЗ для consult-review (текстовые и PDF).
_CONSULT_PDF_EXTENSIONS = frozenset({".pdf"})
_CONSULT_PLAIN_EXTENSIONS = frozenset(
    {".txt", ".text", ".md", ".markdown", ".log", ".csv", ".json", ".xml"}
)
_CONSULT_HTML_EXTENSIONS = frozenset({".html", ".htm"})
_CONSULT_RTF_EXTENSIONS = frozenset({".rtf"})
_CONSULT_DOCX_EXTENSIONS = frozenset({".docx"})
_CONSULT_ODT_EXTENSIONS = frozenset({".odt"})
_CONSULT_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff"}
)

CONSULT_REVIEW_ALLOWED_EXTENSIONS = frozenset(
    _CONSULT_PDF_EXTENSIONS
    | _CONSULT_PLAIN_EXTENSIONS
    | _CONSULT_HTML_EXTENSIONS
    | _CONSULT_RTF_EXTENSIONS
    | _CONSULT_DOCX_EXTENSIONS
    | _CONSULT_ODT_EXTENSIONS
    | _CONSULT_IMAGE_EXTENSIONS
)


def consult_review_allowed_extensions() -> tuple[str, ...]:
    """Отсортированный список допустимых расширений для загрузки КЗ."""
    return tuple(sorted(CONSULT_REVIEW_ALLOWED_EXTENSIONS))


def consult_review_file_accept_attr() -> str:
    """Значение атрибута accept для input[type=file] (B2B и B2C)."""
    mime = (
        "image/*",
        "application/pdf",
        "text/plain",
        "text/html",
        "text/rtf",
        "application/rtf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.oasis.opendocument.text",
    )
    parts: list[str] = list(mime) + list(consult_review_allowed_extensions())
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return ",".join(out)


def consult_review_formats_hint_ru(*, max_files: int = 1) -> str:
    """Краткая подсказка для UI загрузки документов."""
    if max_files <= 1:
        return (
            "Один файл после приёма: КЗ, медицинский осмотр или консультация "
            "(PDF, Word DOCX, ODT, RTF, HTML, TXT/MD или чёткое фото JPG/PNG/HEIC). "
            "Несколько страниц - в одном PDF."
        )
    return (
        f"КЗ, медосмотр или консультация: PDF, Word (DOCX), ODT, RTF, HTML, текст (TXT, MD), "
        f"фото (JPG, PNG, HEIC) - до {max_files} файлов"
    )


def _consult_extension(filename: str) -> str:
    name = (filename or "").strip().lower()
    dot = name.rfind(".")
    if dot < 0:
        return ""
    return name[dot:]


def _decode_text_bytes(data: bytes) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if not data:
        return "", warnings
    for enc in ("utf-8-sig", "utf-16", "cp1251", "latin-1"):
        try:
            return data.decode(enc).strip(), warnings
        except UnicodeDecodeError:
            continue
    warnings.append("Кодировка не определена - использованы замены символов (UTF-8).")
    return data.decode("utf-8", errors="replace").strip(), warnings


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        chunk = (data or "").strip()
        if chunk:
            self._parts.append(chunk)

    def text(self) -> str:
        return "\n".join(self._parts).strip()


def _extract_html_text(data: bytes) -> tuple[str, list[str]]:
    raw, warns = _decode_text_bytes(data)
    parser = _HTMLTextExtractor()
    try:
        parser.feed(raw)
        parser.close()
    except Exception as e:
        warns.append(f"HTML: упрощённое извлечение ({e!s})")
        return raw, warns
    txt = parser.text()
    if not txt:
        return raw, warns
    return txt, warns


def _extract_rtf_text(data: bytes) -> tuple[str, list[str]]:
    raw, warns = _decode_text_bytes(data)
    if not raw.lstrip().startswith("{\\rtf"):
        warns.append("Файл не начинается с {\\rtf - попытка извлечь как текст.")
    text = raw
    text = re.sub(r"\\par[d]?\b", "\n", text, flags=re.I)
    text = re.sub(r"\\line\b", "\n", text, flags=re.I)
    text = re.sub(r"\\tab\b", "\t", text, flags=re.I)
    text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", text)
    text = re.sub(r"\\[a-z]+-?\d*\s?", "", text, flags=re.I)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), warns


def _extract_docx_text(data: bytes) -> tuple[str, list[str]]:
    warnings: list[str] = []
    from clinical_knowledge.text_extract import strip_file_prefix

    w_ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    payload = strip_file_prefix(data)
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            if "word/document.xml" not in zf.namelist():
                raise ValueError("нет word/document.xml")
            xml = zf.read("word/document.xml")
    except zipfile.BadZipFile as e:
        raise HTTPException(status_code=400, detail=f"DOCX не читается как ZIP: {e!s}") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"DOCX: {e!s}") from e
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise HTTPException(status_code=400, detail=f"DOCX: повреждён XML ({e!s})") from e
    paras: list[str] = []
    for p in root.iter(f"{w_ns}p"):
        parts = [(n.text or "") for n in p.iter(f"{w_ns}t")]
        line = "".join(parts).strip()
        if line:
            paras.append(line)
    full = "\n\n".join(paras).strip()
    if not full:
        warnings.append("DOCX: текст не найден в word/document.xml")
    return full, warnings


def _extract_odt_text(data: bytes) -> tuple[str, list[str]]:
    warnings: list[str] = []
    text_ns = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            if "content.xml" not in zf.namelist():
                raise ValueError("нет content.xml")
            xml = zf.read("content.xml")
    except zipfile.BadZipFile as e:
        raise HTTPException(status_code=400, detail=f"ODT не читается как ZIP: {e!s}") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"ODT: {e!s}") from e
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise HTTPException(status_code=400, detail=f"ODT: повреждён XML ({e!s})") from e
    paras: list[str] = []
    for p in root.iter(f"{text_ns}p"):
        parts = [(n.text or "") for n in p.iter(f"{text_ns}span")]
        if not parts:
            parts = [(n.text or "") for n in p.iter()]
        line = "".join(parts).strip()
        if line:
            paras.append(line)
    full = "\n\n".join(paras).strip()
    if not full:
        warnings.append("ODT: текст не найден в content.xml")
    return full, warnings


def _normalize_pdf_bytes(data: bytes, *, max_scan: int = 8192) -> bytes | None:
    from clinical_knowledge.text_extract import normalize_pdf_bytes

    return normalize_pdf_bytes(data, max_scan=max_scan)


def _is_zip_payload(data: bytes) -> bool:
    from clinical_knowledge.text_extract import is_zip_payload

    return is_zip_payload(data)


def _sniff_consult_format(data: bytes, ext: str) -> str:
    from clinical_knowledge.image_ocr import sniff_image_payload

    if ext in _CONSULT_IMAGE_EXTENSIONS or sniff_image_payload(data):
        return "image"
    if _normalize_pdf_bytes(data) is not None:
        return "pdf"
    stripped = data.lstrip(b"\x00 \t\r\n\xef\xbb\xbf\xfe\xff")
    if stripped[:2] == b"PK":
        if ext in _CONSULT_ODT_EXTENSIONS:
            return "odt"
        return "docx"
    if ext in _CONSULT_HTML_EXTENSIONS:
        return "html"
    low = data[:256].lstrip().lower()
    if low.startswith(b"{\\rtf") or ext in _CONSULT_RTF_EXTENSIONS:
        return "rtf"
    if ext in _CONSULT_ODT_EXTENSIONS:
        return "odt"
    if ext in _CONSULT_DOCX_EXTENSIONS:
        return "docx"
    return "plain"


def extract_consult_text_from_bytes(data: bytes, filename: str = "") -> tuple[str, list[str]]:
    """Извлечь текст документа после приёма (КЗ / медосмотр / консультация) из PDF или текстового файла."""
    ext = _consult_extension(filename)
    if ext and ext not in CONSULT_REVIEW_ALLOWED_EXTENSIONS:
        allowed = ", ".join(consult_review_allowed_extensions())
        raise HTTPException(
            status_code=400,
            detail=f"Файл «{filename}»: формат «{ext}» не поддерживается. Допустимо: {allowed}.",
        )
    if not ext:
        fmt = _sniff_consult_format(data, "")
        if fmt == "image":
            ext = ".jpg"
        elif fmt == "pdf":
            ext = ".pdf"
        elif fmt == "docx":
            ext = ".docx"
        elif fmt == "odt":
            ext = ".odt"
        elif fmt == "rtf":
            ext = ".rtf"
        else:
            ext = ".txt"

    fmt = _sniff_consult_format(data, ext)
    if fmt == "image":
        from clinical_knowledge.image_ocr import ocr_image_bytes

        txt, warns = ocr_image_bytes(data)
        if not txt.strip():
            hint = warns[0] if warns else "Переснимите при хорошем свете или загрузите PDF."
            raise HTTPException(
                status_code=400,
                detail=f"Не удалось распознать текст на фото «{filename or 'image'}». {hint}",
            )
        return txt, warns
    if fmt == "pdf":
        if _is_zip_payload(data) and _normalize_pdf_bytes(data) is None:
            return _extract_docx_text(data)
        pdf_data = _normalize_pdf_bytes(data)
        extra_warns: list[str] = []
        if pdf_data is None:
            pdf_data = data
            extra_warns.append(
                "PDF: маркер %PDF- не найден - попытка чтения как есть."
            )
        elif pdf_data is not data and not data.lstrip().startswith(b"%PDF-"):
            extra_warns.append("PDF: пропущен служебный префикс до маркера %PDF-.")
        try:
            txt, warns = extract_pdf_text_from_bytes(pdf_data)
        except HTTPException:
            if _is_zip_payload(data):
                return _extract_docx_text(data)
            raise
        if not txt.strip() and _is_zip_payload(data):
            return _extract_docx_text(data)
        return txt, extra_warns + warns
    if fmt == "docx":
        return _extract_docx_text(data)
    if fmt == "odt":
        return _extract_odt_text(data)
    if fmt == "rtf":
        return _extract_rtf_text(data)
    if fmt == "html":
        return _extract_html_text(data)
    return _decode_text_bytes(data)



def _retrieval_fragment_body(row: dict) -> str:
    """Видимый текст фрагмента из результата retrieve(): в ответах API поле excerpt, не text."""
    if not isinstance(row, dict):
        return ""
    txt = row.get("text")
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    exc = row.get("excerpt")
    if isinstance(exc, str) and exc.strip():
        return exc.strip()
    return ""


def _build_review_chunks_context(
    retrieved: list[dict], max_chars: int
) -> tuple[str, list[str]]:
    """Склеивает топ-чанки retrieve() для промпта сравнения."""
    rich = env_bool("CONSULT_REVIEW_RICH_CONTEXT", True)
    lines: list[str] = []
    paths_order: list[str] = []
    seen: set[str] = set()
    n = 0
    for r in retrieved:
        p = (r.get("path") or "").strip()
        if p and p not in seen:
            seen.add(p)
            paths_order.append(p)
        txt = _retrieval_fragment_body(r)
        if not txt:
            continue
        kind = (r.get("kind") or "").strip()
        header = f"path={p}\ntype={kind}"
        if rich:
            sec = (r.get("section_title") or "").strip()
            if sec:
                header += f"\nsection={sec}"
            pf = int(r.get("page_from") or 0)
            if pf:
                pt = int(r.get("page_to") or 0) or pf
                header += f"\npages={pf}" + (f"-{pt}" if pt != pf else "")
            pts = r.get("point_numbers")
            if isinstance(pts, list) and pts:
                header += "\nпункты=" + ", ".join(str(x) for x in pts[:8])
        block = f"{header}\n{txt}\n"
        if n + len(block) > max_chars:
            rest = max_chars - n
            if rest > 120:
                lines.append(block[:rest])
            break
        lines.append(block)
        n += len(block)
    return "\n---\n".join(lines), paths_order


def _consult_review_synthesize(
    model,
    consultation_excerpt: str,
    protocol_excerpt: str,
    paths_hint: list[str],
    extra_context: str = "",
    clinical_rules_context: str = "",
) -> dict:
    paths_line = ", ".join(paths_hint[:12]) if paths_hint else "(не определены)"
    rules_block = ""
    if clinical_rules_context and clinical_rules_context.strip():
        rules_block = "\n\n--- " + clinical_rules_context.strip() + "\n"
    full_prompt = (
        SYSTEM_CONSULT_REVIEW_JSON
        + "\n\nОжидаемые path протоколов (подсказка): "
        + paths_line
        + rules_block
        + "\n\n--- ТЕКСТ ЗАКЛЮЧЕНИЯ (фрагмент из PDF) ---\n\n"
        + consultation_excerpt
        + "\n\n--- ВЫДЕРЖКИ ПРОТОКОЛОВ (RAG) ---\n\n"
        + protocol_excerpt
        + (extra_context.strip() + "\n" if extra_context and extra_context.strip() else "")
    )
    base_tokens = _consult_review_synth_max_tokens()
    resp = generate_gemini_consult_review_synthesize(model, full_prompt, max_out=base_tokens)
    txt = _extract_gemini_text(resp)
    parsed = _try_parse_json(txt)

    # Если ответ обрезан по лимиту токенов или JSON не распарсился - повтор с большим лимитом.
    if (
        (parsed is None or _finish_hits_max(resp))
        and env_bool("CONSULT_REVIEW_SYNTH_RETRY", True)
        and not _consult_review_fast_mode()
    ):
        retry_tokens = min(32768, base_tokens * 2)
        try:
            resp2 = generate_gemini_consult_review_synthesize(
                model, full_prompt, max_out=retry_tokens
            )
            txt2 = _extract_gemini_text(resp2)
            parsed2 = _try_parse_json(txt2)
            if parsed2 is not None:
                parsed = parsed2
        except HTTPException:
            if parsed is None:
                raise

    if not parsed:
        raise HTTPException(
            status_code=502,
            detail=(
                "Модель не вернула корректный JSON (возможно, объём заключения слишком большой). "
                "Повторите попытку или сократите PDF."
            ),
        )
    _stabilize_overall_compliance(parsed)
    return parsed


def _consult_l2_narrative_synthesize(
    *,
    evidence_pack: dict[str, Any],
    block_gaps: list[dict[str, Any]] | None = None,
    structured_summary: str = "",
) -> str:
    """Один Flash-вызов: пояснение методисту по evidence pack (без полного текста КЗ)."""
    import json as _json

    model = get_gemini()
    gaps_txt = ""
    for g in (block_gaps or [])[:6]:
        if isinstance(g, dict):
            line = str(g.get("gap_ru") or "").strip()
            if line:
                gaps_txt += f"- {line}\n"
    ev_blob = _json.dumps(evidence_pack, ensure_ascii=False)[:5500]
    summary_bit = (structured_summary or "").strip()[:800]
    full_prompt = (
        SYSTEM_CONSULT_L2_NARRATIVE
        + "\n\n--- EVIDENCE PACK (JSON) ---\n"
        + ev_blob
        + "\n\n--- ПРОБЕЛЫ СВЕРКИ ---\n"
        + (gaps_txt or "(нет явных пробелов)\n")
        + ("\n--- КРАТКИЙ СТРУКТУРНЫЙ КОНТЕКСТ ---\n" + summary_bit + "\n" if summary_bit else "")
    )
    resp = generate_gemini_consult_review_synthesize(model, full_prompt, max_out=512)
    txt = (_extract_gemini_text(resp) or "").strip()
    if not txt:
        raise HTTPException(status_code=502, detail="Модель не вернула пояснение L2+")
    return txt[:2000]


def _stabilize_overall_compliance(parsed: dict) -> None:
    """Итоговый процент - детерминированная функция баллов критериев, а не отдельное число модели.

    Делает «Ориентировочное соответствие» прозрачным и воспроизводимым: при одинаковых критериях
    итог всегда одинаков. Отключается CONSULT_REVIEW_OVERALL_FROM_CRITERIA=0.
    """
    if not isinstance(parsed, dict):
        return
    if not env_bool("CONSULT_REVIEW_OVERALL_FROM_CRITERIA", True):
        return
    crits = parsed.get("criteria")
    if not isinstance(crits, list):
        return
    scores: list[float] = []
    for c in crits:
        if not isinstance(c, dict):
            continue
        v = c.get("score_pct")
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if 0.0 <= fv <= 100.0:
            scores.append(fv)
    if not scores:
        return
    parsed["overall_compliance_pct"] = int(round(sum(scores) / len(scores)))
    parsed["overall_compliance_method"] = "mean_of_criteria"


def _consult_ui_protocol_fragments(
    retrieved: list[dict], paths_used: list[str]
) -> list[dict]:
    """Фрагменты текста протоколов по отбору RAG для ссылок клиента с подсветкой (выдержки)."""
    max_frag_chars = int(os.environ.get("CONSULT_REVIEW_UI_FRAG_CHARS", "1400"))
    max_frags = int(os.environ.get("CONSULT_REVIEW_UI_FRAGS_PER_PATH", "4"))
    allow = set(paths_used)
    counts: dict[str, int] = {}
    buckets: dict[str, list[dict[str, str]]] = {}

    def add_frag(pth: str, kind: str, text: str, section: str, pages: str) -> None:
        if pth not in allow:
            return
        n = counts.get(pth, 0)
        if n >= max_frags:
            return
        t = text.strip()
        if not t:
            return
        if len(t) > max_frag_chars:
            t = t[: max_frag_chars - 1].rstrip() + "…"
        frag: dict[str, str] = {
            "kind": kind.strip() if kind.strip() else "fragment",
            "text": t,
        }
        if section:
            frag["section"] = section
        if pages:
            frag["pages"] = pages
        buckets.setdefault(pth, []).append(frag)
        counts[pth] = n + 1

    def _pages_label(row: dict) -> str:
        pf = int(row.get("page_from") or 0)
        if not pf:
            return ""
        pt = int(row.get("page_to") or 0) or pf
        return str(pf) if pt == pf else f"{pf}-{pt}"

    for row in retrieved:
        pth = str(row.get("path") or "").strip()
        if not pth:
            continue
        txt = _retrieval_fragment_body(row)
        if not txt:
            continue
        kd = row.get("kind")
        kind = kd.strip() if isinstance(kd, str) else "fragment"
        section = str(row.get("section_title") or "").strip()
        add_frag(pth, kind, txt, section, _pages_label(row))

    out: list[dict] = []
    for pth in paths_used:
        frags = buckets.get(pth) or []
        pr = _protocols_by_path.get(pth) or {}
        ttl = _protocol_display_title(pth, str(pr.get("title") or "").strip() or None)
        out.append({"path": pth, "title": ttl, "fragments": frags})
    return out


def _consult_onco_scan_source() -> str:
    """kz_only - только текст КЗ; legacy - также рубрика novoobrazovaniya в отборе протоколов."""
    raw = (os.environ.get("CONSULT_ONCO_SCAN_SOURCE") or "kz_only").strip().lower()
    return raw if raw in ("kz_only", "legacy") else "kz_only"


def _consult_oncology_dual_scan(blob: str) -> tuple[list[str], list[str]]:
    """Сильные маркеры (достаточно одного) и слабые (скрининг/общие - по порогу количества)."""
    nb = (_norm_query(blob or "")).strip()
    if not nb:
        return [], []
    strong: set[str] = set()
    weak: set[str] = set()

    for s in (
        "метастаз",
        "полихимиотерап",
        "химиотерапия",
        "химиотерапию",
        "химиотерапией",
        "семейная нагруженность по онколог",
        "инвазивная карцино",
        "инвазивной карцино",
        "рецидив опухоли",
        "злокачественн",
        "стереотакс",
        "лучевая терап",
        "стационар по онколог",
        "онкохирург",
        "онкоцентр",
        "онкодиагност",
    ):
        if s in nb:
            strong.add(s)

    for s in ("онкология", "онкологии", "онкологией", "онкологическ", "онкологию"):
        if s in nb:
            strong.add("онкология (форма слова)")

    for s in ("онколога", "онкологу", "онкологом"):
        if s in nb:
            strong.add("онколог (специалист)")

    for w in (
        "онкомаркер",
        "онкоцитолог",
        "патоморфолог",
        "патолого-анатом",
        "патологоанатом",
        "биопсия",
        " биопс",
        "гистологическ",
        "гистологию",
        "опухоль",
        "опухоли",
        "опухолью",
        "новообразован",
        "онкоскрин",
        "онкоскр",
    ):
        w = w.strip()
        if w and w in nb:
            weak.add(w)

    rxs: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"(?:^|[^а-яё])лимфом(?!енинго)", re.I), "лимфома"),
        (re.compile(r"(?:^|[^а-яё])миелом(?!енинго)", re.I), "миелома"),
        (
            re.compile(
                r"\b(?:карцином\w*|аденокарцином\w*|холангиокарцин\w*|сарком\w*|меланом\w*)",
                re.I,
            ),
            "карцинома/саркома/меланома",
        ),
        (re.compile(r"\bher2\b|\bгер\s*[\-\u2013\u2014]?\s*2\b|\bгер2\b", re.I), "HER2"),
        (re.compile(r"\bki\s*[\-\u2013\u2014]?\s*67\b|\bki67\b", re.I), "Ki-67"),
        (re.compile(r"\btnm\b", re.I), "TNM"),
        (re.compile(r"\bstaging\b", re.I), "staging"),
        (re.compile(r"\boncolog\w+", re.I), "oncology_en"),
        (re.compile(r"\b(?:лейкоз\w*|лейкем)", re.I), "лейкоз"),
        (re.compile(r"(?:^|[^а-яё])раху\b|(?:^|[^а-яё])рахом\b|\bрак\b", re.I), "рак"),
    )
    for rgx, lab in rxs:
        if rgx.search(nb):
            strong.add(lab)

    return sorted(strong), sorted(weak)


def _consult_oncology_flags(
    ui_frags: list[dict], consultation_text_raw: str
) -> dict:
    """Признаки онкологии/повышенного онкологического риска в заключении и в отборе протоколов."""
    weak_min = max(1, min(9, int(os.environ.get("CONSULT_REVIEW_ONCOLOGY_WEAK_MIN_HITS", "2"))))

    def needle_basis_sentence(where_human: str, found: list[str]) -> str:
        if not found:
            return ""
        uniq = sorted(set(found))
        shown = uniq[:14]
        more = len(uniq) - len(shown)
        tail = f" (+ ещё {more})" if more > 0 else ""
        joined = ", ".join(shown)
        return (
            f"{where_human}: эвристика с сильными и слабыми маркерами; признаки: {joined}{tail}. "
            f"Слабые скрининговые признаки учитываются при сумме не менее {weak_min}."
        )

    cons = consultation_text_raw or ""
    cons_strong, cons_weak = _consult_oncology_dual_scan(cons)
    cons_hits = sorted(set(cons_strong + cons_weak))
    in_consult = bool(cons_strong) or len(cons_weak) >= weak_min

    prot_items: list[dict] = []
    seen_paths: set[str] = set()
    for row in ui_frags or []:
        if not isinstance(row, dict):
            continue
        pth = str(row.get("path") or "").strip()
        if not pth or pth in seen_paths:
            continue
        path_l = pth.lower()
        title_raw = row.get("title") or ""
        basis_lines: list[str] = []
        pr_meta = _protocols_by_path.get(pth) or {}
        cat = str(pr_meta.get("category") or "").lower()
        in_onco_catalog = "novoobrazovan" in path_l or "novoobrazovan" in cat
        if in_onco_catalog:
            basis_lines.append(
                'Каталог протоколов: файл отнесён к ветви «новообразования» - в URL пути или в '
                "поле category присутствует «novoobrazovan»."
            )
        if not basis_lines:
            continue
        seen_paths.add(pth)
        prot_items.append(
            {
                "path": pth,
                "title": _protocol_display_title(pth, str(title_raw or "").strip() or None),
                "basis_ru": list(basis_lines),
                # Совместимость со старым фронтом: то же содержание, но конкретнее.
                "hints_ru": list(basis_lines),
                "markers": [],
            }
        )

    in_proto = len(prot_items) > 0
    kz_only = _consult_onco_scan_source() == "kz_only"
    banner_proto = in_proto and not kz_only

    sentences: list[str] = []
    if in_consult:
        sentences.append(
            "В тексте загруженного заключения обнаружены формулировки, которые могут относиться к опухоли, её подозрению, наблюдению после лечения или онкологическому контролю."
        )
    if banner_proto:
        sentences.append(
            "В числе отобранных протоколов есть источники из рубрики «новообразования» каталога - особенно внимательно сверьте заключение с ними."
        )

    banner = " ".join(sentences) if sentences else ""
    instruction_ru = (
        banner
        if banner
        else "При анализе учитывай возможное сочетание общетерапевтического протокола с онкологическим контекстом заключения."
    )

    consultation_basis_ru: list[str] = []
    if in_consult and cons_hits:
        consultation_basis_ru.append(
            needle_basis_sentence(
                "Текст загруженного заключения (извлечённый из PDF)",
                cons_hits,
            )
        )

    basis_for_prompt: list[str] = []
    basis_for_prompt.extend(consultation_basis_ru)
    for pi in prot_items:
        t_raw = str(pi.get("title") or pi.get("path") or "")
        for sentence in pi.get("basis_ru") or []:
            basis_for_prompt.append(f"[{t_raw}] {sentence}")

    if basis_for_prompt:
        clipped = " ".join(basis_for_prompt)
        if len(clipped) > 1400:
            clipped = clipped[:1397].rstrip() + "…"
        instruction_ru = f"{instruction_ru} Основание (эвристика): {clipped}"

    scan_note = (
        "Сканируется только текст заключения."
        if kz_only
        else "Учитываются текст заключения и рубрика «новообразования» в каталоге протоколов."
    )
    method_note_ru = (
        "Это не клиническое решение модели про «онко-риск»: учитываются сильные маркеры (злокачественность, "
        "химиотерапия, типичные паттерны рака/миеломы/лимфомы с отсечением ложных вроде «лимфоменингит», и т.д.) "
        f"или не менее {weak_min} слабых скрининговых; совпадение по одному нейтральному слову недостаточно. "
        f"{scan_note}"
    )

    return {
        "any": in_consult or banner_proto,
        "consultation_hit": in_consult,
        "protocol_hit": in_proto,
        "scan_source": _consult_onco_scan_source(),
        "consultation_markers": cons_hits[:16],
        "consultation_strong_markers": list(cons_strong),
        "consultation_weak_markers": list(cons_weak),
        "consultation_basis_ru": consultation_basis_ru,
        "protocol_items": prot_items,
        "banner_ru": banner,
        "instruction_ru": instruction_ru,
        "method_note_ru": method_note_ru,
    }


def _icd_client_payload(icd_analysis: dict) -> dict:
    """Единый JSON для API и llm_json.icd_codes."""
    detected: list[dict] = []
    for d in icd_analysis.get("detected") or []:
        if not isinstance(d, dict):
            continue
        detected.append(
            {
                "code": d.get("code"),
                "title_ru": d.get("title_ru"),
                "title_en": d.get("title_en"),
                "role": "detected_in_query",
            }
        )
    suggested: list[dict] = []
    for s in icd_analysis.get("suggested") or []:
        if not isinstance(s, dict):
            continue
        role = (
            "suggested_gemini"
            if s.get("match_method") == "gemini_from_pool"
            else "suggested_lexicon"
        )
        row = {
            "code": s.get("code"),
            "title_ru": s.get("title_ru"),
            "title_en": s.get("title_en"),
            "role": role,
            "score": s.get("score"),
        }
        if role == "suggested_gemini" and s.get("rationale"):
            row["rationale"] = s.get("rationale")
        suggested.append(row)
    du_list: list[dict] = []
    for x in icd_analysis.get("detected_unknown") or []:
        if not isinstance(x, dict):
            continue
        du_list.append(
            {
                "code": x.get("code"),
                "title_ru": x.get("title_ru"),
                "title_en": x.get("title_en"),
            }
        )
    out: dict = {
        "detected": detected,
        "suggested": suggested,
        "codes_for_retrieval": icd_analysis.get("codes_for_retrieval") or [],
        "explicit_icd_in_query": bool(icd_analysis.get("explicit_icd_in_query")),
        "detected_unknown": du_list,
    }
    meta = icd_analysis.get("icd_meta")
    if meta:
        out["meta"] = meta
    return out


def _icd_block_for_prompt(icd_analysis: dict) -> str:
    lines: list[str] = []
    for d in icd_analysis.get("detected_unknown") or []:
        if not isinstance(d, dict):
            continue
        c = d.get("code") or ""
        lines.append(
            f"- {c}: код не найден в справочнике МКБ-10 (Excel→JSON); проверьте написание."
        )
    for d in icd_analysis.get("detected") or []:
        if not isinstance(d, dict):
            continue
        c = d.get("code") or ""
        tr = (d.get("title_ru") or "").strip()
        ten = (d.get("title_en") or "").strip()
        if tr and ten:
            lines.append(f"- {c}: {tr} ({ten})")
        elif tr:
            lines.append(f"- {c}: {tr}")
        elif ten:
            lines.append(f"- {c}: ({ten})")
        else:
            lines.append(f"- {c}")
    for s in icd_analysis.get("suggested") or []:
        if not isinstance(s, dict):
            continue
        c = s.get("code") or ""
        tr = (s.get("title_ru") or "").strip()
        ten = (s.get("title_en") or "").strip()
        sc = s.get("score")
        if s.get("match_method") == "gemini_from_pool":
            rat = (s.get("rationale") or "").strip()
            tail = " [подбор из пула кандидатов]"
            if rat:
                tail = f" [подбор: {rat[:160]}]"
        elif sc is not None:
            tail = f" [лексикон, score={sc}]"
        else:
            tail = " [лексикон]"
        if tr and ten:
            lines.append(f"- {c}: {tr} ({ten}){tail}")
        elif tr:
            lines.append(f"- {c}: {tr}{tail}")
        else:
            lines.append(f"- {c}{tail}")
    if not lines:
        return ""
    return (
        "=== Сопоставление МКБ-10 (автоматически, справочно) ===\n"
        + "Не выдумывай коды вне этого списка. При кратком summary можно упомянуть релевантные коды из списка.\n"
        + "\n".join(lines)
    )


def _diagnostic_mode_summary(icd_payload: dict, retrieved: list[dict]) -> dict:
    explicit = bool(icd_payload.get("explicit_icd_in_query"))
    detected = icd_payload.get("detected") or []
    suggested = icd_payload.get("suggested") or []
    top_score = 0.0
    if retrieved:
        try:
            top_score = float(retrieved[0].get("score") or 0.0)
        except (TypeError, ValueError):
            top_score = 0.0
    top_score = max(0.0, min(1.0, top_score))
    if explicit or detected:
        mode = "diagnosis_or_icd"
        conf = max(0.72, min(0.98, 0.78 + top_score * 0.2))
        notice = (
            "Подбор выполнен с опорой на диагноз/код МКБ-10; "
            "соответствие обычно выше, но всё равно сверяйте с полным текстом протокола."
        )
    elif suggested:
        mode = "symptom_inferred"
        conf = max(0.45, min(0.86, 0.52 + top_score * 0.26))
        notice = (
            "Точный диагноз/код МКБ-10 не указан. Сервис использовал симптомный поиск и "
            "предположительное сопоставление с МКБ-10; результаты ориентировочные."
        )
    else:
        mode = "symptom_only"
        conf = max(0.3, min(0.74, 0.38 + top_score * 0.18))
        notice = (
            "Диагноз/код МКБ-10 не определён. Протоколы подобраны по симптомам; "
            "точность ограничена, рекомендуется уточнить клинические детали."
        )
    return {
        "mode": mode,
        "confidence": round(float(conf), 4),
        "notice": notice,
    }


def _ensure_symptom_followup_questions(parsed: dict | None, diag_mode: str, conf: float) -> None:
    if not parsed or not isinstance(parsed, dict):
        return
    if diag_mode not in ("symptom_inferred", "symptom_only"):
        return
    if conf >= 0.62:
        return
    existing = parsed.get("questions_for_patient")
    questions: list[str] = []
    if isinstance(existing, list):
        for q in existing:
            s = str(q).strip()
            if s:
                questions.append(s)
    extra = [
        "Какова длительность симптомов и динамика ухудшения за последние 24-72 часа?",
        "Есть ли объективные показатели: температура, SpO2, АД, ЧСС или другие измерения?",
        "Какие симптомы тревоги присутствуют сейчас (одышка в покое, боль в груди, выраженная слабость, нарушение сознания)?",
    ]
    seen = set(questions)
    for e in extra:
        if e not in seen:
            questions.append(e)
            seen.add(e)
        if len(questions) >= 4:
            break
    parsed["questions_for_patient"] = questions[:4]


def _assist_lite_enabled(*, assist_full: bool = False) -> bool:
    if assist_full:
        return False
    return os.environ.get("RAG_ASSIST_LITE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _strip_assist_verbose_fields(parsed: dict | None) -> None:
    """Lite assist: только protocols + disclaimer в JSON для UI."""
    if not parsed or not isinstance(parsed, dict):
        return
    parsed.pop("summary", None)
    parsed.pop("differential", None)
    parsed.pop("questions_for_patient", None)


def _normalize_protocol_path_key(p: str) -> str:
    s = (p or "").strip()
    if not s:
        return ""
    try:
        if "%" in s:
            s = unquote(s)
    except Exception:
        pass
    s = s.replace("\\", "/")
    while "//" in s:
        s = s.replace("//", "/")
    return s


def _protocol_basename_key(p: str) -> str:
    s = _normalize_protocol_path_key(p)
    if not s:
        return ""
    return Path(s).name.lower()


def _path_category_slug(p: str) -> str:
    s = _normalize_protocol_path_key(p)
    parts = [x for x in s.split("/") if x]
    if "minzdrav_protocols" in parts:
        idx = parts.index("minzdrav_protocols")
        if idx + 1 < len(parts):
            return parts[idx + 1].lower()
    if len(parts) >= 2:
        return parts[-2].lower()
    return ""


_RUBRIC_PATH_PRIORITY = (
    "khirurgiya",
    "gastroenterologiya",
    "koloproktologiya",
    "proktologiya",
    "novoobrazovaniya",
    "dermatovenerologiya",
    "infektsionnye-zabolevaniya",
)


def _pick_better_protocol_entry(
    a: dict,
    b: dict,
    *,
    prefer_slugs: list[str] | None = None,
) -> dict:
    """Выбор канонической копии PDF при дублях filename в разных рубриках."""
    sc_a = _confidence_numeric(a.get("confidence_score")) or float(a.get("score") or 0.0)
    sc_b = _confidence_numeric(b.get("confidence_score")) or float(b.get("score") or 0.0)
    if sc_b > sc_a + 0.001:
        winner, loser = b, a
    elif sc_a > sc_b + 0.001:
        winner, loser = a, b
    else:
        slugs = {str(s).strip().lower() for s in (prefer_slugs or []) if str(s).strip()}
        pa = _path_category_slug(str(a.get("path") or ""))
        pb = _path_category_slug(str(b.get("path") or ""))
        if pb in slugs and pa not in slugs:
            winner, loser = b, a
        elif pa in slugs and pb not in slugs:
            winner, loser = a, b
        else:
            ia = next((i for i, s in enumerate(_RUBRIC_PATH_PRIORITY) if s == pa), 99)
            ib = next((i for i, s in enumerate(_RUBRIC_PATH_PRIORITY) if s == pb), 99)
            if ib < ia:
                winner, loser = b, a
            elif ia < ib:
                winner, loser = a, b
            else:
                winner = a if len(str(a.get("path") or "")) <= len(str(b.get("path") or "")) else b
                loser = b if winner is a else a
    dup = list(winner.get("duplicate_catalog_paths") or [])
    other = str(loser.get("path") or "").strip()
    if other and other not in dup:
        dup.append(other)
    if dup:
        winner = {**winner, "duplicate_catalog_paths": dup}
    return winner


def dedupe_protocols_list(
    protocols: list,
    *,
    prefer_slugs: list[str] | None = None,
) -> list:
    """Один PDF (basename) и один title - с максимальным confidence_score."""
    if not protocols:
        return []
    by_path: dict[str, dict] = {}
    for pr in protocols:
        if not isinstance(pr, dict):
            continue
        p = _normalize_protocol_path_key(str(pr.get("path") or ""))
        if not p:
            continue
        sc = _confidence_numeric(pr.get("confidence_score")) or 0.0
        prev = by_path.get(p)
        if prev is None:
            by_path[p] = pr
        else:
            psc = _confidence_numeric(prev.get("confidence_score")) or 0.0
            if sc > psc:
                by_path[p] = pr
    merged = list(by_path.values())
    by_title: dict[str, dict] = {}
    for pr in merged:
        tk = _normalize_protocol_title_key(str(pr.get("title") or ""))
        if not tk:
            tk = _normalize_protocol_path_key(str(pr.get("path") or ""))
        prev = by_title.get(tk)
        if prev is None:
            by_title[tk] = pr
        else:
            by_title[tk] = _pick_better_protocol_entry(prev, pr, prefer_slugs=prefer_slugs)
    out = list(by_title.values())
    by_base: dict[str, dict] = {}
    for pr in out:
        bk = _protocol_basename_key(str(pr.get("path") or ""))
        if not bk:
            by_base[f"__path__:{pr.get('path')}"] = pr
            continue
        prev = by_base.get(bk)
        if prev is None:
            by_base[bk] = pr
        else:
            by_base[bk] = _pick_better_protocol_entry(prev, pr, prefer_slugs=prefer_slugs)
    out = list(by_base.values())
    out.sort(
        key=lambda x: -(_confidence_numeric(x.get("confidence_score")) or 0.0)
    )
    return out


def _normalize_protocol_title_key(t: str) -> str:
    return " ".join((t or "").strip().lower().split())


def _retrieval_path_for_chunk(ch: dict) -> str:
    """PDF path для ranking/dedupe: summary-чанки → catalog_source_path."""
    catalog = str(ch.get("catalog_source_path") or "").strip()
    if catalog:
        return catalog.replace("\\", "/")
    return str(ch.get("path") or "").replace("\\", "/")


def _path_matches_allowlist(ch: dict, allow: frozenset[str]) -> bool:
    if not allow:
        return True
    pth = _retrieval_path_for_chunk(ch)
    if pth in allow:
        return True
    from clinical_knowledge.protocol_summary.icd_index import catalog_path_matches_chunk

    return catalog_path_matches_chunk(pth, allow)


def dedupe_retrieval_by_basename(
    rows: list[dict],
    *,
    prefer_slugs: list[str] | None = None,
) -> list[dict]:
    """Один фрагмент на уникальный PDF (basename) - убирает копии в разных рубриках."""
    if not rows:
        return []
    by_base: dict[str, dict] = {}
    order: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        p = _retrieval_path_for_chunk(row)
        bk = _protocol_basename_key(p) or f"__path__:{p}"
        prev = by_base.get(bk)
        if prev is None:
            by_base[bk] = dict(row)
            order.append(bk)
        else:
            by_base[bk] = _pick_better_protocol_entry(prev, row, prefer_slugs=prefer_slugs)
    return [by_base[k] for k in order if k in by_base]


def _build_protocols_from_retrieval(
    retrieved: list[dict],
    *,
    prefer_slugs: list[str] | None = None,
    icd_codes: list[str] | None = None,
) -> list[dict]:
    """Список протоколов только из RAG (без вызова LLM для ranking)."""
    from clinical_knowledge.rich_chunk_search import build_chunk_match_reason

    rows = dedupe_retrieval_by_basename(retrieved, prefer_slugs=prefer_slugs)
    if not rows:
        return []
    raw_scores: list[float] = []
    for row in rows:
        try:
            raw_scores.append(float(row.get("score") or row.get("lexical_score") or 0.0))
        except (TypeError, ValueError):
            raw_scores.append(0.0)
    max_sc = max(raw_scores) if raw_scores else 1.0
    if max_sc <= 0:
        max_sc = 1.0
    protos: list[dict] = []
    for row, raw in zip(rows, raw_scores):
        p = _normalize_protocol_path_key(_retrieval_path_for_chunk(row))
        if not p:
            continue
        meta = _protocols_by_path.get(p) or {}
        raw_title = str(meta.get("title") or p.rsplit("/", 1)[-1])
        title = _protocol_display_title(p, raw_title)
        conf = min(0.97, max(0.38, 0.35 + 0.62 * (raw / max_sc)))
        rag_sup = min(0.95, max(0.2, conf * 0.88))
        match_reason = row.get("match_reason") or build_chunk_match_reason(row, icd_codes)
        entry: dict = {
            "path": p,
            "title": title,
            "confidence_score": round(conf, 4),
            "rag_support": round(rag_sup, 4),
            "match_reason": match_reason,
        }
        if row.get("section_title"):
            entry["section_title"] = row.get("section_title")
        if row.get("page_from"):
            entry["page_from"] = row.get("page_from")
        if row.get("chunk_type"):
            entry["best_chunk_type"] = row.get("chunk_type")
        protos.append(entry)
    return dedupe_protocols_list(protos, prefer_slugs=prefer_slugs)


def build_hybrid_search_payload(
    *,
    query: str,
    icd_codes: list[str] | None,
    population: str | None,
    category_slugs: list[str] | None,
    icd_analysis: dict | None,
    lookup_result: dict | None,
) -> dict:
    """Гибрид ICD lookup + rich-chunk RAG для шага 4 воронки."""
    from clinical_knowledge.rich_chunk_search import (
        hybrid_merge_protocols,
        hybrid_pin_trusted_icd_top1,
        query_wants_tables,
    )

    lookup = lookup_result or {}
    icd_protos = list(lookup.get("protocols") or [])
    match_reasons = lookup.get("match_reasons") or {}
    for pr in icd_protos:
        p = str(pr.get("path") or "")
        reasons = match_reasons.get(p) or []
        if reasons and not pr.get("match_reason"):
            pr["match_reason"] = ", ".join(str(r) for r in reasons[:2])[:70]

    user_slugs = [
        s for s in (category_slugs or []) if isinstance(s, str) and s in ALLOWED_SPECIALTY_SLUGS
    ]
    icd_codes_for_lex = list(icd_codes or [])
    if icd_analysis and not icd_codes:
        icd_codes_for_lex = list(
            dict.fromkeys(
                icd_codes_for_lex
                + [str(c) for c in (icd_analysis.get("codes_for_retrieval") or []) if c]
            )
        )

    summary_first = env_bool("SEARCH_RAG_SUMMARY_FIRST", True)
    summary_paths: list[str] = []
    if summary_first and icd_codes_for_lex:
        try:
            from clinical_knowledge.protocol_summary.icd_index import find_catalog_paths_by_icd_codes

            summary_paths = find_catalog_paths_by_icd_codes(icd_codes_for_lex, limit=10) or []
            existing_paths = {str(p.get("path") or "").replace("\\", "/") for p in icd_protos}
            for sp in summary_paths:
                norm = sp.replace("\\", "/")
                if norm in existing_paths:
                    continue
                icd_protos.insert(
                    0,
                    {
                        "path": norm,
                        "match_reason": "По структурированной карточке протокола (МКБ)",
                        "from_summary_index": True,
                    },
                )
                existing_paths.add(norm)
        except Exception:
            summary_paths = []

    path_allowlist = lookup.get("path_allowlist") or None
    if not path_allowlist and icd_protos:
        path_allowlist = [str(p.get("path") or "") for p in icd_protos if p.get("path")][:15]

    aud_hint = infer_audience_from_funnel_context(query)
    if aud_hint is None:
        aud_hint = infer_audience_from_query(query, _routing)
    if population in ("adult", "pediatric", "child"):
        aud_hint = "child" if population in ("pediatric", "child") else "adult"
    elif population == "pregnant":
        aud_hint = "pregnant"
    elif population == "emergency":
        ql_em = (query or "").lower()
        if any(
            x in ql_em
            for x in ("ребен", "ребён", "детск", "новорожд", "грудн", "младен", "педиатр")
        ):
            aud_hint = "child"
        else:
            aud_hint = "adult"

    path_boost: list[str] | None = path_allowlist
    if icd_codes_for_lex:
        try:
            from clinical_knowledge.protocol_summary.icd_index import find_catalog_paths_by_icd_codes

            extra = find_catalog_paths_by_icd_codes(icd_codes_for_lex, limit=8) or []
            if summary_first and summary_paths:
                extra = list(dict.fromkeys(summary_paths + extra))
            path_boost = list(dict.fromkeys((path_boost or []) + extra)) or None
        except Exception:
            pass

    retrieved = retrieve(
        query,
        routing_query=query,
        user_category_slugs=user_slugs or None,
        icd_codes_for_lex=icd_codes_for_lex or None,
        path_boost=path_boost,
        path_allowlist=path_allowlist,
        audience_hint=aud_hint,
        max_chunks=int(os.environ.get("RAG_HYBRID_MAX_CHUNKS", "12")),
        max_per_path=int(os.environ.get("RAG_HYBRID_MAX_PER_PATH", "3")),
    )
    if not retrieved and path_allowlist:
        retrieved = retrieve(
            query,
            routing_query=query,
            user_category_slugs=user_slugs or None,
            icd_codes_for_lex=icd_codes_for_lex or None,
            path_boost=path_boost,
            path_allowlist=None,
            audience_hint=aud_hint,
            max_chunks=int(os.environ.get("RAG_HYBRID_MAX_CHUNKS", "12")),
        )
        retrieved, _, _ = filter_retrieval_by_audience(retrieved, query, _routing)

    table_rows: list[dict] = []
    if query_wants_tables(query) and retrieved:
        top_paths_norm = {
            str(tp or "").replace("\\", "/")
            for tp in (str(r.get("path") or "") for r in retrieved[:6])
            if tp
        }
        for tp in top_paths_norm:
            for ch in _chunks_by_path.get(tp) or []:
                kind = (ch.get("kind") or "").strip().lower()
                if kind != "table":
                    continue
                table_rows.append(
                    {
                        "path": tp,
                        "kind": "table",
                        "text": (ch.get("text") or "")[:900],
                        "page_from": ch.get("page_from"),
                        "section_title": ch.get("section_title"),
                    }
                )
                if len(table_rows) >= 4:
                    break
            if len(table_rows) >= 4:
                break

    retrieved = dedupe_retrieval_by_basename(retrieved, prefer_slugs=user_slugs)
    rag_protos = _build_protocols_from_retrieval(
        retrieved,
        prefer_slugs=user_slugs,
        icd_codes=icd_codes_for_lex,
    )
    rag_protos = _filter_protocols_by_funnel_audience(rag_protos, query)
    icd_w = env_float("RAG_HYBRID_ICD_WEIGHT", 0.4)
    rag_w = env_float("RAG_HYBRID_RAG_WEIGHT", 0.6)
    try:
        from clinical_knowledge.protocol_icd_index import _acute_respiratory_query
        from icd_mkb import is_symptom_code, normalize_code

        codes_norm = [normalize_code(str(c)) for c in (icd_codes or []) if normalize_code(str(c))]
        symptom_acute = bool(codes_norm) and all(is_symptom_code(c) for c in codes_norm)
        if symptom_acute or (_acute_respiratory_query(query) and not codes_norm):
            icd_w = env_float("RAG_HYBRID_ICD_WEIGHT_SYMPTOM", 0.62)
            rag_w = max(0.05, 1.0 - icd_w)
        elif icd_protos and not lookup.get("ambiguous"):
            icd_w = env_float("RAG_HYBRID_ICD_WEIGHT_STRONG", 0.58)
            rag_w = max(0.05, 1.0 - icd_w)
    except Exception:
        pass
    merged = hybrid_merge_protocols(
        icd_protos, rag_protos, icd_weight=icd_w, rag_weight=rag_w
    )
    merged = hybrid_pin_trusted_icd_top1(
        merged,
        icd_protos,
        query=query,
        ambiguous=bool(lookup.get("ambiguous")),
        icd_codes=icd_codes_for_lex or None,
    )
    if merged:
        merged = _rerank_protocols_symptom_only(merged, query, icd_analysis)
        merged = _filter_protocols_by_funnel_audience(merged, query)

    from clinical_knowledge.protocol_match_ui import enrich_protocol_match_ui

    icd_for_ui = list(icd_codes or []) or icd_codes_for_lex
    merged_enriched = enrich_protocol_match_ui(merged, icd_for_ui or None)
    protocols_total = len(merged_enriched)
    protocols_top = merged_enriched[:3]
    table_payload = table_rows if query_wants_tables(query) else []

    return {
        "query": query,
        "retrieve_only": True,
        "assist_lite": True,
        "hybrid_search": True,
        "icd_fast_lookup": bool(icd_protos),
        "lookup_ms": lookup.get("lookup_ms", 0),
        "icd_lookup_ambiguous": lookup.get("ambiguous", False),
        "match_reasons": match_reasons,
        "protocols_total": protocols_total,
        "llm_json": {
            "protocols": protocols_top,
            "icd_codes": (icd_analysis or {}).get("detected") or [],
            "table_excerpts": table_payload,
        },
        "icd": icd_analysis,
        "expanded_icd_codes": lookup.get("expanded_icd") or icd_codes_for_lex,
        "finish_reason": "HYBRID_SEARCH",
        "retrieved_count": len(retrieved),
    }


def get_rich_chunks_for_path(path: str) -> list[dict]:
    """Чанки одного PDF из RAG-индекса или lazy chunk store."""
    from clinical_knowledge.lazy_rag_config import lazy_chunk_store_enabled

    norm = path.replace("\\", "/").strip()
    if not norm:
        return []
    if lazy_chunk_store_enabled():
        store = _ensure_lazy_chunk_store()
        if store:
            rows = store.get_chunks_for_path(norm, max_chunks=256)
            if rows:
                return rows
    chunks = _chunks_by_path.get(norm) or []
    if chunks:
        return chunks
    for p, rows in _chunks_by_path.items():
        if p.replace("\\", "/") == norm or p.endswith(norm.split("/")[-1]):
            return rows
    return []


def get_rich_chunks_for_consult(path: str) -> list[dict]:
    """Чанки одного PDF для КЗ: урезанный набор (OOM-safe на Render)."""
    from clinical_knowledge.consult_memory import cap_chunks_for_consult

    return cap_chunks_for_consult(get_rich_chunks_for_path(path))


def _infer_funnel_audience(query: str, routing: dict | None = None) -> str | None:
    routing = routing or _routing or {}
    return infer_audience_from_funnel_context(query) or infer_audience_from_query(query, routing)


def _filter_protocols_by_funnel_audience(
    protos: list[dict],
    query: str,
    routing: dict | None = None,
) -> list[dict]:
    """Жёстко убирает КП несовпадающей аудитории при подтверждённом контексте воронки."""
    if not protos:
        return protos
    routing = routing or _routing or {}
    aud = _infer_funnel_audience(query, routing)
    if aud not in ("adult", "child"):
        return protos
    kept: list[dict] = []
    for pr in protos:
        path = str(pr.get("path") or "")
        title = str(pr.get("title") or path)
        hint = doc_audience_hint(path, title, routing)
        if aud == "adult" and hint == "pediatric":
            continue
        if aud == "child" and hint == "adult":
            continue
        kept.append(pr)
    return kept


def _demote_pediatric_for_adult_query(
    protos: list[dict],
    query: str,
    routing: dict | None = None,
) -> list[dict]:
    """Переносит явно детские КП в конец списка при запросе для взрослых."""
    if not protos:
        return protos
    routing = routing or _routing or {}
    if _infer_funnel_audience(query, routing) not in ("adult", "pregnant"):
        return protos
    obstetric_first: list[dict] = []
    adult_first: list[dict] = []
    pediatric: list[dict] = []
    mixed: list[dict] = []
    is_pregnant = _infer_funnel_audience(query, routing) == "pregnant"
    for pr in protos:
        path = str(pr.get("path") or "")
        title = str(pr.get("title") or path)
        hint = doc_audience_hint(path, title, routing)
        if is_pregnant and "akusherstvo-ginekologiya" in path.lower():
            obstetric_first.append(pr)
        elif hint == "pediatric":
            pediatric.append(pr)
        elif hint == "mixed":
            mixed.append(pr)
        else:
            adult_first.append(pr)
    if is_pregnant and obstetric_first:
        return obstetric_first + adult_first + mixed + pediatric
    if not adult_first and not mixed:
        return []
    return adult_first + mixed + pediatric


def _query_has_throat_or_uri_context(query: str, icd_analysis: dict | None) -> bool:
    ql = (query or "").lower()
    if any(
        w in ql
        for w in (
            "кашел",
            "температ",
            "лихорад",
            "озноб",
            "орви",
            "простуд",
            "горл",
            "глот",
            "насморк",
            "ангин",
            "фаринг",
            "дисфаг",
        )
    ):
        return True
    icd = icd_analysis or {}
    codes: list[str] = []
    for bucket in ("detected", "suggested"):
        for row in icd.get(bucket) or []:
            if isinstance(row, dict) and row.get("code"):
                codes.append(str(row["code"]).upper())
    for c in icd.get("codes_for_retrieval") or []:
        codes.append(str(c).upper())
    for m in re.finditer(r"\b([A-TV-Z]\d{2}(?:\.\d{1,2})?)\b", query or "", re.I):
        codes.append(m.group(1).upper())
    for code in codes:
        if code.startswith(("R07", "J0", "J2", "J06")):
            return True
    return False


def _icd_codes_from_analysis(icd_analysis: dict | None, query: str) -> list[str]:
    codes: list[str] = []
    icd = icd_analysis or {}
    for bucket in ("detected", "suggested"):
        for row in icd.get(bucket) or []:
            if isinstance(row, dict) and row.get("code"):
                codes.append(str(row["code"]))
    for c in icd.get("codes_for_retrieval") or []:
        codes.append(str(c))
    for m in re.finditer(r"\b([A-TV-Z]\d{2}(?:\.\d{1,2})?)\b", query or "", re.I):
        codes.append(m.group(1).upper())
    return list(dict.fromkeys(codes))


def _icd_codes_for_clinical_routing(icd_analysis: dict | None, query: str) -> list[str]:
    """МКБ для rerank/routing: только явные коды в тексте запроса (+ detected при explicit ICD)."""
    icd = icd_analysis or {}
    codes: list[str] = []
    for m in re.finditer(r"\b([A-TV-Z]\d{2}(?:\.\d{1,2})?)\b", query or "", re.I):
        codes.append(m.group(1).upper())
    if icd.get("explicit_icd_in_query"):
        for bucket in ("detected", "suggested"):
            for row in icd.get(bucket) or []:
                if isinstance(row, dict) and row.get("code"):
                    c = str(row["code"]).upper()
                    if c not in codes:
                        codes.append(c)
    return list(dict.fromkeys(codes))


def _rerank_protocols_symptom_only(
    protos: list[dict],
    query: str,
    icd_analysis: dict | None,
) -> list[dict]:
    """Понижает маловероятные КП по клиническому контексту, жалобам/ОРИ и аудитории."""
    if not protos:
        return protos
    from clinical_knowledge.search_clinical_routing import (
        detect_clinical_route_ids,
        score_path_for_clinical_routes,
    )

    icd_codes = _icd_codes_for_clinical_routing(icd_analysis, query)
    route_ids = detect_clinical_route_ids(query, icd_codes)
    uri_context = _query_has_throat_or_uri_context(query, icd_analysis)
    ql = (query or "").lower()
    routing = _routing or {}
    aud = _infer_funnel_audience(query, routing)
    child_query = aud == "child"
    adult_query = aud == "adult"
    pregnant_query = aud == "pregnant" or "pregnancy" in route_ids
    has_fever = any(w in ql for w in ("температ", "лихорад", "жар", "озноб"))
    has_cough = any(w in ql for w in ("кашел", "кашель", "сухой каш"))
    has_throat = any(w in ql for w in ("горл", "глот", "дисфаг", "глотать", "глотан"))

    def _apply_clinical_and_audience(pr: dict, penalty: int, boost: int) -> tuple[int, int]:
        path = str(pr.get("path") or "").lower()
        title = str(pr.get("title") or path).lower()
        blob = path + " " + title
        hint = doc_audience_hint(path, title, routing)
        if route_ids:
            delta, _ = score_path_for_clinical_routes(path, title, route_ids=route_ids)
            if delta >= 0:
                boost += int(delta)
            else:
                penalty += int(-delta)
        if pregnant_query:
            if "akusherstvo-ginekologiya" in path or any(
                k in blob for k in ("акушер", "беремен", "гинекolog", "гинеколог", "родов", "плацент")
            ):
                boost += 8
            if any(k in blob for k in ("невrolog", "нервн", "нейро", "детс", "д-нас", "дет_")):
                penalty += 18
            if hint == "pediatric":
                penalty += 14
        elif not pregnant_query and "pregnancy" not in route_ids:
            if "akusherstvo-ginekologiya" in path:
                penalty += 36
            elif any(k in blob for k in ("женщинам", "послерод", "акушer", "акушер", "репродуктивного возраста")):
                penalty += 30
        if "orvi_uri" in route_ids:
            if any(k in blob for k in ("psikhiatr", "психическ", "аффектив", "depress", "поведенческ")):
                penalty += 40
            if any(k in blob for k in ("инфекциями кожи", "кожи и подкожной")):
                penalty += 18
            if any(k in blob for k in ("респиратор", "орви", "орз", "простуд", "бронхит")):
                boost += 14
        if "otitis" in route_ids or any(str(c).upper().startswith(("H65", "H66", "H67")) for c in icd_codes):
            if any(k in blob for k in ("оторин", "otorin", "лор", "фаринг", "отит")):
                boost += 14
            if any(k in blob for k in ("нейрохирург", "нервной систем", "эпилепс", "врожденн")) and "отит" not in blob:
                penalty += 22
        if adult_query and hint == "pediatric":
            penalty += 48
        elif not child_query and not pregnant_query and hint == "pediatric":
            penalty += 8
        if child_query:
            if hint == "pediatric":
                boost += 3
            elif hint == "adult":
                penalty += 10
            if "бронхит" in blob and "дет" not in blob and "д-нас" not in blob and "pediatr" not in blob:
                penalty += 8
            if has_cough and any(
                k in blob
                for k in ("дет нас", "дет_нас", "д-нас", "детс", "pediatr", "орви", "респиратор")
            ):
                boost += 12
        if "orvi_uri" in route_ids or any(c in ql for c in ("орви", "орз")) or any(
            str(c).upper().startswith("J06") for c in icd_codes
        ):
            if any(k in blob for k in ("гепатит", "hepat", "вирусн гепат")):
                penalty += 16
            if "оториноларингологическ" in blob and "орви" not in blob and "респиратор" not in blob:
                penalty += 18
            if "риносинус" in blob and not any(
                w in ql for w in ("sinus", "синус", "лоб", "лобно", "рinosinus", "риносинус")
            ):
                if "орви" not in blob and "респиратор" not in blob:
                    penalty += 16
        if any(w in ql for w in ("горл", "ангин", "глот", "отит", "ухо", "фаринг")):
            if any(k in blob for k in ("оторин", "otorin", "лор", "фаринг", "отит")):
                boost += 20
            if any(k in blob for k in ("эпилепс", "врожденн", "нейрохирург")) and "отит" not in blob:
                penalty += 35
        if "wound" in route_ids and any(w in ql for w in ("рана", "раны", "раной", "порез")):
            if not any(w in ql for w in ("перелом",)):
                if any(
                    k in blob
                    for k in (
                        "травмой живота",
                        "травма живота",
                        "травм живота",
                        "женщинам",
                        "акушer",
                    )
                ):
                    penalty += 30
                if any(k in blob for k in ("огнестрел", "огнестр", "ран", "ранен", "khirurg", "хирург")):
                    boost += 12
        if "burn" in route_ids and any(w in ql for w in ("ожог", "термическ", "обвар")):
            if any(k in blob for k in ("скорой неотложной", "неотложной медицинской помощи")):
                penalty += 20
        if any(str(c).upper().startswith("J20") for c in icd_codes) and "дистресс" in blob:
            if "бронхит" not in blob:
                penalty += 18
        if "dermatology_appendage" in route_ids or any(
            str(c).upper().startswith(("L60", "L61", "L62", "L63", "L64", "L65", "L66", "L67", "L68"))
            for c in icd_codes
        ):
            if any(k in blob for k in ("придатков", "ногт", "оних", "L60")):
                boost += 18
            if any(k in blob for k in ("папулосквамоз", "псориаз")) and not any(
                k in ql for k in ("папулосквамоз", "псориаз", "бляшк")
            ):
                penalty += 22
            if adult_query and hint == "pediatric":
                penalty += 20
        if "otitis" in route_ids or any(str(c).upper().startswith("H66") for c in icd_codes):
            if any(k in blob for k in ("нейрохирург", "нервной систем", "nevrolog")) and "отит" not in blob:
                penalty += 20
        return penalty, boost

    if not uri_context:
        def _score_clinical_only(pr: dict) -> tuple[int, float]:
            penalty, boost = _apply_clinical_and_audience(pr, 0, 0)
            path = str(pr.get("path") or "").lower()
            title = str(pr.get("title") or path).lower()
            blob = path + " " + title
            from clinical_knowledge.search_retrieval import (
                _GI_TITLE_STRONG,
                _GI_TITLE_WRONG,
                _has_functional_gi_context,
            )

            icd_codes_local = _icd_codes_from_analysis(icd_analysis, query)
            if _has_functional_gi_context(query, icd_codes_local):
                if any(k in blob for k in _GI_TITLE_WRONG):
                    penalty += 22
                elif any(k in blob for k in _GI_TITLE_STRONG):
                    boost += 10
            try:
                conf = float(pr.get("confidence_score") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            return (penalty - boost, -conf)

        if route_ids or child_query or adult_query or pregnant_query:
            ranked = sorted(protos, key=_score_clinical_only)
            ranked = _demote_pediatric_for_adult_query(ranked, query, routing)
            ranked = _swap_if_clinical_second_beats_first(ranked, query, icd_analysis)
            ranked = _promote_clinical_route_top1(ranked, query, icd_analysis)
            return _filter_protocols_by_funnel_audience(ranked, query, routing)
        ranked = _demote_pediatric_for_adult_query(protos, query, routing)
        ranked = _swap_if_clinical_second_beats_first(ranked, query, icd_analysis)
        ranked = _promote_clinical_route_top1(ranked, query, icd_analysis)
        return _filter_protocols_by_funnel_audience(ranked, query, routing)

    rare = ("саркоид", "микобактер", "туберкул", "лихорадка ку")
    chronic_mismatch = ("аллерг", "ринит", "хобл", "реабилит", "реабилитац", "интерстициальн")
    gi_mismatch = ("пищевод", "желудк", "двенадцатипер", "гастроэзофаг", "рефлюкс", "гэрб", "срыгив")
    acute = ("пневмон", "орви", "бронхит", "орз", "остры", "неотложн", "жаропонижа")
    uri = ("фаринг", "ангин", "тонзилл", "ларинг", "респираторн", "орз", "грипп", "фингит", "оторин", "лор", "уха горла носа")
    cough_acute = ("бронхит", "пневмон", "орви", "орз", "респиратор", "грипп", "трахеит")
    ent_only = ("отит", "риносинус", "синусит", "аденоид")
    wrong_acute = ("паллиат", "саркоид", "иммунодефиц", "трансплант", "онколог")
    has_gi = any(w in ql for w in ("живот", "изжог", "тошн", "рвот", "желуд", "кишеч", "стул"))
    has_functional_gi = any(
        w in ql
        for w in (
            "запор",
            "вздут",
            "метеор",
            "дисхез",
            "дефекац",
            "срк",
            "гастр",
            "кишечник",
        )
    ) or any(str(c).upper().startswith(("K58", "K59")) for c in (icd_codes or []))
    gi_trauma_title = (
        "травмой живота",
        "травма живота",
        "травм живота",
        "огнестрел",
        "огнестр",
        "ранени",
        "ранен",
        "ранами",
    )
    gi_strong_title = (
        "кишеч",
        "кишечник",
        "запор",
        "дефекац",
        "гастр",
        "моторно",
        "эвакуатор",
        "пищевар",
    )
    throat_distress = any(w in ql for w in ("одыш", "дистресс", "синус", "сатурац", "загиб", "трипод"))
    has_sinus_hint = any(w in ql for w in ("синус", "риносинус", "лоб", "лобно", "maxillar"))

    def _score_row(pr: dict) -> tuple[int, float]:
        path = str(pr.get("path") or "").lower()
        title = str(pr.get("title") or path).lower()
        blob = path + " " + title
        penalty = 0
        boost = 0
        if any(k in blob for k in rare) and not any(k in ql for k in rare):
            penalty += 2
        if has_fever and any(k in blob for k in chronic_mismatch):
            penalty += 5
        if has_throat and not has_gi and any(k in blob for k in gi_mismatch):
            penalty += 8
        if has_functional_gi and not any(w in ql for w in ("травм", "ранен", "огнестрел")):
            if any(k in blob for k in gi_trauma_title):
                penalty += 20
            elif any(k in blob for k in gi_strong_title):
                boost += 8
        if child_query and has_cough and (has_fever or "температ" in ql):
            if any(k in blob for k in ("дет нас", "дет_нас", "д-нас", "детс", "pediatr", "орви", "респиратор")):
                boost += 14
            if "бронхит" in blob and not any(
                k in blob for k in ("дет нас", "дет_нас", "д-нас", "детс", "pediatr")
            ):
                penalty += 22
        elif child_query and has_cough:
            if any(k in blob for k in ("дет нас", "дет_нас", "д-нас", "детс", "pediatr", "орви", "респиратор")):
                boost += 16
            if "бронхит" in blob and not any(
                k in blob for k in ("дет нас", "дет_нас", "д-нас", "детс", "pediatr")
            ):
                penalty += 24
            if not has_fever and doc_audience_hint(path, title, routing) == "adult":
                penalty += 14
        if has_throat and not throat_distress and "эпиглоттит" in blob:
            penalty += 4
        if any(k in blob for k in ("анестезиолог", "анестези", "хирургическ")) and has_throat:
            penalty += 10
        if has_throat and any(k in blob for k in ("аллерг", "ринит")):
            penalty += 6
        if (has_cough or has_fever) and any(k in blob for k in wrong_acute):
            penalty += 8
        if has_cough and not has_throat:
            if any(k in blob for k in cough_acute):
                boost += 7
            elif any(k in blob for k in ent_only) and not any(k in blob for k in cough_acute):
                penalty += 10
            if not has_sinus_hint and "риносинус" in blob and "орви" not in blob:
                penalty += 14
            if "оториноларингологическ" in blob and "орви" not in blob and "респиратор" not in blob:
                penalty += 22
            if adult_query and has_fever and "бронхит" in blob and "орви" not in blob and "респиратор" not in blob:
                if "оториноларингологическ" in blob or "уха горла носа" in blob:
                    penalty += 18
                else:
                    penalty += 8
            if adult_query and has_fever and any(k in blob for k in ("респиратор", "орви", "орз", "пневмон")):
                boost += 6
        elif has_throat:
            if "пневмон" in blob:
                boost += 2
            elif any(k in blob for k in uri):
                boost += 2
            elif any(k in blob for k in acute):
                boost += 1
        else:
            if "пневмон" in blob:
                boost += 2
            elif any(k in blob for k in cough_acute):
                boost += 2
            elif any(k in blob for k in acute):
                boost += 1
        if child_query and has_cough and any(k in blob for k in cough_acute):
            boost += 2
        penalty, boost = _apply_clinical_and_audience(pr, penalty, boost)
        try:
            conf = float(pr.get("confidence_score") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        return (penalty - boost, -conf)

    ranked = sorted(protos, key=_score_row)
    ranked = _demote_pediatric_for_adult_query(ranked, query, routing)
    ranked = _swap_if_clinical_second_beats_first(ranked, query, icd_analysis)
    ranked = _promote_clinical_route_top1(ranked, query, icd_analysis)
    return _filter_protocols_by_funnel_audience(ranked, query, routing)


def _swap_if_clinical_second_beats_first(
    protos: list[dict],
    query: str,
    icd_analysis: dict | None,
) -> list[dict]:
    """Поднимает в top-1 протокол с лучшим clinical delta среди top-N, если он заметно лучше текущего."""
    if len(protos) < 2:
        return protos
    from clinical_knowledge.search_clinical_routing import (
        detect_clinical_route_ids,
        score_path_for_clinical_routes,
    )

    route_ids = detect_clinical_route_ids(query, _icd_codes_for_clinical_routing(icd_analysis, query))
    if not route_ids:
        return protos

    def _delta(pr: dict) -> float:
        path = str(pr.get("path") or "")
        title = str(pr.get("title") or path)
        d, matched = score_path_for_clinical_routes(path, title, route_ids=route_ids)
        return d if matched else 0.0

    window = min(8, len(protos))
    deltas = [_delta(protos[i]) for i in range(window)]
    best_i = max(range(window), key=lambda i: deltas[i])
    if best_i > 0 and deltas[best_i] >= 10.0 and deltas[best_i] > deltas[0] + 4.0:
        out = list(protos)
        out.insert(0, out.pop(best_i))
        return out
    return protos


def _promote_clinical_route_top1(
    protos: list[dict],
    query: str,
    icd_analysis: dict | None,
) -> list[dict]:
    """Поднимает top-1 по активному клиническому маршруту (беременность, ОРВИ…)."""
    if len(protos) < 2:
        return protos
    from clinical_knowledge.search_clinical_routing import (
        detect_clinical_route_ids,
        score_path_for_clinical_routes,
    )

    icd_codes = _icd_codes_for_clinical_routing(icd_analysis, query)
    route_ids = detect_clinical_route_ids(query, icd_codes)
    if not route_ids:
        return protos

    def _route_delta(pr: dict) -> float:
        path = str(pr.get("path") or "")
        title = str(pr.get("title") or path)
        delta, matched = score_path_for_clinical_routes(path, title, route_ids=route_ids)
        return delta if matched else 0.0

    top0_delta = _route_delta(protos[0])
    if top0_delta >= 8.0:
        return protos

    best_idx: int | None = None
    best_delta = max(8.0, top0_delta + 4.0)
    for i, pr in enumerate(protos[:12]):
        delta = _route_delta(pr)
        if delta >= best_delta:
            best_delta = delta
            best_idx = i
    if best_idx is not None and best_idx > 0:
        out = list(protos)
        out.insert(0, out.pop(best_idx))
        return out
    return protos


def dedupe_parsed_protocols(
    parsed: dict | None,
    *,
    prefer_slugs: list[str] | None = None,
) -> None:
    if not parsed or not isinstance(parsed, dict):
        return
    protos = parsed.get("protocols")
    if not isinstance(protos, list):
        return
    parsed["protocols"] = dedupe_protocols_list(protos, prefer_slugs=prefer_slugs)


# --- Выдержки из PDF: склейка переносов, обрезка по границам слов (для UI) ---

_RU_SINGLE_LETTER_WORDS = frozenset(
    "и а в к о с у я ы э ю ё".split()
)

_PDF_HYPHEN_PAIR = re.compile(
    r"([а-яёА-ЯЁa-zA-Z])-\s+([а-яёА-ЯЁa-zA-Z])"
)
_PDF_HYPHEN_NL = re.compile(
    r"([а-яёА-ЯЁa-zA-Z])-\s*\n\s*([а-яёА-ЯЁa-zA-Z])"
)


def _normalize_pdf_hyphenation(text: str) -> str:
    """Склеивает переносы из PDF: «меди- цинской», «Воз- можны» → цельные слова."""
    if not text:
        return ""
    t = text.replace("\u00ad", "")
    for _ in range(24):
        t2 = _PDF_HYPHEN_PAIR.sub(lambda m: m.group(1) + m.group(2), t)
        if t2 == t:
            break
        t = t2
    for _ in range(24):
        t2 = _PDF_HYPHEN_NL.sub(lambda m: m.group(1) + m.group(2), t)
        if t2 == t:
            break
        t = t2
    return t


def _collapse_whitespace_for_excerpt(text: str) -> str:
    """Один блок текста без разрывов строк из верстки PDF."""
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t\u00a0]+", " ", t)
    t = re.sub(r"\s*\n\s*", " ", t)
    t = re.sub(r" +", " ", t)
    return t.strip()


def _strip_leading_word_fragment(text: str) -> str:
    """Убирает обрезанное первое «слово» (часто 1 буква: «й» от «Настоящий»)."""
    t = text.strip()
    if len(t) < 3:
        return t
    m = re.match(r"^(\S+)(\s+)", t)
    if not m:
        return t
    first = m.group(1)
    if len(first) != 1:
        return t
    if not first.isalpha():
        return t
    if first.lower() in _RU_SINGLE_LETTER_WORDS:
        return t
    return t[m.end() :].lstrip()


def _truncate_excerpt_for_ui(text: str, max_chars: int) -> str:
    """Обрезка по границе слова; без обрыва на середине слова; многоточие при необходимости."""
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    min_cut = max(24, int(max_chars * 0.5))
    sp = window.rfind(" ")
    if sp >= min_cut:
        window = window[:sp]
    else:
        for sep in (";", ":", ",", "»", ")"):
            ix = window.rfind(sep)
            if ix >= min_cut // 2:
                window = window[: ix + 1]
                break
    window = window.rstrip(" -- - ")
    while window.endswith("-") and len(window) > 1:
        window = window[:-1].rstrip()
    if not window:
        window = text[: max_chars - 1].rstrip() + "…"
        return window
    if window[-1] not in ".!?…:;»)]":
        window += "…"
    return window


def format_excerpt_for_display(raw: str, max_chars: int) -> str:
    """Пайплайн для фрагмента КП в ответе API и промпте.

    Обрезка по границе предложения (без обрывков на полуфразе); при невозможности -
    fallback на обрезку по границе слова.
    """
    from clinical_knowledge.protocol_audience import is_synthetic_summary_excerpt

    if is_synthetic_summary_excerpt(raw or ""):
        return ""
    t = _normalize_pdf_hyphenation(raw or "")
    t = _collapse_whitespace_for_excerpt(t)
    t = _strip_leading_word_fragment(t)
    if len(t) <= max_chars:
        return t
    try:
        from clinical_knowledge.meaningful_excerpt import meaningful_excerpt

        sent = meaningful_excerpt(t, limit=max_chars)
        if sent:
            if len(sent) < len(t) and sent[-1] not in ".!?…:;»)]":
                sent += "…"
            return sent
    except Exception:
        pass
    return _truncate_excerpt_for_ui(t, max_chars)


def format_structured_index_text(raw: str, max_chars: int) -> str:
    """Текст из structured_index: те же правила, другой лимит."""
    t = _normalize_pdf_hyphenation(raw or "")
    t = _collapse_whitespace_for_excerpt(t)
    t = _strip_leading_word_fragment(t)
    if len(t) <= max_chars:
        return t
    return _truncate_excerpt_for_ui(t, max_chars)


_REDFLAGS_KEYWORDS = (
    "госпитализац",
    "стационар",
    "неотложн",
    "скорой помощ",
    "экстренн",
    "немедленн",
    "угроза жизни",
    "реанимац",
    "орит",
    "интенсивной терапии",
    "показания к госпитал",
    "направлени",
    "жизнеугрожающ",
    "опасн для жизни",
    "срочной медицинской",
)


def _red_flags_from_retrieval(retrieved: list[dict]) -> list[str]:
    """Эвристика по отобранным фрагментам: предложения/строки с маркерами срочности/стационара."""
    if not retrieved:
        return []
    parts: list[str] = []
    for row in retrieved[:6]:
        ex = (row.get("excerpt") or "").strip()
        if not ex:
            continue
        for para in re.split(r"(?<=[.!?])\s+|\n+", ex):
            t = para.strip()
            if len(t) < 30:
                continue
            low = t.lower()
            if any(k in low for k in _REDFLAGS_KEYWORDS):
                parts.append(t)
    seen: set[str] = set()
    out: list[str] = []
    for s in parts:
        s = re.sub(r"\s+", " ", s)
        if len(s) > 240:
            s = s[:237] + "…"
        key = s[:72]
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= 5:
            break
    return out


def _protocol_icd_mentions_for_response(
    protocols: list,
    *,
    top_n: int = 5,
    focus_codes: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Топ кодов МКБ-10 по тексту протокола; при focus_codes - сначала сверка с запросом."""
    out: dict[str, list[dict]] = {}
    if not _chunks_by_path and not _structured_by_path:
        return out
    fc = [normalize_icd_code(str(c)) for c in (focus_codes or []) if c]
    fc = list(dict.fromkeys([x for x in fc if x]))
    use_focus = fc or None
    for pr in protocols:
        if not isinstance(pr, dict):
            continue
        raw = str(pr.get("path") or "").strip()
        if not raw:
            continue
        nk = _normalize_protocol_path_key(raw)
        blob_parts: list[str] = []
        # Приоритет - полный текст по чанкам данного протокола (на практике точнее structured_index).
        rows = _chunks_by_path.get(raw) or _chunks_by_path.get(nk) or []
        if rows:
            for ch in rows:
                if not isinstance(ch, dict):
                    continue
                txt = str(ch.get("text") or "").strip()
                if txt:
                    blob_parts.append(txt)
        if not blob_parts:
            struct = _structured_by_path.get(raw) or _structured_by_path.get(nk)
            if struct and isinstance(struct, dict):
                blob_parts = [
                    str(struct.get("diagnosis") or "").strip(),
                    str(struct.get("treatment") or "").strip(),
                    str(struct.get("summary") or "").strip(),
                ]
        blob = "\n\n".join(p for p in blob_parts if p).strip()
        if not blob:
            continue
        top_rows = count_icd_code_mentions(
            blob, top_n=top_n, focus_codes=use_focus
        )
        if top_rows:
            out[raw] = top_rows
    return out


def _icd_codes_from_query_and_analysis(
    query: str, icd_codes: list[str] | None = None
) -> list[str]:
    if icd_codes:
        return [normalize_icd_code(str(c)) for c in icd_codes if normalize_icd_code(str(c))]
    q_rag = clinical_query_for_rag(query)
    analysis = analyze_query_for_icd(query, q_rag)
    raw = analysis.get("codes_for_retrieval") or []
    out: list[str] = []
    for c in raw:
        nc = normalize_icd_code(str(c))
        if nc and nc not in out:
            out.append(nc)
    return out[:12]


def _mark_icd_related_in_matrix(matrix: dict, icd_codes: list[str]) -> None:
    if not icd_codes:
        return
    icd_low = [c.lower() for c in icd_codes]
    for sec in matrix.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        for it in sec.get("items") or []:
            if not isinstance(it, dict):
                continue
            blob = " ".join(
                str(it.get(x) or "")
                for x in ("text", "protocol_excerpt", "protocol_ref")
            ).lower()
            for c in icd_low:
                if c in blob:
                    it["icd_related"] = True
                    break


def _normalize_kz_matrix(parsed: dict, path: str, title: str, icd_codes: list[str]) -> dict:
    sections_out: list[dict] = []
    allowed = set(KZ_MATRIX_SECTIONS)
    for sec in parsed.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        kz = str(sec.get("kz_section") or "").strip()
        if kz not in allowed:
            for cand in KZ_MATRIX_SECTIONS:
                if cand.lower() in kz.lower() or kz.lower() in cand.lower():
                    kz = cand
                    break
            else:
                continue
        items_in: list[dict] = []
        for it in sec.get("items") or []:
            if not isinstance(it, dict):
                continue
            text = str(it.get("text") or "").strip()
            if not text:
                continue
            ob = str(it.get("obligation") or "recommended").strip().lower()
            if ob not in ("required", "recommended", "conditional", "not_applicable"):
                ob = "recommended"
            items_in.append(
                {
                    "text": text[:500],
                    "obligation": ob,
                    "protocol_ref": str(it.get("protocol_ref") or "").strip()[:120],
                    "protocol_excerpt": str(it.get("protocol_excerpt") or "").strip()[:600],
                    "icd_related": bool(it.get("icd_related")),
                }
            )
        if items_in:
            sections_out.append({"kz_section": kz, "items": items_in})
    order = {s: i for i, s in enumerate(KZ_MATRIX_SECTIONS)}
    sections_out.sort(key=lambda s: order.get(s["kz_section"], 99))
    return {
        "path": path,
        "protocol_title": title,
        "icd_codes": icd_codes,
        "summary_ru": str(parsed.get("summary_ru") or "").strip(),
        "sections": sections_out,
        "disclaimer_ru": str(parsed.get("disclaimer_ru") or "").strip()
        or "Ориентир по протоколу; не замена очной экспертизы.",
    }


def kz_matrix_from_clinical_detail_heuristic(
    cd: dict,
    path: str,
    title: str,
    icd_codes: list[str],
) -> dict:
    """Быстрый ориентир без LLM (если отдельный вызов матрицы недоступен)."""
    ex = cd.get("extraction") if isinstance(cd.get("extraction"), dict) else cd
    sections: list[dict] = []

    def add_section(kz_section: str, texts: list[str], obligation: str = "recommended") -> None:
        items = []
        for t in texts:
            t = str(t).strip()
            if not t:
                continue
            items.append(
                {
                    "text": t[:500],
                    "obligation": obligation,
                    "protocol_ref": "",
                    "protocol_excerpt": "",
                    "icd_related": False,
                }
            )
        if items:
            sections.append({"kz_section": kz_section, "items": items})

    diag = str(ex.get("diagnosis") or "").strip()
    if diag:
        add_section("Диагноз и коды МКБ-10", [diag], "required")
    inv = [str(x).strip() for x in (ex.get("investigations") or []) if str(x).strip()]
    if inv:
        add_section("Обследование", inv[:12], "required")
    meds = [str(x).strip() for x in (ex.get("medications") or []) if str(x).strip()]
    treat = [str(x).strip() for x in (ex.get("treatment_methods") or []) if str(x).strip()]
    if meds or treat:
        add_section("Лечение и назначения", (meds + treat)[:14], "recommended")
    mon = str(ex.get("monitoring_frequency") or "").strip()
    fol = str(ex.get("monitoring_followup") or "").strip()
    if mon or fol:
        add_section("Наблюдение и контроль", [x for x in [mon, fol] if x], "recommended")
    recs = [str(x).strip() for x in (ex.get("recommendations") or []) if str(x).strip()]
    if recs:
        add_section("Направления и консультации специалистов", recs[:8], "conditional")
    out = {
        "path": path,
        "protocol_title": title,
        "icd_codes": icd_codes,
        "summary_ru": "Ориентир составлен по развёрнутой выдержке из протокола (без отдельного прохода модели).",
        "sections": sections,
        "disclaimer_ru": "Проверьте полноту по PDF на сайте Минздрава.",
        "source": "heuristic",
    }
    _mark_icd_related_in_matrix(out, icd_codes)
    return out


def build_kz_matrix(
    model,
    query: str,
    path: str,
    title: str,
    icd_codes: list[str] | None = None,
    clinical_detail: dict | None = None,
) -> dict:
    icd_norm = _icd_codes_from_query_and_analysis(query, icd_codes)
    meta = _protocol_meta.get(path) or {}
    title_line = _protocol_display_title(path, title or meta.get("title"))
    max_body = int(os.environ.get("RAG_KZ_MATRIX_MAX_CHARS", "24000"))
    body = gather_protocol_text(path, max_body)
    if len(body.strip()) < 80:
        if clinical_detail and not clinical_detail.get("error"):
            return kz_matrix_from_clinical_detail_heuristic(
                clinical_detail, path, title_line, icd_norm
            )
        return {
            "path": path,
            "protocol_title": title_line,
            "icd_codes": icd_norm,
            "sections": [],
            "summary_ru": "Недостаточно текста протокола для матрицы КЗ.",
            "disclaimer_ru": "",
            "source": "empty",
        }
    icd_line = ", ".join(icd_norm) if icd_norm else "(не указаны явно)"
    cd_hint = ""
    if clinical_detail and isinstance(clinical_detail.get("extraction"), dict):
        cd_hint = "\n\nПодсказка (уже извлечённая выдержка JSON, не добавляй лишнего):\n" + json.dumps(
            clinical_detail.get("extraction"),
            ensure_ascii=False,
        )[:6000]
    prompt = (
        SYSTEM_KZ_MATRIX
        + "\n\n---\n\nЗапрос пользователя / случай:\n"
        + query[:8000]
        + "\n\nКоды МКБ для акцента:\n"
        + icd_line
        + "\n\nНазвание протокола:\n"
        + title_line
        + "\n\nТекст протокола:\n"
        + body
        + cd_hint
    )
    plim = int(os.environ.get("RAG_KZ_MATRIX_PROMPT_MAX_CHARS", "28000"))
    if len(prompt) > plim:
        prompt = prompt[: plim - 80] + "\n…[обрезано]"
    try:
        resp = generate_gemini(model, prompt)
        txt = _extract_gemini_text(resp)
        parsed = _try_parse_json(txt)
    except HTTPException:
        raise
    except Exception as e:
        if clinical_detail and not clinical_detail.get("error"):
            out = kz_matrix_from_clinical_detail_heuristic(
                clinical_detail, path, title_line, icd_norm
            )
            out["note"] = f"Матрица КЗ (эвристика): {e!s}"[:200]
            return out
        raise HTTPException(status_code=502, detail=f"Матрица КЗ: {e!s}") from e
    if not parsed or not isinstance(parsed, dict):
        if clinical_detail and not clinical_detail.get("error"):
            return kz_matrix_from_clinical_detail_heuristic(
                clinical_detail, path, title_line, icd_norm
            )
        raise HTTPException(
            status_code=502,
            detail="Модель не вернула JSON для матрицы консультативного заключения.",
        )
    out = _normalize_kz_matrix(parsed, path, title_line, icd_norm)
    out["source"] = "llm"
    _mark_icd_related_in_matrix(out, icd_norm)
    return out


def normalize_differential_field(parsed: dict | None) -> None:
    """До 5 строк; порядок как у модели (сверху - наиболее вероятное)."""
    if not parsed or not isinstance(parsed, dict):
        return
    d = parsed.get("differential")
    if not isinstance(d, list):
        return
    out: list[str] = []
    for x in d:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
        elif isinstance(x, dict):
            t = (x.get("text") or x.get("label") or x.get("diagnosis") or "").strip()
            if t:
                out.append(t)
        if len(out) >= 5:
            break
    parsed["differential"] = out


def _finish_hits_max(resp) -> bool:
    fr = (_gemini_finish_reason(resp) or "").upper()
    return "MAX" in fr or "LENGTH" in fr


_rag_load_thread_started = False
_rag_load_thread_lock = threading.Lock()
_log = logging.getLogger("protocol.rag")


def _start_rag_load_thread() -> None:
    """Загрузка корпуса после bind порта uvicorn (Render health check / port scan)."""
    global _rag_load_thread_started
    with _rag_load_thread_lock:
        if _rag_load_thread_started:
            return
        _rag_load_thread_started = True

    delay = env_float("RAG_STARTUP_LOAD_DELAY_SEC", 1.5 if env_bool("RENDER", False) else 0.0)

    def _runner() -> None:
        if delay > 0:
            time.sleep(delay)
        _run_load_data_background()

    threading.Thread(target=_runner, daemon=True, name="rag-load-chunks").start()
    _log.info("RAG corpus load scheduled (delay=%.1fs)", delay)


@asynccontextmanager
async def _app_lifespan(_application: FastAPI):
    _log.info("Protocol RAG lifespan startup")
    _start_rag_load_thread()
    yield
    _log.info("Protocol RAG lifespan shutdown")


app = FastAPI(title="Protocol RAG", version="1", lifespan=_app_lifespan)


# --- Безопасность: раздача статики только безопасных файлов ---
# Фронтенду нужны: index.html, consult_review.html (явные маршруты), docs/*.html, protocols.json,
# css/js/шрифты/картинки. Всё остальное (код, конфиги, ПДн-PDF, данные) не должно отдаваться по '/'.
_STATIC_BLOCKED_DIRS = {
    "data", "clients_consult", "tests", "eval", "scripts", "corpus_pipeline",
    "output", "corpus_chunks_parts", "minzdrav_protocols", "e2e", "__pycache__",
    "terminals", "node_modules",
}
_STATIC_BLOCKED_EXTS = {
    ".py", ".pyc", ".pyo", ".env", ".sh", ".toml", ".ini", ".cfg", ".lock",
    ".csv", ".jsonl", ".txt", ".md", ".mdc", ".yaml", ".yml", ".log",
}
_STATIC_BLOCKED_FILES = {
    "protocol_meta.json", "symptom_routing.json", "structured_index.json",
    "chunks.json", "corpus.json", "semantic_embeddings.json",
}


def _is_blocked_static_path(path: str) -> bool:
    norm = (path or "").replace("\\", "/").strip("/")
    if not norm:
        return False
    segments = [s for s in norm.split("/") if s]
    for seg in segments:
        low = seg.lower()
        if low in _STATIC_BLOCKED_DIRS or low.startswith("."):
            return True
    base = segments[-1].lower()
    if base in _STATIC_BLOCKED_FILES:
        return True
    ext = ("." + base.rsplit(".", 1)[1]) if "." in base else ""
    if ext in _STATIC_BLOCKED_EXTS:
        return True
    return False


class SafeStaticFiles(StaticFiles):
    """StaticFiles, не отдающий исходники, конфиги, данные и ПДн-PDF из корня репозитория."""

    async def get_response(self, path, scope):  # type: ignore[override]
        if _is_blocked_static_path(path):
            return PlainTextResponse("Not found", status_code=404)
        return await super().get_response(path, scope)


# --- Безопасность: CORS из окружения (по умолчанию только same-origin) ---
def _parse_cors_origins() -> list[str]:
    raw = (os.environ.get("ALLOWED_ORIGINS") or "").strip()
    if not raw:
        return []
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


_CORS_ORIGINS = _parse_cors_origins()
if _CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )


# --- Безопасность: rate-limiting (in-memory, по IP) на дорогие маршруты ---
_RATE_LIMIT_ENABLED = env_bool("RATE_LIMIT_ENABLED", True)
_RATE_WINDOW_SEC = 60.0
_RATE_LIMITS: dict[str, int] = {
    "/api/assist": env_int("RATE_LIMIT_ASSIST_PER_MIN", 10),
    "/api/consult-review": env_int("RATE_LIMIT_CONSULT_PER_MIN", 2),
    "/api/patient/review": env_int("RATE_LIMIT_PATIENT_PER_MIN", 5),
    "/api/patient/review/stream": env_int("RATE_LIMIT_PATIENT_PER_MIN", 5),
    "/api/patient/analytics": env_int("RATE_LIMIT_PATIENT_PER_MIN", 30),
    "/api/ml/feedback": env_int("RATE_LIMIT_ML_FEEDBACK_PER_MIN", 30),
    "/api/ml/feedback/export": env_int("RATE_LIMIT_ML_FEEDBACK_EXPORT_PER_MIN", 10),
    "/api/protocol-practical": env_int("RATE_LIMIT_PRACTICAL_PER_MIN", 5),
    "/api/search/feedback": env_int("RATE_LIMIT_SEARCH_FEEDBACK_PER_MIN", 20),
    "/api/search/run": env_int("RATE_LIMIT_SEARCH_RUN_PER_MIN", 15),
    "/api/protocol-detail": env_int("RATE_LIMIT_DETAIL_PER_MIN", 30),
    "/api/consultation-template": env_int("RATE_LIMIT_TEMPLATE_PER_MIN", 20),
    "/api/icd-suggest": env_int("RATE_LIMIT_ICD_PER_MIN", 40),
    "/api/verify-key": env_int("RATE_LIMIT_VERIFY_PER_MIN", 3),
}
_RATE_LIMIT_DEFAULT = env_int("RATE_LIMIT_DEFAULT_PER_MIN", 60)
_rate_state: dict[str, list[float]] = {}
_rate_lock = threading.Lock()


def _client_ip(request: "Request") -> str:
    xff = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if xff:
        return xff
    return request.client.host if request.client else "unknown"


def _rate_limit_allows(key: str, limit: int) -> bool:
    if limit <= 0:
        return True
    now = time.time()
    cutoff = now - _RATE_WINDOW_SEC
    with _rate_lock:
        bucket = _rate_state.setdefault(key, [])
        drop = 0
        for ts in bucket:
            if ts < cutoff:
                drop += 1
            else:
                break
        if drop:
            del bucket[:drop]
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True


_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
}
_CSP_VALUE = (os.environ.get("CONTENT_SECURITY_POLICY") or "").strip()
if not _CSP_VALUE and env_bool("ENABLE_DEFAULT_CSP", False):
    _CSP_VALUE = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "img-src 'self' data: https:; "
        "connect-src 'self' https:; "
        "frame-ancestors 'self'"
    )


_logger = logging.getLogger("protocol.rag")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    _logger.addHandler(_h)
    _logger.setLevel(getattr(logging, (os.environ.get("LOG_LEVEL") or "INFO").upper(), logging.INFO))
_REQUEST_LOG = env_bool("REQUEST_LOG", True)
_SLOW_REQUEST_MS = env_int("SLOW_REQUEST_MS", 8000)


def _health_live_response(request: "Request") -> Response | None:
    """Liveness до роутинга и StaticFiles - Render health check не получает 404."""
    path = (request.url.path or "").rstrip("/") or "/"
    if path != "/health/live":
        return None
    if request.method not in ("GET", "HEAD"):
        return None
    headers = {"X-App-Version": _app_version(), "Cache-Control": "no-store"}
    if request.method == "HEAD":
        return Response(status_code=200, headers=headers)
    return JSONResponse({"ok": True, "version": _app_version()}, headers=headers)


@app.middleware("http")
async def _security_and_rate_limit(request: "Request", call_next):
    live = _health_live_response(request)
    if live is not None:
        return live
    if _RATE_LIMIT_ENABLED:
        path = request.url.path
        explicit = path in _RATE_LIMITS
        if explicit or (request.method == "POST" and path.startswith("/api/")):
            limit = _RATE_LIMITS.get(path, _RATE_LIMIT_DEFAULT)
            if not _rate_limit_allows(f"{_client_ip(request)}|{path}", limit):
                if _REQUEST_LOG:
                    _logger.warning("429 rate-limit %s %s ip=%s", request.method, path, _client_ip(request))
                return JSONResponse(
                    {"detail": "Слишком много запросов. Подождите минуту и повторите."},
                    status_code=429,
                )
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        dur_ms = (time.perf_counter() - started) * 1000.0
        _logger.exception("500 %s %s after %.0f ms", request.method, request.url.path, dur_ms)
        raise
    dur_ms = (time.perf_counter() - started) * 1000.0
    for hk, hv in _SECURITY_HEADERS.items():
        response.headers.setdefault(hk, hv)
    if _CSP_VALUE:
        response.headers.setdefault("Content-Security-Policy", _CSP_VALUE)
    response.headers.setdefault("X-Process-Time-Ms", str(int(dur_ms)))
    if _REQUEST_LOG and (request.url.path.startswith("/api/") or response.status_code >= 400):
        level = logging.WARNING if (response.status_code >= 400 or dur_ms >= _SLOW_REQUEST_MS) else logging.INFO
        _logger.log(
            level,
            "%s %s -> %s %.0f ms",
            request.method,
            request.url.path,
            response.status_code,
            dur_ms,
        )
    return response


class AssistIn(BaseModel):
    query: str = Field(..., min_length=2, max_length=12000)
    category_slugs: list[str] = Field(
        default_factory=list,
        description="Рубрики Минздрава (slug), выбранные пользователем - усиливают отбор",
    )
    inline_clinical_detail: bool = Field(
        default=False,
        description="true - сразу развёрнутая выдержка (дольше); false - только кнопка загрузки",
    )
    assist_full: bool = Field(
        default=False,
        description="true - полный JSON (summary, differential); false - lite (только protocols, быстрее)",
    )
    retrieve_only: bool = Field(
        default=False,
        description="true - только RAG ranking без LLM (быстрый пошаговый поиск)",
    )
    icd_codes: list[str] = Field(
        default_factory=list,
        max_length=24,
        description="Явные коды МКБ из воронки - пропуск повторного подбора",
    )
    funnel_population: str | None = Field(
        default=None,
        max_length=32,
        description="adult|pediatric|pregnant|emergency из шага популяции",
    )
    icd_fast_path: bool = Field(
        default=False,
        description="true - сначала детерминированный lookup по индексу МКБ",
    )
    search_tier: str | None = Field(
        default=None,
        max_length=4,
        description="S0|S1|S2 - уровень поиска (S0=МКБ index, S1=retrieve, S2=LLM)",
    )


class SearchRunIn(BaseModel):
    """Единая точка: воронка (step>=0) или прямой поиск по tier S0/S1/S2."""

    query: str = Field(..., min_length=2, max_length=12000)
    tier: str = Field(default="S1", max_length=4)
    step: int = Field(
        default=-1,
        ge=-1,
        le=7,
        description="-1 = прямой поиск по tier; 0-7 = шаг воронки",
    )
    context: dict[str, Any] = Field(default_factory=dict)
    category_slugs: list[str] = Field(default_factory=list)
    icd_codes: list[str] = Field(default_factory=list, max_length=24)
    funnel_population: str | None = Field(default=None, max_length=32)
    session_id: str | None = Field(default=None, max_length=64)


class ProtocolsByIcdIn(BaseModel):
    query: str = Field(..., min_length=2, max_length=12000)
    icd_codes: list[str] = Field(..., min_length=1, max_length=24)
    population: str | None = Field(default=None, max_length=32)
    category_slugs: list[str] = Field(default_factory=list)
    limit: int = Field(default=8, ge=1, le=12)


class SearchFunnelIn(BaseModel):
    query: str = Field(..., min_length=2, max_length=12000)
    step: int = Field(default=0, ge=0, le=7)
    context: dict[str, Any] = Field(default_factory=dict)
    category_slugs: list[str] = Field(default_factory=list)
    session_id: str | None = Field(default=None, max_length=64)


class IcdSuggestIn(BaseModel):
    """Подбор кодов МКБ-10 по жалобам до полного поиска протоколов (шаг 1)."""

    query: str = Field(..., min_length=4, max_length=12000)


class OncoRiskIn(BaseModel):
    """Советующая оценка онконастороженности (decision-support, не диагноз, без send_gate)."""

    text: str = Field(..., min_length=2, max_length=20000)
    age: int | None = Field(default=None, ge=0, le=120)
    sex: str = Field(default="unknown", max_length=10)
    labs_text: str = Field(default="", max_length=8000)
    smoking: bool | None = None
    family_history: bool | None = None
    bmi: float | None = Field(default=None, ge=5.0, le=120.0)
    symptom_duration_known: bool = False
    adult_or_child: str = Field(default="adult", max_length=8)
    audience: str = Field(default="both", max_length=8, description="b2b | b2c | both")


class ProtocolDetailIn(BaseModel):
    """Развёрнутая выдержка по протоколу - отдельный запрос (после краткого ответа assist)."""

    query: str = Field(..., min_length=2, max_length=12000)
    path: str = Field(..., min_length=1, max_length=2048)
    title: str = Field(default="", max_length=2000)
    protocol_confidence: float | None = Field(
        default=None,
        description="Оценка соответствия из assist (0-1), для подписи в блоке",
    )
    extract_focus: str | None = Field(
        default=None,
        max_length=32,
        description="Узкий фокус: investigations, medications, treatment_methods, monitoring, algorithms",
    )
    client_rag_support: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="rag_support из assist для предупреждения о слабом отборе",
    )


class KzMatrixIn(BaseModel):
    """Матрица «что должно быть в КЗ» по протоколу."""

    query: str = Field(..., min_length=2, max_length=12000)
    path: str = Field(..., min_length=1, max_length=2048)
    title: str = Field(default="", max_length=2000)
    icd_codes: list[str] = Field(default_factory=list, max_length=24)
    clinical_detail: dict | None = Field(
        default=None,
        description="Уже загруженная выдержка - ускоряет fallback и подсказку модели",
    )


class ProtocolPracticalIn(BaseModel):
    """Развёрнутая выдержка + матрица КЗ одним запросом (после подбора протокола ≥80%)."""

    query: str = Field(..., min_length=2, max_length=12000)
    path: str = Field(..., min_length=1, max_length=2048)
    title: str = Field(default="", max_length=2000)
    icd_codes: list[str] = Field(default_factory=list, max_length=24)
    protocol_confidence: float | None = Field(default=None)
    client_rag_support: float | None = Field(default=None, ge=0.0, le=1.0)
    extract_focus: str | None = Field(default=None, max_length=32)
    skip_kz_matrix: bool = Field(
        default=False,
        description="Только clinical_detail (без второго вызова LLM для матрицы)",
    )
    mode: str = Field(
        default="lite",
        description="lite - rich-чанки без LLM; full - разбор моделью (медленнее)",
    )


class ProtocolPracticalSectionIn(BaseModel):
    """Один раздел практического разбора (препараты, лечение, …) без LLM."""

    query: str = Field(..., min_length=2, max_length=12000)
    path: str = Field(..., min_length=1, max_length=2048)
    title: str = Field(default="", max_length=2000)
    section: str = Field(
        ...,
        min_length=3,
        max_length=32,
        description="investigations, medications, treatment_methods, monitoring_frequency, care_algorithms",
    )
    icd_codes: list[str] = Field(default_factory=list, max_length=24)


class ConsultComplianceScreenIn(BaseModel):
    """L0-скрининг КЗ для МИС (текст или FHIR BY Bundle)."""

    text: str | None = Field(default=None, max_length=120000)
    bundle: dict | None = Field(default=None, description="FHIR BY Bundle document")
    consultation_id: str = Field(default="screen", max_length=64)
    methodist_mode: bool = Field(default=False, description="Режим методиста (требует X-Methodist-Token)")
    sandbox: bool = Field(default=False, description="Песочница - не в production-очереди")


class ConsultReviewJsonIn(BaseModel):
    """Полная проверка КЗ из структуры МИС / FHIR (без PDF)."""

    text: str | None = Field(default=None, max_length=200000)
    bundle: dict | None = Field(default=None, description="FHIR BY Bundle document")
    category_slugs: str = Field(default="", max_length=400)
    tier: str = Field(
        default="L2",
        max_length=8,
        description="Уровень проверки: L0 (скрининг), L1 (structured), L2 (полный RAG+LLM)",
    )
    methodist_mode: bool = Field(default=False)
    sandbox: bool = Field(default=False)
    l2_narrative: bool = Field(default=False, description="L2+: пояснение методиста")


class ConsultReviewTierIn(BaseModel):
    """Явный выбор уровня L0/L1/L2 для массового потока КЗ."""

    tier: str = Field(default="L0", max_length=8)
    text: str | None = Field(default=None, max_length=200000)
    bundle: dict | None = Field(default=None, description="FHIR BY Bundle document")
    consultation_id: str = Field(default="tier", max_length=64)
    category_slugs: str = Field(default="", max_length=400)
    methodist_mode: bool = Field(default=False)
    sandbox: bool = Field(default=False)
    l2_narrative: bool = Field(
        default=False,
        description="L2+: пояснение методиста (один вызов Flash поверх evidence pack)",
    )


class ConsultL2NarrativeIn(BaseModel):
    """Дополнительное пояснение L2+ по уже собранному evidence pack."""

    evidence_pack: dict = Field(default_factory=dict)
    block_gaps: list[dict] = Field(default_factory=list)
    structured_summary: str = Field(default="", max_length=4000)


class ConsultValidateBundleIn(BaseModel):
    """Проверка FHIR BY Bundle на готовность к ЦИСЗ (без клинического RAG)."""

    bundle: dict = Field(..., description="FHIR BY Bundle document")
    scenario: str = Field(default="auto", max_length=32)


class PatientReviewJsonIn(BaseModel):
    """B2C: проверка своего КЗ (tier P1/P2) - текст без файла."""

    text: str = Field(..., min_length=40, max_length=200000)
    age_years: int | None = Field(default=None, ge=1, le=120)
    sex: str | None = Field(default=None, max_length=16, description="male | female")
    clinic_id: str | None = Field(default=None, max_length=32)
    tier_id: str | None = Field(default=None, max_length=24)
    payment_token: str | None = Field(default=None, max_length=128)


class PatientAnalyticsIn(BaseModel):
    event: str = Field(..., min_length=2, max_length=48)
    clinic_id: str | None = Field(default=None, max_length=32)
    tier_id: str | None = Field(default=None, max_length=24)
    meta: dict | None = None
    text_hash: str | None = Field(default=None, max_length=64)


class PatientAccountSyncIn(BaseModel):
    session_token: str = Field(..., min_length=8, max_length=256)
    history: list[dict] = Field(default_factory=list)


class PatientPaymentSessionIn(BaseModel):
    tier_id: str = Field(default="basic", max_length=24)
    clinic_id: str | None = Field(default=None, max_length=32)


class PatientMonetizationPatchIn(BaseModel):
    monetization_enabled: bool | None = None
    payment_required: bool | None = None
    show_tier_picker: bool | None = None
    show_prices: bool | None = None
    default_tier_id: str | None = Field(default=None, max_length=24)
    enabled_tier_ids: list[str] | None = None
    demo_note_ru: str | None = Field(default=None, max_length=500)
    paid_note_ru: str | None = Field(default=None, max_length=500)
    value_banner_ru: str | None = Field(default=None, max_length=500)


class ConsultationTemplateIn(BaseModel):
    """Шаблон консультативного заключения по развёрнутой выдержке."""

    query: str = Field(..., min_length=2, max_length=12000)
    clinical_detail: dict = Field(
        ...,
        description="Объект clinical_detail из /api/assist или /api/protocol-detail",
    )
    refine: bool = Field(
        default=False,
        description="Повторная генерация: дополнить черновик по протоколу с учётом подстановок и примечаний",
    )
    previous_template: str | None = Field(
        default=None,
        max_length=120000,
        description="Черновик заключения (после подстановки данных пользователя в плейсхолдеры)",
    )
    additional_notes: str | None = Field(
        default=None,
        max_length=16000,
        description="Дополнительные сведения врача для доработки текста",
    )
    patient_context: str | None = Field(
        default=None,
        max_length=4000,
        description="Возраст, пол и др. из формы - для подстановки в жалобы/анамнез",
    )
    selected_facts_payload: dict | None = Field(
        default=None,
        description="Структурированные выбранные пользователем пункты (sections/items)",
    )


def _normalize_selected_facts_payload(raw: object) -> dict:
    out = {"selected_count": 0, "sections": []}
    if not isinstance(raw, dict):
        return out
    sections = raw.get("sections")
    if not isinstance(sections, list):
        return out
    norm_sections: list[dict] = []
    count = 0
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        title = str(sec.get("title") or "").strip()
        items_raw = sec.get("items")
        if not title or not isinstance(items_raw, list):
            continue
        items = [str(x).strip() for x in items_raw if str(x).strip()]
        if not items:
            continue
        count += len(items)
        norm_sections.append(
            {
                "key": str(sec.get("key") or "").strip(),
                "title": title[:120],
                "items": items[:40],
            }
        )
    out["selected_count"] = count
    out["sections"] = norm_sections
    return out


def _selected_facts_coverage(template_text: str, payload: dict) -> tuple[float, list[str]]:
    txt = (template_text or "").lower()
    if not txt:
        return 0.0, []
    sections = payload.get("sections") if isinstance(payload, dict) else []
    if not isinstance(sections, list) or not sections:
        return 1.0, []
    total = 0
    hit = 0
    missing: list[str] = []
    for sec in sections:
        items = sec.get("items") if isinstance(sec, dict) else None
        if not isinstance(items, list):
            continue
        for it in items:
            s = str(it).strip()
            if len(s) < 6:
                continue
            total += 1
            toks = [t for t in re.split(r"[\s,;:.()]+", s.lower()) if len(t) >= 5]
            toks = toks[:6]
            ok = False
            for tk in toks:
                if tk in txt:
                    ok = True
                    break
            if ok:
                hit += 1
            else:
                missing.append(s)
    if total <= 0:
        return 1.0, []
    return hit / float(total), missing[:20]


def _corpus_stats_from_index_csv() -> dict:
    """Сводка по index.csv (скачанный каталог PDF) без загрузки RAG-чанков."""
    p = INDEX_CSV_PATH
    if not p.is_file():
        return {"index_csv_available": False}
    rows: list[dict[str, str]] = []
    with p.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    post_mz = sum(1 for r in rows if (r.get("has_post_mz") or "").strip().lower() == "yes")
    years: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    for r in rows:
        y = (r.get("years_in_filename") or "").strip()
        if y:
            years[y] += 1
        cat = (r.get("category") or "").strip()
        if cat:
            categories[cat] += 1
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    categories_top = [
        {
            "slug": cat,
            "label": SPECIALTY_LABELS_RU.get(cat, cat.replace("-", " ").title()),
            "count": c,
        }
        for cat, c in categories.most_common(12)
    ]
    return {
        "index_csv_available": True,
        "protocols_in_index": len(rows),
        "protocols_post_mz": post_mz,
        "rubrics_in_index": len(categories),
        "index_csv_updated_utc": mtime.isoformat(),
        "years_top": [{"year": y, "count": c} for y, c in years.most_common(8)],
        "categories_top": categories_top,
        "source_url": MINZDRAV_PROTOCOLS_INDEX_URL,
    }


@app.get("/api/corpus-stats")
def api_corpus_stats() -> dict:
    """Состояние корпуса для UI: каталог index.csv + загруженные чанки (если RAG готов)."""
    out = _corpus_stats_from_index_csv()
    out["version"] = _app_version()
    out["specialties_catalog"] = len(SPECIALTY_LABELS_RU)
    out["rag_ready"] = _chunks_load_done.is_set()
    out["rag_load_error"] = public_error_text(_chunks_load_error)
    if _chunks_load_done.is_set():
        out["chunks_loaded"] = len(_chunks)
        out["protocols_loaded"] = len(_protocols_by_path)
        out["protocol_meta_entries"] = len(_protocol_meta)
        out["chunks_with_embedding"] = sum(
            1 for c in _chunks if isinstance(c.get("embedding"), list)
        )
        try:
            from clinical_knowledge.vector_index import index_stats

            out["vector_index"] = index_stats()
        except Exception:
            out["vector_index"] = {"loaded": False}
    else:
        out["chunks_loaded"] = None
        out["protocols_loaded"] = None
        out["protocol_meta_entries"] = None
    return out


@app.get("/api/training-cases")
def api_training_cases() -> dict:
    if not TRAINING_CASES_PATH.is_file():
        raise HTTPException(status_code=404, detail="data/training_cases.json не найден")
    try:
        data = json.loads(TRAINING_CASES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="training_cases.json: ожидается объект")
    return data


@app.get("/api/demo-consult-text")
def api_demo_consult_text() -> dict:
    """Текст демо-КЗ для показа (без PDF)."""
    if not DEMO_CONSULT_TEXT_PATH.is_file():
        raise HTTPException(status_code=404, detail="demo_consult_kz_sample.txt не найден")
    try:
        text = DEMO_CONSULT_TEXT_PATH.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"text": text, "title": "Демо: консультативное заключение (СКВ, обезличено)"}


@app.get("/api/presentation-stats")
def api_presentation_stats() -> dict:
    """Агрегаты корпуса, правил и protocol summaries для презентации MVP."""
    from clinical_knowledge.presentation_stats import build_presentation_stats_bundle

    corpus = _corpus_stats_from_index_csv()
    ck: dict = {}
    try:
        from clinical_knowledge import clinical_knowledge_status

        ck = clinical_knowledge_status()
    except Exception:
        pass
    quality = None
    if QUALITY_BENCHMARK_PATH.is_file():
        try:
            quality = json.loads(QUALITY_BENCHMARK_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    chunks_total = 0
    structured = 0
    try:
        chunks_total = len(_chunks)
        for ch in _chunks:
            if ch.get("section_title") or ch.get("section_path"):
                structured += 1
    except Exception:
        pass
    bundle = build_presentation_stats_bundle(
        corpus=corpus,
        version=_app_version(),
        clinical_knowledge=ck,
        quality_benchmark=quality,
        rag_version={
            "corpus_chunks": chunks_total,
            "corpus_structured_chunks": structured,
        },
    )
    snap = ROOT / "docs" / "presentation-stats.json"
    if snap.is_file():
        try:
            prev = json.loads(snap.read_text(encoding="utf-8"))
            if isinstance(prev, dict) and prev.get("kz_fixtures"):
                bundle["kz_fixtures"] = prev["kz_fixtures"]
        except (OSError, ValueError):
            pass
    return bundle


@app.get("/api/pilot-analytics-demo")
def api_pilot_analytics_demo() -> dict:
    """Агрегированная аналитика пилота (обезличенно, live из feedback)."""
    from clinical_knowledge.pilot_analytics_public import build_public_pilot_analytics

    corpus = _corpus_stats_from_index_csv()
    corpus["chunks_loaded"] = len(_chunks) if _chunks_load_done.is_set() else None
    return build_public_pilot_analytics(
        corpus=corpus,
        version=_app_version(),
        rag_ready=_chunks_load_done.is_set(),
    )


def _load_quality_benchmark_dict() -> dict:
    if not QUALITY_BENCHMARK_PATH.is_file():
        return {}
    try:
        data = json.loads(QUALITY_BENCHMARK_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


@app.get("/api/search-analytics")
def api_search_analytics() -> dict:
    """Статистика поиска протоколов для вкладки «Поиск» (корпус, эталон, телеметрия)."""
    from clinical_knowledge.lazy_rag_config import startup_mode
    from clinical_knowledge.search_analytics_public import build_public_search_analytics

    corpus = _corpus_stats_from_index_csv()
    corpus["chunks_loaded"] = len(_chunks) if _chunks_load_done.is_set() else None
    corpus["rag_ready"] = _chunks_load_done.is_set()
    corpus["startup_mode"] = startup_mode()
    if _path_manifest is not None:
        corpus["manifest_paths"] = len(_path_manifest.entries)
    elif _lazy_chunk_store is not None and getattr(_lazy_chunk_store, "manifest", None):
        corpus["manifest_paths"] = len(_lazy_chunk_store.manifest.entries)
    return build_public_search_analytics(
        corpus=corpus,
        quality=_load_quality_benchmark_dict(),
        version=_app_version(),
        rag_ready=_chunks_load_done.is_set(),
    )


@app.get("/api/quality-benchmark")
def api_quality_benchmark() -> dict:
    """Эталонные метрики качества подбора (для блока «качество поиска» на главной)."""
    if not QUALITY_BENCHMARK_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="data/quality_benchmark.json не найден",
        )
    try:
        data = json.loads(QUALITY_BENCHMARK_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"quality_benchmark.json: {e!s}") from e
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="quality_benchmark.json: ожидается объект")
    return data


_icd_ru_count_cache: int | None = None


def _icd_ru_entries_count() -> int:
    """Число записей русского справочника МКБ - читаем файл один раз и кэшируем."""
    global _icd_ru_count_cache
    if _icd_ru_count_cache is not None:
        return _icd_ru_count_cache
    n = 0
    try:
        n = len(
            json.loads(
                (ROOT / "data/icd_reference/icd10_ru_mkb10su.json").read_text(encoding="utf-8")
            )
        )
    except (OSError, ValueError, TypeError):
        n = 0
    _icd_ru_count_cache = n
    return n


# Версия сборки: меняйте при значимых изменениях, чтобы по сайту/ответам видеть, новый ли код развёрнут.
BUILD_VERSION = "2026-07-20-r3-item-grounding"


def _app_version() -> str:
    """Версия сборки: APP_VERSION из окружения или встроенная BUILD_VERSION."""
    return (os.environ.get("APP_VERSION") or BUILD_VERSION).strip() or BUILD_VERSION


@app.get("/health/live")
@app.head("/health/live", include_in_schema=False)
def health_live() -> dict:
    """Минимальный liveness для Render (без обхода индексов)."""
    return {"ok": True, "version": _app_version()}


@app.get("/health")
def health() -> dict:
    has_key = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    icd_ru_n = _icd_ru_entries_count()
    manifest_stats = None
    chunk_cache_stats = None
    if _path_manifest is not None:
        manifest_stats = _path_manifest.manifest_stats()
    store = _lazy_chunk_store
    if store is not None:
        chunk_cache_stats = store.cache_stats()
    from clinical_knowledge.lazy_rag_config import startup_mode

    payload = {
        "ok": True,
        "version": _app_version(),
        "startup_mode": startup_mode(),
        "manifest_paths": (manifest_stats or {}).get("paths"),
        "manifest_sha256": (manifest_stats or {}).get("manifest_sha256"),
        "chunk_store": chunk_cache_stats,
        "rag_ready": _chunks_load_done.is_set(),
        "rag_load_error": public_error_text(_chunks_load_error),
        "chunks": len(_chunks),
        "protocols": len(_protocols_by_path),
        "protocol_meta": len(_protocol_meta),
        "structured_index": len(_structured_by_path),
        "icd_ru_entries": icd_ru_n,
        "gemini_configured": has_key,
        "specialties_count": len(SPECIALTY_LABELS_RU),
        "memory_saver": _memory_saver_enabled(),
        "lex_bm25_alpha": float(os.environ.get("RAG_LEX_BM25_ALPHA", "0.55") or "0.55"),
        "lexical_max_chars": env_int("RAG_LEXICAL_MAX_CHARS", 0),
        "bm25_index": _bm25_index is not None,
        "embedding_rerank": os.environ.get("RAG_GEMINI_EMBED_RERANK", "1"),
        "embedding_model": os.environ.get(
            "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2-preview"
        ),
        "consult_rag_second_pass": _consult_rag_second_pass_enabled(),
        "consult_review_fast": _consult_review_fast_mode(),
        "consult_render_l2_lite": _consult_render_l2_lite_enabled(),
        "consult_render_l2_skip_llm": _consult_render_l2_skip_llm(),
        "consult_l2_fast": _consult_l2_fast_enabled(),
        "consult_l2_narrative": os.environ.get("CONSULT_L2_NARRATIVE", "0"),
        "consult_l2_mode": _consult_l2_mode_label(),
        "consult_l2_skip_rag_warm": _consult_l2_skip_rag_warm(),
        "consult_l2_latency": _consult_l2_feedback_latency(),
        "consult_review_profile": (
            "fast"
            if _consult_review_fast_mode()
            else (os.environ.get("CONSULT_REVIEW_PROFILE") or "full").strip().lower() or "full"
        ),
        "consult_review_timeout_sec": max(
            120,
            int(os.environ.get("CONSULT_REVIEW_CLIENT_TIMEOUT_SEC", "600") or "600"),
        ),
        "protocol_summary_rag_merge": os.environ.get("PROTOCOL_SUMMARY_RAG_MERGE", "1"),
        "rag_lex_max_candidates": env_int("RAG_LEX_MAX_CANDIDATES", 0),
        "rag_lex_max_union": env_int("RAG_LEX_MAX_UNION", 0),
        "rag_retrieve_concurrency": env_int("RAG_RETRIEVE_CONCURRENCY", 4),
        "lex_index_ready": _lex_inverted_index is not None,
        "consult_alignment_enabled": os.environ.get("CONSULT_ALIGNMENT_ENABLED", "1"),
        "consult_concurrency": env_int("CONSULT_CONCURRENCY", 3),
        "search_concurrency": env_int("SEARCH_CONCURRENCY", 4),
        "search_require_allowlist": os.environ.get("RAG_SEARCH_REQUIRE_ALLOWLIST_ON_RENDER", ""),
        "search_last": dict(_search_last_metrics),
        "render_plan": (os.environ.get("RENDER_PLAN") or "").strip() or None,
        "render_extended_ram": _render_extended_ram(),
    }
    try:
        from clinical_knowledge.vector_index import index_stats

        payload["vector_index"] = index_stats()
    except Exception:
        payload["vector_index"] = {"loaded": False}
    return payload


@app.get("/api/version")
def api_version() -> dict:
    """Лёгкий маршрут версии/готовности (для мониторинга и баннеров деплоя)."""
    chunks_total = 0
    structured = 0
    try:
        chunks_total = len(_chunks)
        for ch in _chunks:
            if ch.get("section_title") or ch.get("section_path"):
                structured += 1
    except Exception:
        pass
    payload = {
        "version": _app_version(),
        "rag_ready": _chunks_load_done.is_set(),
        "corpus_chunks": chunks_total,
        "corpus_structured_chunks": structured,
        "keep_struct": env_bool("RAG_KEEP_STRUCT", True),
        "consult_rich_context": env_bool("CONSULT_REVIEW_RICH_CONTEXT", True),
    }
    try:
        from clinical_knowledge import clinical_knowledge_status

        payload["clinical_knowledge"] = clinical_knowledge_status()
    except Exception:
        payload["clinical_knowledge"] = {"enabled": False}
    return payload


@app.get("/api/clinical-knowledge/benchmark")
def api_clinical_knowledge_benchmark() -> dict:
    """Эталонные метрики rule checker на consult_gold.jsonl (гастро MVP)."""
    bench_path = ROOT / "data" / "gastro_mvp" / "benchmark.json"
    if not bench_path.is_file():
        try:
            from clinical_knowledge.benchmark import run_gastro_gold_benchmark

            return {"ok": True, **run_gastro_gold_benchmark()}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}
    try:
        data = json.loads(bench_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail=str(e)[:200]) from e
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="benchmark.json: ожидается объект")
    return {"ok": True, **data}


@app.get("/api/clinical-knowledge/status")
def api_clinical_knowledge_status() -> dict:
    """Статус базы правил и структуризации каталога (все рубрики)."""
    try:
        from clinical_knowledge import clinical_knowledge_status

        return {"ok": True, **clinical_knowledge_status()}
    except Exception as e:
        return {"ok": False, "enabled": False, "error": str(e)[:200]}


@app.get("/api/clinical-knowledge/build-status")
def api_clinical_knowledge_build_status() -> dict:
    """Прогресс полной структуризации каталога (% PDF, conditions, rules)."""
    try:
        from clinical_knowledge.catalog_full_build import build_status_payload

        return {"ok": True, **build_status_payload()}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


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


_verify_key_cache: dict = {"ts": 0.0, "ok": None, "msg": "", "model": ""}
_verify_key_lock = threading.Lock()


def _verify_key_admin_guard(request: "Request") -> None:
    """Если задан API_ADMIN_TOKEN - требовать заголовок X-Admin-Token (по умолчанию открыт)."""
    expected = (os.environ.get("API_ADMIN_TOKEN") or "").strip()
    if not expected:
        return
    got = (request.headers.get("x-admin-token") or "").strip()
    if got != expected:
        raise HTTPException(status_code=401, detail="Требуется корректный X-Admin-Token.")


def _require_methodist_auth(request: "Request") -> None:
    from clinical_knowledge.feedback_store import is_methodist_authenticated, methodist_auth_enabled

    if not methodist_auth_enabled():
        raise HTTPException(status_code=503, detail="METHODIST_TOKEN не настроен на сервере.")
    if not is_methodist_authenticated(request.headers):
        raise HTTPException(status_code=403, detail="Требуется корректный X-Methodist-Token.")


def _methodist_request_active(request: "Request", body_flag: bool = False) -> bool:
    from clinical_knowledge.feedback_store import is_methodist_authenticated, methodist_auth_enabled

    if not methodist_auth_enabled():
        return False
    if is_methodist_authenticated(request.headers):
        return True
    return bool(body_flag) and is_methodist_authenticated(request.headers)


def _consult_text_from_screen_body(body: "ConsultComplianceScreenIn") -> str:
    if (body.text or "").strip():
        return body.text or ""
    if body.bundle:
        from clinical_knowledge.fhir_bundle_adapter import bundle_to_consultation_text

        return bundle_to_consultation_text(body.bundle)
    return ""


def _consult_onco_risk_advisory_enabled() -> bool:
    """B2B-advisory онконастороженности в ответе consult-review (по умолчанию вкл, см. render.yaml)."""
    return env_bool("CONSULT_ONCO_RISK_ADVISORY_ENABLED", True)


def _onco_demographics_from_text(text: str) -> tuple[int | None, str, str]:
    """Возраст/пол/категория из текста КЗ для онко-оценки (best-effort, без падений)."""
    age: int | None = None
    try:
        dob, _ = _consult_extract_date_of_birth(text)
        if dob:
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            age = max(0, age)
    except Exception:
        age = None
    low = (text or "").lower()
    if age is None:
        m = re.search(r"(\d{1,3})\s*(?:лет|год[ауы]?)\b", low)
        if m:
            try:
                a = int(m.group(1))
                if 0 < a < 130:
                    age = a
            except Exception:
                age = None
    sex = "unknown"
    if re.search(r"пол[:\s]*ж|женщин|женск|\bfemale\b", low):
        sex = "female"
    elif re.search(r"пол[:\s]*м|мужчин|мужск|\bmale\b", low):
        sex = "male"
    aoc = "child" if (age is not None and age < 18) else "adult"
    return age, sex, aoc


def _consult_attach_onco_risk(result: dict, full_text: str) -> None:
    """Добавляет советующий B2B-блок onco_risk в ответ (не gate, не диагноз)."""
    if not isinstance(result, dict) or not _consult_onco_risk_advisory_enabled():
        return
    text = (full_text or "").strip()
    if len(text) < 2:
        return
    try:
        from clinical_knowledge import onco_risk as orisk

        age, sex, aoc = _onco_demographics_from_text(text)
        assessment = orisk.assess(
            orisk.OncoInputs(text=text, age=age, sex=sex, adult_or_child=aoc)
        )
        result["onco_risk"] = orisk.to_b2b_payload(assessment)
    except Exception:
        pass


def _maybe_methodist_autolog(
    request: "Request",
    result: dict,
    *,
    tier: str,
    full_text: str,
    consultation_id: str = "",
    latency_ms: int | None = None,
    sandbox: bool = False,
    body_methodist_mode: bool = False,
    category_slugs: str = "",
) -> dict:
    _consult_attach_onco_risk(result, full_text)
    if not _methodist_request_active(request, body_methodist_mode):
        return result
    from clinical_knowledge.feedback_store import enrich_result_with_methodist_autolog

    reviewer = (request.headers.get("x-methodist-reviewer") or "").strip()
    return enrich_result_with_methodist_autolog(
        result,
        tier=tier,
        full_text=full_text,
        consultation_id=consultation_id,
        latency_ms=latency_ms,
        sandbox=sandbox,
        reviewer=reviewer,
        category_slugs=category_slugs,
    )


@app.get("/api/verify-key")
def verify_key(request: "Request") -> dict:
    """Один тестовый запрос к модели - проверка ключа из .env (кэш + опц. admin-токен)."""
    _verify_key_admin_guard(request)
    if _verify_gemini_key is None:
        raise HTTPException(
            status_code=501,
            detail="Модуль проверки ключа API не найден",
        )
    ttl = env_int("VERIFY_KEY_CACHE_TTL", 300)
    now = time.time()
    with _verify_key_lock:
        cached_ok = _verify_key_cache["ok"]
        if cached_ok is not None and (now - _verify_key_cache["ts"]) < ttl:
            if not cached_ok:
                raise HTTPException(status_code=502, detail=_verify_key_cache["msg"])
            return {
                "ok": True,
                "reply_preview": _verify_key_cache["msg"],
                "model": _verify_key_cache["model"],
                "cached": True,
            }
    ok, msg = _verify_gemini_key()
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    with _verify_key_lock:
        _verify_key_cache.update({"ts": now, "ok": ok, "msg": msg, "model": model})
    if not ok:
        raise HTTPException(status_code=502, detail=msg)
    return {"ok": True, "reply_preview": msg, "model": model, "cached": False}


def _infer_icd_pipeline_from_full_query(
    full_query: str,
    model,
    *,
    skip_query_refine: bool = False,
    skip_icd_gemini: bool = False,
    force_icd_gemini: bool = False,
) -> tuple[dict | None, str, str, dict | None, str | None]:
    """Та же цепочка МКБ, что в начале api_assist (до retrieve).

    Возвращает:
      icd_analysis, q_эффективный, q_rag, query_clinical_refinement | None, сообщение_об_ошибке | None
    """
    q = (full_query or "").strip()
    q_rag = clinical_query_for_rag(q)
    q_rag_lexicon = q_rag
    if not q_rag:
        return (
            None,
            q,
            q_rag,
            None,
            "Пустой текст жалобы - заполните блок «Жалобы и вопрос»",
        )
    lq_lex = strip_funnel_context_lines(q_rag_lexicon)
    hints_confident = clinical_hints_confident(lq_lex)
    query_clinical_refinement: dict | None = None
    if (
        not skip_query_refine
        and not hints_confident
        and os.environ.get("RAG_GEMINI_QUERY_REFINE", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        )
    ):
        q_rag_new, rmeta = refine_clinical_query_gemini(q_rag, q, model)
        if rmeta is not None:
            q_rag = q_rag_new
            q = apply_clinical_correction(q, q_rag)
            query_clinical_refinement = rmeta
    icd_analysis = analyze_query_for_icd(q, q_rag, lexicon_query=q_rag_lexicon)
    pre_icd_infer_on = os.environ.get("RAG_ICD_PRE_RETRIEVE_INFER", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    should_gemini = (
        not skip_icd_gemini
        and pre_icd_infer_on
        and not icd_analysis.get("explicit_icd_in_query")
        and (force_icd_gemini or not (icd_analysis.get("detected") or []))
    )
    if should_gemini:
        _refine_icd_analysis_with_gemini(
            q_rag, icd_analysis, model, lexicon_query=q_rag_lexicon
        )
    from icd_mkb import finalize_icd_analysis_codes

    finalize_icd_analysis_codes(icd_analysis, lq_lex)
    # Детерминированный expander после МКБ: улучшает retrieve, не ломая lexicon/hints.
    q_rag, expand_meta = expand_query_for_retrieve(q_rag)
    if expand_meta and query_clinical_refinement is None:
        query_clinical_refinement = {
            "applied": True,
            "source": "deterministic_expand",
            "profiles": expand_meta.get("profiles") or [],
            "extra_terms": expand_meta.get("extra_terms") or [],
        }
    elif expand_meta and isinstance(query_clinical_refinement, dict):
        query_clinical_refinement["deterministic_expand"] = expand_meta
    # S2 shadow: локальный роутер query->специальность/ICD-глава логируется рядом с
    # фактическим путём (Gemini/эвристика), не влияя на ответ. Флаг RAG_QUERY_ROUTER_SHADOW.
    try:
        from clinical_knowledge.query_router import log_shadow as _router_log_shadow

        _router_log_shadow(
            q_rag,
            gemini_used=bool(should_gemini),
            detected_codes=[str(c) for c in (icd_analysis.get("codes_for_retrieval") or [])],
        )
    except Exception:
        pass
    return icd_analysis, q, q_rag, query_clinical_refinement, None


def _format_icd_append_line(icd_analysis: dict) -> str | None:
    """Строка для добавления в поле запроса перед поиском протокола."""
    codes = icd_analysis.get("codes_for_retrieval") or []
    if not codes:
        return None
    by_code: dict[str, dict] = {}
    for bucket in (icd_analysis.get("detected") or [], icd_analysis.get("suggested") or []):
        for row in bucket:
            if not isinstance(row, dict):
                continue
            c = normalize_icd_code(str(row.get("code") or ""))
            if c:
                by_code[c] = row
    parts: list[str] = []
    for raw in codes[:8]:
        c = normalize_icd_code(str(raw))
        if not c:
            continue
        row = by_code.get(c) or {}
        tr = (row.get("title_ru") or "").strip()
        if tr:
            parts.append(f"{c} ({tr})")
        else:
            parts.append(c)
    if not parts:
        return None
    return "МКБ-10 для поиска протокола: " + "; ".join(parts)


def _merge_explicit_icd_into_analysis(
    icd_analysis: dict,
    explicit_codes: list[str] | None,
) -> dict:
    """Подставляет коды из воронки в icd_analysis (без повторного LLM)."""
    if not explicit_codes:
        return icd_analysis
    from icd_mkb import normalize_icd_code

    out = dict(icd_analysis or {})
    codes = [
        normalize_icd_code(str(c))
        for c in explicit_codes
        if normalize_icd_code(str(c))
    ]
    if not codes:
        return out
    out["explicit_icd_in_query"] = True
    out["codes_for_retrieval"] = list(
        dict.fromkeys(codes + list(out.get("codes_for_retrieval") or []))
    )
    detected = list(out.get("detected") or [])
    seen = {str(r.get("code") or "").upper() for r in detected if isinstance(r, dict)}
    for c in codes:
        if c.upper() not in seen:
            detected.append({"code": c, "title_ru": "", "confidence": "funnel"})
    out["detected"] = detected
    return out


def _normalize_icd_code_list(codes: list[str] | None) -> list[str]:
    from icd_mkb import normalize_icd_code

    out: list[str] = []
    seen: set[str] = set()
    for raw in codes or []:
        nc = normalize_icd_code(str(raw))
        if nc and nc not in seen:
            seen.add(nc)
            out.append(nc)
    return out


def _icd_codes_for_fast_lookup(
    *,
    body_codes: list[str] | None,
    icd_analysis: dict,
) -> list[str]:
    """Коды для ICD lookup: явные из воронки, иначе из текста запроса (codes_for_retrieval)."""
    explicit = _normalize_icd_code_list(body_codes)
    if explicit:
        return explicit
    return _normalize_icd_code_list(icd_analysis.get("codes_for_retrieval"))


def _icd_fast_auto_enabled() -> bool:
    return os.environ.get("RAG_ICD_FAST_AUTO", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _search_skip_llm_on_retrieve_only() -> bool:
    return os.environ.get("RAG_SEARCH_SKIP_LLM_ON_RETRIEVE_ONLY", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _build_search_timing(**fields: object) -> dict[str, int | str]:
    out: dict[str, int | str] = {}
    for key, val in fields.items():
        if val is None:
            continue
        if key.endswith("_ms") and isinstance(val, (int, float)):
            out[key] = int(round(float(val)))
        else:
            out[key] = val  # type: ignore[assignment]
    return out


def _try_icd_fast_assist(
    *,
    query: str,
    icd_codes: list[str],
    population: str | None,
    category_slugs: list[str] | None,
    icd_analysis: dict,
) -> dict | None:
    """Мгновенный ответ из индекса МКБ; None - нужен RAG fallback."""
    if not icd_codes:
        return None
    from clinical_knowledge.protocol_icd_index import (
        format_assist_payload,
        icd_fast_lookup_trusted,
        lookup_protocols_by_icd,
    )

    lookup = lookup_protocols_by_icd(
        icd_codes=icd_codes,
        query=query,
        population=population,
        rubric_slugs=category_slugs,
    )
    if not lookup.get("protocols"):
        return None
    if not icd_fast_lookup_trusted(query, lookup, icd_codes=icd_codes):
        return None
    return format_assist_payload(
        query=query,
        lookup_result=lookup,
        icd_analysis=icd_analysis,
    )


def _record_search_metrics(**fields: object) -> None:
    global _search_last_metrics
    snap = {k: v for k, v in fields.items() if v is not None}
    with _search_metrics_lock:
        _search_last_metrics = snap


@app.post("/api/assist")
def api_assist(body: AssistIn) -> dict:
    with _search_sem:
        return _api_assist_impl(body)


def _api_assist_impl(body: AssistIn) -> dict:
    from clinical_knowledge.search_tiering import (
        apply_search_tier_flags,
        build_search_path_allowlist,
        resolve_search_tier,
        search_require_allowlist,
    )

    t_start = time.perf_counter()
    _require_rag_loaded()
    model = get_gemini()

    tier_raw = (body.search_tier or "").strip() or None
    if not tier_raw and body.retrieve_only and not body.assist_full:
        tier_raw = "S1"
    elif not tier_raw and not body.retrieve_only:
        tier_raw = "S2"
    tier_flags = apply_search_tier_flags(
        resolve_search_tier(tier_raw),
        explicit_icd_codes=body.icd_codes or None,
        query=body.query,
    )
    if tier_flags.get("error"):
        raise HTTPException(status_code=400, detail=str(tier_flags["error"]))
    search_tier = str(tier_flags.get("tier") or "S1")

    if search_tier == "S2":
        retrieve_only = False
    elif search_tier in ("S0", "S1"):
        retrieve_only = True
    else:
        retrieve_only = bool(body.retrieve_only)
    assist_lite = _assist_lite_enabled(assist_full=body.assist_full)
    if search_tier != "S2":
        assist_lite = True
    skip_llm_search = retrieve_only and _search_skip_llm_on_retrieve_only()
    icd_fast = bool(body.icd_fast_path) or bool(tier_flags.get("icd_fast_path"))
    skip_refine = skip_llm_search or icd_fast or bool(body.icd_codes)
    skip_icd_gemini = skip_llm_search or icd_fast or bool(body.icd_codes)
    icd_analysis, q, q_rag, query_clinical_refinement, icd_err = (
        _infer_icd_pipeline_from_full_query(
            body.query,
            model,
            skip_query_refine=skip_refine,
            skip_icd_gemini=skip_icd_gemini,
        )
    )
    t_icd_done = time.perf_counter()
    if icd_err:
        raise HTTPException(status_code=400, detail=icd_err)
    assert icd_analysis is not None
    icd_analysis = _merge_explicit_icd_into_analysis(icd_analysis, body.icd_codes or None)
    icd_codes_for_lex = icd_analysis.get("codes_for_retrieval") or None

    fast_icd_codes = _icd_codes_for_fast_lookup(
        body_codes=body.icd_codes or None,
        icd_analysis=icd_analysis,
    )
    use_icd_fast = icd_fast or (_icd_fast_auto_enabled() and bool(fast_icd_codes))
    if search_tier == "S0":
        use_icd_fast = True

    if use_icd_fast and fast_icd_codes:
        fast = _try_icd_fast_assist(
            query=q,
            icd_codes=fast_icd_codes,
            population=body.funnel_population,
            category_slugs=body.category_slugs or None,
            icd_analysis=icd_analysis,
        )
        if fast:
            fast["icd"] = _icd_client_payload(icd_analysis)
            fast["search_timing"] = _build_search_timing(
                path="icd_fast_lookup",
                total_ms=(time.perf_counter() - t_start) * 1000,
                icd_pipeline_ms=(t_icd_done - t_start) * 1000,
                lookup_ms=fast.get("lookup_ms"),
            )
            fast["search_tier"] = search_tier
            _record_search_metrics(
                search_path="icd_fast_lookup",
                search_tier=search_tier,
                total_ms=fast["search_timing"].get("total_ms"),
                retrieve_candidates=0,
            )
            try:
                from clinical_knowledge.protocol_nav_cache import attach_protocol_nav_map
                from clinical_knowledge.structured_retrieval_excerpt import attach_structured_excerpts

                nav_limit = _embed_protocol_nav_limit()
                if nav_limit > 0:
                    attach_protocol_nav_map(
                        fast,
                        query=q,
                        icd_codes=list(icd_analysis.get("codes_for_retrieval") or []),
                        limit=nav_limit,
                    )
                attach_structured_excerpts(fast, fast.get("retrieval") or [], limit=4)
            except Exception:
                pass
            return fast
        if search_tier == "S0" or tier_flags.get("require_icd_fast"):
            raise HTTPException(
                status_code=400,
                detail="S0: не найдены протоколы по указанным кодам МКБ. Уточните код или выберите S1/S2.",
            )

    icd_lookup_allowlist: list[str] | None = None
    if use_icd_fast and fast_icd_codes:
        from clinical_knowledge.protocol_icd_index import lookup_protocols_by_icd

        lk = lookup_protocols_by_icd(
            icd_codes=fast_icd_codes,
            query=body.query,
            population=body.funnel_population,
            rubric_slugs=body.category_slugs or None,
        )
        icd_lookup_allowlist = lk.get("path_allowlist") or None

    user_slugs = [
        s
        for s in (body.category_slugs or [])
        if isinstance(s, str) and s in ALLOWED_SPECIALTY_SLUGS
    ]
    search_ctx: dict | None = None
    path_allowlist: list[str] | None = None
    if icd_codes_for_lex or q:
        from clinical_knowledge.search_retrieval import build_protocol_search_context

        search_ctx = build_protocol_search_context(
            query=q,
            icd_codes=icd_codes_for_lex,
            category_slugs=user_slugs or None,
        )
        if search_ctx.get("expanded_icd_codes"):
            icd_codes_for_lex = search_ctx["expanded_icd_codes"]
        path_allowlist = search_ctx.get("path_allowlist")
    if icd_lookup_allowlist:
        merged_allow = list(
            dict.fromkeys((path_allowlist or []) + list(icd_lookup_allowlist))
        )[:15]
        path_allowlist = merged_allow or icd_lookup_allowlist[:15]
    from clinical_knowledge.search_clinical_routing import (
        detect_clinical_route_ids,
        expand_slugs_for_clinical_routes,
    )

    _clinical_routes = detect_clinical_route_ids(q, icd_codes_for_lex or [])
    _pregnant_ctx = (
        infer_audience_from_funnel_context(q) == "pregnant" or "pregnancy" in _clinical_routes
    )
    if user_slugs or icd_codes_for_lex or _pregnant_ctx or _clinical_routes or skip_llm_search:
        query_specialties = []
    else:
        query_specialties = infer_specialties_gemini(q, model)
    boost_merged = list(dict.fromkeys((query_specialties or []) + user_slugs))
    if _clinical_routes:
        route_slugs = sorted(
            expand_slugs_for_clinical_routes(set(boost_merged), q, icd_codes_for_lex or [])
        )
        boost_merged = list(dict.fromkeys(route_slugs + boost_merged))
    if _pregnant_ctx:
        boost_merged = [
            s
            for s in boost_merged
            if s
            not in (
                "nevrologiya-neyrokhirurgiya",
                "psikhiatriya-narkologiya",
            )
        ]
        if "akusherstvo-ginekologiya" not in boost_merged:
            boost_merged = ["akusherstvo-ginekologiya"] + boost_merged
    icd_path_boost: list[str] | None = None
    if icd_codes_for_lex:
        from clinical_knowledge.protocol_summary.icd_index import find_catalog_paths_by_icd_codes

        icd_path_boost = find_catalog_paths_by_icd_codes(icd_codes_for_lex, limit=8) or None
    if search_ctx and search_ctx.get("path_boost"):
        icd_path_boost = list(
            dict.fromkeys((icd_path_boost or []) + list(search_ctx["path_boost"]))
        ) or None

    require_allow = search_require_allowlist()
    merged_allowlist = build_search_path_allowlist(
        path_allowlist=path_allowlist,
        icd_lookup_allowlist=icd_lookup_allowlist,
        icd_codes=icd_codes_for_lex,
        path_boost=icd_path_boost,
        search_ctx=search_ctx,
    )

    def _assist_retrieve(rag_q: str, *, routing_q: str | None = None, strict_paths: bool = True) -> list[dict]:
        aud_hint = infer_audience_from_funnel_context(q)
        if aud_hint is None:
            aud_hint = infer_audience_from_query(q, _routing)
        allow = merged_allowlist if strict_paths else None
        if require_allow and strict_paths and not allow:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Уточните код МКБ-10 или выберите рубрику каталога - "
                    "полный поиск по корпусу отключён для стабильности сервера."
                ),
            )
        skip_embed = bool(
            allow
            and len(allow) <= 15
            and (use_icd_fast or icd_lookup_allowlist or retrieve_only)
        )
        rows = retrieve(
            rag_q,
            routing_query=routing_q if routing_q is not None else q,
            category_boost=boost_merged or None,
            user_category_slugs=user_slugs or None,
            icd_codes_for_lex=icd_codes_for_lex,
            path_boost=icd_path_boost,
            path_allowlist=allow,
            catalog_path_extra=(search_ctx or {}).get("path_boost"),
            audience_hint=aud_hint,
            embed_rerank=False if skip_embed else None,
        )
        if not rows and allow and not require_allow:
            rows = retrieve(
                rag_q,
                routing_query=routing_q if routing_q is not None else q,
                category_boost=boost_merged or None,
                user_category_slugs=user_slugs or None,
                icd_codes_for_lex=icd_codes_for_lex,
                path_boost=icd_path_boost,
                path_allowlist=None,
                catalog_path_extra=(search_ctx or {}).get("path_boost"),
                audience_hint=aud_hint,
            )
            rows, _, _ = filter_retrieval_by_audience(rows, q, _routing)
        return rows

    t_retrieve_start = time.perf_counter()
    retrieved = _assist_retrieve(q_rag)
    query_spelling_correction: dict | None = None
    if (
        not retrieved
        and not skip_llm_search
        and not require_allow
        and os.environ.get("RAG_SPELLFIX_ON_EMPTY", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        )
    ):
        fixed, changed = fix_query_spelling_medical(q_rag, model)
        if changed and fixed.strip():
            q_llm = apply_clinical_correction(q, fixed)
            retrieved = _assist_retrieve(fixed, routing_q=q_llm)
            if retrieved:
                q = q_llm
                query_spelling_correction = {
                    "applied": True,
                    "rag_query_before": q_rag,
                    "rag_query_after": fixed,
                }

    if not retrieved:
        raise HTTPException(status_code=400, detail="Пустой отбор - уточните запрос")

    retrieved, audience_inferred, audience_fallback = filter_retrieval_by_audience(
        retrieved, q, _routing
    )
    if not retrieved and search_ctx and (
        infer_audience_from_funnel_context(q) or infer_audience_from_query(q, _routing)
    ) and not require_allow:
        retrieved = _assist_retrieve(q_rag, strict_paths=False)
        retrieved, audience_inferred, audience_fallback = filter_retrieval_by_audience(
            retrieved, q, _routing
        )
    if not retrieved:
        raise HTTPException(status_code=400, detail="Пустой отбор - уточните запрос")

    chunk_vote_majority: str | None = None
    if retrieved and os.environ.get("RAG_CHUNK_VOTE_RERETRIEVE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    ) and not require_allow:
        maj = _majority_category_from_retrieval(retrieved)
        chunk_vote_majority = maj
        if maj and (not boost_merged or maj not in boost_merged):
            boost2 = [maj] + [x for x in (boost_merged or []) if x != maj]
            r2 = retrieve(
                q_rag,
                routing_query=q,
                category_boost=boost2,
                user_category_slugs=user_slugs or None,
                icd_codes_for_lex=icd_codes_for_lex,
                path_boost=icd_path_boost,
                audience_hint=infer_audience_from_funnel_context(q)
                or infer_audience_from_query(q, _routing),
            )
            if r2:
                retrieved, audience_inferred, audience_fallback = (
                    filter_retrieval_by_audience(r2, q, _routing)
                )

    retrieve_only = bool(body.retrieve_only)

    if not retrieve_only:
        maybe_refine_icd_with_gemini_after_retrieve(
            model,
            q_rag,
            icd_analysis,
            retrieved,
        )

    if retrieve_only:
        assist_lite = True
        retrieved = dedupe_retrieval_by_basename(retrieved, prefer_slugs=user_slugs)
        protos = _build_protocols_from_retrieval(
            retrieved,
            prefer_slugs=user_slugs,
            icd_codes=icd_codes_for_lex,
        )
        protos = _filter_protocols_by_funnel_audience(protos, q)
        parsed: dict | None = {"protocols": protos}
        text = ""
        finish = "RETRIEVE_ONLY"
        retry_used = False
        if parsed and protos:
            apply_protocol_confidence_calibration(parsed, retrieved)
            dedupe_parsed_protocols(parsed, prefer_slugs=user_slugs)
            if isinstance(parsed.get("protocols"), list) and parsed["protocols"]:
                parsed["protocols"] = _rerank_protocols_symptom_only(
                    parsed["protocols"], q, icd_analysis
                )
                parsed["protocols"] = _filter_protocols_by_funnel_audience(
                    parsed["protocols"], q
                )
    else:
        lines: list[str] = []
        meta_specs: list[str] = []
        for i, r in enumerate(retrieved, 1):
            cat = ""
            p = r["path"]
            if p in _protocols_by_path:
                cat = _protocols_by_path[p].get("category") or ""
            pm = _protocol_meta.get(p)
            if pm and pm.get("specialty_ru"):
                meta_specs.append(pm["specialty_ru"])
            sc = r.get("score")
            lx = r.get("lexical_score")
            rm = r.get("routing_multiplier")
            lines.append(
                f"[{i}] path={p}\n"
                f"рубрика={cat}\n"
                f"тип_фрагмента={r['kind']}\n"
                f"score={sc} lexical_score={lx} routing_multiplier={rm}\n"
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
        icd_block = _icd_block_for_prompt(icd_analysis)
        if icd_block:
            icd_block = icd_block + "\n\n"
        user_block = (
            icd_block
            + hint_block
            + f"Запрос пользователя:\n{q}\n\nФрагменты протоколов:\n{context}\n\n"
            + ASSIST_USER_CONTEXT_GUIDE
        )
        full_prompt = (
            (SYSTEM_JSON_LITE if assist_lite else SYSTEM_JSON)
            + "\n\n---\n\n"
            + user_block
        )
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
            retry_prompt = (
                (SYSTEM_JSON_LITE_RETRY if assist_lite else SYSTEM_JSON_RETRY)
                + "\n\n---\n\n"
                + user_block
            )
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

        if parsed and isinstance(parsed, dict):
            apply_protocol_confidence_calibration(parsed, retrieved)
            dedupe_parsed_protocols(parsed, prefer_slugs=user_slugs)
            if assist_lite:
                _strip_assist_verbose_fields(parsed)

    if not retrieve_only:
        retrieved = dedupe_retrieval_by_basename(retrieved, prefer_slugs=user_slugs)

    icd_payload = _icd_client_payload(icd_analysis)
    diag_mode = _diagnostic_mode_summary(icd_payload, retrieved)
    if not assist_lite:
        _ensure_symptom_followup_questions(
            parsed,
            str(diag_mode.get("mode") or ""),
            float(diag_mode.get("confidence") or 0.0),
        )
    if parsed and isinstance(parsed, dict):
        merged_icd: list[dict] = []
        for it in icd_payload.get("detected") or []:
            merged_icd.append(dict(it))
        for it in icd_payload.get("suggested") or []:
            merged_icd.append(dict(it))
        parsed["icd_codes"] = merged_icd

    confidence_second_pass_used = False
    if parsed and isinstance(parsed, dict) and os.environ.get(
        "RAG_CONFIDENCE_SECOND_PASS", "0"
    ).strip().lower() in ("1", "true", "yes"):
        confidence_second_pass_used = bool(
            refine_protocol_confidences_gemini(model, q, parsed, retrieved)
        )

    clinical_detail = None
    clinical_detail_offer: dict | None = None
    inline_detail = body.inline_clinical_detail or os.environ.get(
        "RAG_ASSIST_INLINE_DETAIL", "0"
    ).strip().lower() in ("1", "true", "yes")
    if (
        parsed
        and inline_detail
        and os.environ.get("GEMINI_EXTRACT_FULL_MATCH", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        )
    ):
        candidates: list[tuple[float, dict]] = []
        min_detail_rag = float(os.environ.get("RAG_DETAIL_MIN_RAG_SUPPORT", "0.12"))
        for pr in parsed.get("protocols") or []:
            if not confidence_for_detailed_extraction(pr.get("confidence_score")):
                continue
            if float(pr.get("rag_support") or 0.0) < min_detail_rag:
                continue
            raw_p = str(pr.get("path") or "")
            pth = raw_p if raw_p in _chunks_by_path else ""
            if not pth:
                nk = _normalize_protocol_path_key(raw_p)
                if nk in _chunks_by_path:
                    pth = nk
            if not pth:
                continue
            sc = _confidence_numeric(pr.get("confidence_score")) or 0.0
            candidates.append((sc, pr))
        candidates.sort(key=lambda x: -x[0])
        if candidates:
            best_sc, pr = candidates[0]
            pth = pr.get("path") or ""
            score_obj = pr.get("confidence_score")
            if confidence_display_full(score_obj):
                _rs = pr.get("rag_support")
                _rs_f: float | None = None
                if _rs is not None:
                    try:
                        _rs_f = float(_rs)
                    except (TypeError, ValueError):
                        _rs_f = None
                clinical_detail = extract_clinical_detail(
                    pth,
                    q,
                    str(pr.get("title") or ""),
                    model,
                    detailed=True,
                    protocol_confidence=best_sc,
                    client_rag_support=_rs_f,
                )
            else:
                clinical_detail_offer = {
                    "path": pth,
                    "title": str(pr.get("title") or ""),
                    "confidence_score": best_sc,
                    "rag_support": pr.get("rag_support"),
                }

    if not assist_lite:
        normalize_differential_field(parsed)

    proto_list = (parsed.get("protocols") or []) if parsed else []
    icd_for_focus = icd_analysis.get("codes_for_retrieval") or []
    protocol_icd_mentions = _protocol_icd_mentions_for_response(
        proto_list,
        top_n=5,
        focus_codes=icd_for_focus if icd_for_focus else None,
    )
    red_flags = _red_flags_from_retrieval(retrieved)
    from clinical_knowledge.clinical_attention import build_clinical_attention

    clinical_attention = build_clinical_attention(
        query=q,
        proto_list=proto_list if isinstance(proto_list, list) else [],
        red_flags=red_flags,
        audience_inferred=audience_inferred,
        diagnostic_notice=diag_mode.get("notice"),
        diagnostic_mode=diag_mode.get("mode"),
    )

    meta_paths: set[str] = set()
    for r in retrieved:
        p = str(r.get("path") or "").strip()
        if p:
            meta_paths.add(p)
    for pr in proto_list:
        if isinstance(pr, dict):
            p = str(pr.get("path") or "").strip()
            if p:
                meta_paths.add(p)

    try:
        from clinical_knowledge.search_telemetry import log_protocol_search_from_payload

        log_protocol_search_from_payload(
            query=q,
            payload={
                "llm_json": {"protocols": proto_list if isinstance(proto_list, list) else []},
                "icd": icd_analysis,
                "retrieved_count": len(retrieved),
            },
            icd_codes=list(icd_analysis.get("codes_for_retrieval") or []),
            user_slugs=user_slugs,
            audience_inferred=audience_inferred,
            search_source="assist",
        )
    except Exception:
        pass

    t_total = time.perf_counter()
    search_path = "retrieve_only" if retrieve_only else ("assist_lite" if assist_lite else "assist_llm")
    if search_tier == "S0":
        search_path = "icd_fast_lookup"
    search_timing = _build_search_timing(
        path=search_path,
        total_ms=(t_total - t_start) * 1000,
        icd_pipeline_ms=(t_icd_done - t_start) * 1000,
        retrieve_ms=(t_total - t_retrieve_start) * 1000,
    )
    _record_search_metrics(
        search_path=search_path,
        search_tier=search_tier,
        total_ms=search_timing.get("total_ms"),
        retrieve_candidates=len(retrieved),
    )

    out = {
        "query": q,
        "search_tier": search_tier,
        "retrieval": retrieved,
        "protocol_ui_meta": protocol_ui_meta_bundle(meta_paths),
        "audience_inferred": audience_inferred,
        "retrieval_audience_fallback": audience_fallback,
        "query_specialties": query_specialties,
        "user_category_slugs": user_slugs,
        "icd": icd_payload,
        "diagnostic_mode": diag_mode.get("mode"),
        "diagnostic_confidence": diag_mode.get("confidence"),
        "diagnostic_notice": diag_mode.get("notice"),
        "llm_text": text,
        "llm_json": parsed,
        "gemini_finish_reason": finish,
        "gemini_retry_used": retry_used,
        "clinical_detail": clinical_detail,
        "clinical_detail_offer": clinical_detail_offer,
        "query_spelling_correction": query_spelling_correction,
        "query_clinical_refinement": query_clinical_refinement,
        "retrieval_embedding": dict(_get_retrieval_embed_meta() or {"used": False}),
        "red_flags": red_flags,
        "clinical_attention": clinical_attention,
        "protocol_icd_mentions": protocol_icd_mentions,
        "routing_version": int(_routing.get("version", 1)) if _routing else 1,
        "chunk_vote_majority": chunk_vote_majority,
        "confidence_second_pass_used": confidence_second_pass_used,
        "assist_lite": assist_lite,
        "retrieve_only": retrieve_only,
        "search_timing": search_timing,
    }
    try:
        from clinical_knowledge.protocol_nav_cache import attach_protocol_nav_map
        from clinical_knowledge.structured_retrieval_excerpt import attach_structured_excerpts

        nav_limit = _embed_protocol_nav_limit()
        if nav_limit > 0:
            attach_protocol_nav_map(
                out,
                query=q,
                icd_codes=list(icd_analysis.get("codes_for_retrieval") or []),
                limit=nav_limit,
            )
        attach_structured_excerpts(out, retrieved, limit=4)
    except Exception:
        pass
    return out


@app.post("/api/protocol-detail")
def api_protocol_detail(body: ProtocolDetailIn) -> dict:
    """Развёрнутая выдержка по одному протоколу (второй вызов модели) - по кнопке после краткого ответа."""
    _require_rag_loaded()
    q = body.query.strip()
    pth = body.path.strip()
    if not pth or pth not in _chunks_by_path:
        raise HTTPException(
            status_code=404,
            detail="Протокол не найден в индексе",
        )
    model = get_gemini()
    pc = body.protocol_confidence
    if pc is not None:
        try:
            pc = float(max(0.0, min(1.0, pc)))
        except (TypeError, ValueError):
            pc = None
    clinical_detail = extract_clinical_detail(
        pth,
        q,
        body.title.strip(),
        model,
        detailed=True,
        protocol_confidence=pc,
        extract_focus=body.extract_focus,
        client_rag_support=body.client_rag_support,
    )
    return {"clinical_detail": clinical_detail}


@app.get("/api/protocol-pdf")
def api_protocol_pdf(path: str = Query(..., min_length=8, max_length=512)) -> FileResponse:
    """Отдаёт PDF протокола из minzdrav_protocols/ (каталог закрыт от StaticFiles)."""
    from clinical_knowledge.protocol_links import content_disposition_inline, normalize_protocol_path

    p = normalize_protocol_path(path)
    if not p:
        raise HTTPException(status_code=404, detail="Протокол не найден")
    full = (ROOT / p).resolve()
    try:
        full.relative_to((ROOT / "minzdrav_protocols").resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Протокол не найден") from exc
    if not full.is_file():
        raise HTTPException(status_code=404, detail="Файл протокола отсутствует на сервере")
    return FileResponse(
        full,
        media_type="application/pdf",
        headers={"Content-Disposition": content_disposition_inline(full.name)},
    )


@app.post("/api/kz-matrix")
def api_kz_matrix(body: KzMatrixIn) -> dict:
    """Структура пунктов консультативного заключения по тексту протокола."""
    _require_rag_loaded()
    pth = body.path.strip()
    if not pth or pth not in _chunks_by_path:
        raise HTTPException(status_code=404, detail="Протокол не найден в индексе")
    model = get_gemini()
    cd = body.clinical_detail if isinstance(body.clinical_detail, dict) else None
    matrix = build_kz_matrix(
        model,
        body.query.strip(),
        pth,
        body.title.strip(),
        icd_codes=body.icd_codes or None,
        clinical_detail=cd,
    )
    return {"kz_matrix": matrix, "protocol_ui_meta": protocol_ui_meta_for_path(pth)}


@app.post("/api/protocol-practical")
def api_protocol_practical(body: ProtocolPracticalIn) -> dict:
    """Практический разбор: выдержка из протокола + матрица для оформления КЗ."""
    q = body.query.strip()
    pth = body.path.strip()
    mode = (body.mode or "lite").strip().lower()
    if mode not in ("lite", "full"):
        raise HTTPException(status_code=400, detail="mode должен быть lite или full")
    if mode == "lite":
        _require_rag_loaded(
            max_wait_sec=max(5.0, env_float("RAG_LOAD_WAIT_LITE_SEC", 28.0)),
        )
    else:
        _require_rag_loaded()
    if not pth or pth not in _chunks_by_path:
        raise HTTPException(status_code=404, detail="Протокол не найден в индексе")
    icd_norm = _icd_codes_from_query_and_analysis(q, body.icd_codes or None)
    meta = _protocol_meta.get(pth) or {}
    title_line = _protocol_display_title(pth, body.title.strip() or meta.get("title"))

    if mode == "lite":
        from clinical_knowledge.protocol_practical_lite import build_clinical_detail_lite

        chunks = get_rich_chunks_for_path(pth) or list(_chunks_by_path.get(pth) or [])
        clinical_detail = build_clinical_detail_lite(
            pth, q, title_line, chunks, icd_norm or None
        )
        kz_matrix: dict | None = None
        if not body.skip_kz_matrix:
            kz_matrix = kz_matrix_from_clinical_detail_heuristic(
                clinical_detail, pth, title_line, icd_norm
            )
        return {
            "mode": "lite",
            "clinical_detail": clinical_detail,
            "kz_matrix": kz_matrix,
            "protocol_ui_meta": protocol_ui_meta_for_path(pth),
        }

    model = get_gemini()
    pc = body.protocol_confidence
    if pc is not None:
        try:
            pc = float(max(0.0, min(1.0, pc)))
        except (TypeError, ValueError):
            pc = None
    crs = body.client_rag_support
    if crs is not None:
        try:
            crs = float(max(0.0, min(1.0, crs)))
        except (TypeError, ValueError):
            crs = None
    clinical_detail = extract_clinical_detail(
        pth,
        q,
        body.title.strip(),
        model,
        detailed=True,
        protocol_confidence=pc,
        extract_focus=body.extract_focus,
        client_rag_support=crs,
    )
    kz_matrix: dict | None = None
    if not body.skip_kz_matrix and not clinical_detail.get("error"):
        kz_matrix = build_kz_matrix(
            model,
            q,
            pth,
            body.title.strip(),
            icd_codes=body.icd_codes or None,
            clinical_detail=clinical_detail,
        )
    return {
        "mode": "full",
        "clinical_detail": clinical_detail,
        "kz_matrix": kz_matrix,
        "protocol_ui_meta": protocol_ui_meta_for_path(pth),
    }


@app.post("/api/protocol-practical-section")
def api_protocol_practical_section(body: ProtocolPracticalSectionIn) -> dict:
    """Быстрый разбор одного раздела протокола из rich-чанков (без LLM)."""
    _require_rag_loaded()
    from clinical_knowledge.protocol_practical_lite import (
        build_practical_section,
        normalize_practical_section,
    )

    q = body.query.strip()
    pth = body.path.strip()
    if not pth or pth not in _chunks_by_path:
        raise HTTPException(status_code=404, detail="Протокол не найден в индексе")
    sec = normalize_practical_section(body.section)
    if not sec:
        raise HTTPException(
            status_code=400,
            detail="section: investigations, medications, treatment_methods, monitoring_frequency, care_algorithms",
        )
    icd_norm = _icd_codes_from_query_and_analysis(q, body.icd_codes or None)
    meta = _protocol_meta.get(pth) or {}
    title_line = _protocol_display_title(pth, body.title.strip() or meta.get("title"))
    chunks = get_rich_chunks_for_path(pth) or list(_chunks_by_path.get(pth) or [])
    payload = build_practical_section(pth, q, title_line, chunks, sec, icd_norm or None)
    return payload


@app.post("/api/icd-suggest")
def api_icd_suggest(body: IcdSuggestIn) -> dict:
    """Та же логика МКБ, что в начале /api/assist, без RAG и без ответа LLM по протоколам."""
    model = get_gemini()
    icd_analysis, q, q_rag, _, err = _infer_icd_pipeline_from_full_query(
        body.query.strip(), model, force_icd_gemini=True
    )
    if err:
        raise HTTPException(status_code=400, detail=err)
    assert icd_analysis is not None
    payload = _icd_client_payload(icd_analysis)
    append_line = _format_icd_append_line(icd_analysis)
    hint = None
    if not (icd_analysis.get("codes_for_retrieval") or []) and not (
        icd_analysis.get("detected") or []
    ):
        hint = (
            "Коды МКБ-10 по описанию не подобраны - уточните формулировку, рубрику или введите код вручную."
        )
    return {
        "icd": payload,
        "query_effective": q,
        "rag_query": q_rag,
        "append_line": append_line,
        "hint": hint,
    }


@app.post("/api/onco-risk")
def api_onco_risk(body: OncoRiskIn) -> dict:
    """Советующая онконастороженность: байес LR+priors, полнота данных, B2B/B2C.

    Decision-support, НЕ диагноз и НЕ влияет на send_gate. Рантайм - локальный
    lookup + арифметика, без внешних вызовов. Числа отдаются только аудитории b2b.
    """
    from clinical_knowledge import onco_risk as orisk

    sex = (body.sex or "unknown").strip().lower()
    if sex not in ("male", "female", "unknown"):
        sex = "unknown"
    aoc = (body.adult_or_child or "adult").strip().lower()
    if aoc not in ("adult", "child"):
        aoc = "adult"
    audience = (body.audience or "both").strip().lower()
    if audience not in ("b2b", "b2c", "both"):
        audience = "both"

    inp = orisk.OncoInputs(
        text=body.text.strip(),
        age=body.age,
        sex=sex,
        labs_text=(body.labs_text or "").strip(),
        smoking=body.smoking,
        family_history=body.family_history,
        bmi=body.bmi,
        symptom_duration_known=bool(body.symptom_duration_known),
        adult_or_child=aoc,
    )
    assessment = orisk.assess(inp)

    out: dict = {"server_version": BUILD_VERSION, "audience": audience}
    if audience in ("b2b", "both"):
        out["assessment"] = orisk.to_b2b_payload(assessment)
    if audience in ("b2c", "both"):
        out["b2c"] = orisk.to_b2c_payload(assessment)
    return out


@app.post("/api/search/protocols-by-icd")
def api_search_protocols_by_icd(body: ProtocolsByIcdIn) -> dict:
    """Мгновенный подбор протоколов по кодам МКБ (без RAG)."""
    from icd_mkb import analyze_query_for_icd

    from clinical_knowledge.protocol_icd_index import format_assist_payload, lookup_protocols_by_icd

    t_start = time.perf_counter()
    q = body.query.strip()
    icd_analysis = analyze_query_for_icd(q, clinical_query_for_rag(q))
    icd_analysis = _merge_explicit_icd_into_analysis(icd_analysis, body.icd_codes)
    lookup = lookup_protocols_by_icd(
        icd_codes=body.icd_codes,
        query=q,
        population=body.population,
        rubric_slugs=body.category_slugs or None,
        limit=body.limit,
    )
    payload = format_assist_payload(
        query=q,
        lookup_result=lookup,
        icd_analysis=icd_analysis,
    )
    payload["icd"] = _icd_client_payload(icd_analysis)
    payload["search_timing"] = _build_search_timing(
        path="icd_fast_lookup",
        total_ms=(time.perf_counter() - t_start) * 1000,
        lookup_ms=lookup.get("lookup_ms"),
    )
    try:
        from clinical_knowledge.search_telemetry import log_protocol_search_from_payload

        aud = body.population if body.population in ("adult", "child", "pediatric") else None
        if aud == "pediatric":
            aud = "child"
        log_protocol_search_from_payload(
            query=q,
            payload=payload,
            icd_codes=list(body.icd_codes or []),
            user_slugs=list(body.category_slugs or []),
            audience_inferred=aud,
            search_source="icd_fast_lookup",
        )
    except Exception:
        pass
    return payload


@app.post("/api/search/funnel")
def api_search_funnel(body: SearchFunnelIn) -> dict:
    """Единый контракт шагов воронки 0-7 (C5)."""
    from clinical_knowledge.search_funnel import handle_search_funnel

    with _search_sem:
        return handle_search_funnel(
            query=body.query.strip(),
            step=int(body.step),
            context=body.context,
            category_slugs=list(body.category_slugs or []),
            session_id=body.session_id,
        )


@app.post("/api/search/run")
def api_search_run(body: SearchRunIn) -> dict:
    """Единая точка: воронка (step>=0) или tier S0/S1/S2."""
    from clinical_knowledge.search_run import run_search_request

    with _search_sem:

        def _assist_from_dict(payload: dict) -> dict:
            ain = AssistIn(
                query=str(payload.get("query") or ""),
                category_slugs=list(payload.get("category_slugs") or []),
                icd_codes=list(payload.get("icd_codes") or []),
                funnel_population=payload.get("funnel_population"),
                retrieve_only=bool(payload.get("retrieve_only")),
                icd_fast_path=bool(payload.get("icd_fast_path")),
                assist_full=bool(payload.get("assist_full")),
                search_tier=str(payload.get("search_tier") or "S1"),
            )
            return _api_assist_impl(ain)

        def _funnel(**kwargs: object) -> dict:
            from clinical_knowledge.search_funnel import handle_search_funnel

            return handle_search_funnel(
                query=str(kwargs.get("query") or ""),
                step=int(kwargs.get("step") or 0),
                context=dict(kwargs.get("context") or {}),
                category_slugs=list(kwargs.get("category_slugs") or []),
                session_id=kwargs.get("session_id"),  # type: ignore[arg-type]
            )

        def _by_icd(**kwargs: object) -> dict:
            pin = ProtocolsByIcdIn(
                query=str(kwargs.get("query") or ""),
                icd_codes=list(kwargs.get("icd_codes") or []),
                population=kwargs.get("population"),  # type: ignore[arg-type]
                category_slugs=list(kwargs.get("category_slugs") or []),
            )
            return api_search_protocols_by_icd(pin)

        return run_search_request(
            query=body.query,
            tier=body.tier,
            step=int(body.step),
            context=body.context,
            category_slugs=body.category_slugs,
            icd_codes=body.icd_codes,
            funnel_population=body.funnel_population,
            session_id=body.session_id,
            assist_fn=_assist_from_dict,
            funnel_fn=_funnel,
            protocols_by_icd_fn=_by_icd,
        )


@app.post("/api/consultation-template")
def api_consultation_template(body: ConsultationTemplateIn) -> dict:
    """Текстовый шаблон консультативного заключения по выдержке из протокола."""
    cd = body.clinical_detail
    if not isinstance(cd, dict) or cd.get("error"):
        raise HTTPException(
            status_code=400,
            detail="Нет корректной развёрнутой выдержки (clinical_detail)",
        )
    if body.refine and not (body.previous_template or "").strip():
        raise HTTPException(
            status_code=400,
            detail="Для доработки передайте черновик заключения (previous_template).",
        )
    model = get_gemini()
    payload = json.dumps(cd, ensure_ascii=False)
    plim = int(os.environ.get("GEMINI_TEMPLATE_PROMPT_MAX_CHARS", "28000"))
    if len(payload) > plim:
        payload = payload[: plim - 80] + "\n…[обрезано]"
    q = body.query.strip()[:8000]
    selected_payload = _normalize_selected_facts_payload(body.selected_facts_payload)
    selected_payload_json = ""
    if selected_payload.get("sections"):
        selected_payload_json = json.dumps(selected_payload, ensure_ascii=False)
    pctx = ""
    pc = (body.patient_context or "").strip()[:4000]
    if pc:
        pctx = "\n\nКонтекст пациента (из формы пользователя):\n" + pc
    if body.refine:
        draft = (body.previous_template or "").strip()
        dlim = int(os.environ.get("GEMINI_TEMPLATE_DRAFT_MAX_CHARS", "100000"))
        if len(draft) > dlim:
            draft = draft[: dlim - 80] + "\n…[черновик обрезан]"
        notes = (body.additional_notes or "").strip()[:8000]
        notes_block = (
            "\n\nДополнительные сведения от пользователя:\n" + notes
            if notes
            else ""
        )
        full_prompt = (
            SYSTEM_CONSULTATION_REFINE
            + "\n\n---\n\nЗапрос пользователя:\n"
            + q
            + pctx
            + "\n\nЧерновик заключения:\n"
            + draft
            + notes_block
            + "\n\nВыдержка из протокола (JSON):\n"
            + payload
        )
    else:
        notes0 = (body.additional_notes or "").strip()[:8000]
        notes_block0 = (
            "\n\nВыбранные пользователем пункты для включения в заключение (приоритетно отразить в соответствующих разделах):\n"
            + notes0
            if notes0
            else ""
        )
        selected_block = (
            "\n\nselected_facts_payload (структурировано, обязательно отразить):\n"
            + selected_payload_json
            if selected_payload_json
            else ""
        )
        full_prompt = (
            SYSTEM_CONSULTATION_TEMPLATE
            + "\n\n---\n\nЗапрос пользователя:\n"
            + q
            + pctx
            + notes_block0
            + selected_block
            + "\n\nВыдержка (JSON):\n"
            + payload
        )
    try:
        resp = generate_gemini_plain(model, full_prompt)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Модель: {e!s}") from e
    pf = getattr(resp, "prompt_feedback", None)
    if pf is not None and getattr(pf, "block_reason", None):
        raise HTTPException(
            status_code=502,
            detail=f"Запрос отклонён моделью: {pf.block_reason}",
        )
    txt = _extract_gemini_text(resp)
    if not txt:
        raise HTTPException(
            status_code=502,
            detail="Пустой ответ модели при формировании шаблона.",
        )
    out: dict = {"template": txt}
    if not body.refine and selected_payload.get("sections"):
        cov, missing = _selected_facts_coverage(txt, selected_payload)
        out["selected_facts_coverage"] = round(float(cov), 4)
        if cov < 0.72 and missing:
            fix_prompt = (
                SYSTEM_CONSULTATION_TEMPLATE
                + "\n\n---\n\nЗапрос пользователя:\n"
                + q
                + pctx
                + "\n\nselected_facts_payload (обязателен к покрытию):\n"
                + selected_payload_json
                + "\n\nПропущенные выбранные пункты (обязательно включить):\n- "
                + "\n- ".join(missing[:12])
                + "\n\nТекущий черновик шаблона:\n"
                + txt[:90000]
                + "\n\nЗадача: верни ПОЛНЫЙ исправленный текст шаблона, включив пропущенные пункты без выдумывания фактов."
                + "\n\nВыдержка (JSON):\n"
                + payload
            )
            try:
                fix_resp = generate_gemini_plain(model, fix_prompt)
                fix_txt = _extract_gemini_text(fix_resp)
                if fix_txt:
                    txt2 = fix_txt.strip()
                    cov2, missing2 = _selected_facts_coverage(txt2, selected_payload)
                    if cov2 >= cov:
                        out["template"] = txt2
                        out["selected_facts_coverage"] = round(float(cov2), 4)
                        out["selected_facts_repair_used"] = True
                        if missing2:
                            out["selected_facts_missing"] = missing2[:8]
                    else:
                        out["selected_facts_repair_used"] = False
                        out["selected_facts_missing"] = missing[:8]
            except Exception:
                out["selected_facts_repair_used"] = False
                out["selected_facts_missing"] = missing[:8]
    return out


# Кэш результата проверки КЗ по контент-хэшу файлов: один и тот же PDF -> один и тот же результат.
# Это даёт строгую воспроизводимость даже при остаточной недетерминированности модели.
_CONSULT_CACHE_VERSION = "2026-06-22.1-tier-key"
_consult_review_cache: dict[str, dict] = {}
_consult_cache_order: list[str] = []
_consult_cache_lock = threading.Lock()


def _consult_cache_enabled() -> bool:
    return env_bool("CONSULT_REVIEW_CACHE", True)


def _normalize_for_cache(text: str) -> str:
    """Нормализация текста для ключа кэша: один и тот же по содержанию PDF -> один ключ,
    даже если байты PDF отличаются (другой экспорт/пересохранение)."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _consult_cache_key(
    content_signature: str,
    category_slugs: str,
    *,
    tier: str = "L2",
    l2_variant: str = "",
) -> str:
    """Ключ по нормализованному содержанию (а не по байтам) + рубрики + tier + модель + настройки."""
    slugs_norm = ",".join(sorted(s.strip() for s in (category_slugs or "").split(",") if s.strip()))
    content_hash = hashlib.sha256(content_signature.encode("utf-8")).hexdigest()
    tier_norm = (tier or "L2").strip().upper()
    parts = [
        _CONSULT_CACHE_VERSION,
        content_hash,
        slugs_norm,
        tier_norm,
        "lite" if _consult_render_l2_lite_enabled() else "full",
        (l2_variant or "").strip() or _consult_l2_mode_label(),
        os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        str(env_float("GEMINI_TEMPERATURE", 0.0)),
        os.environ.get("RAG_GEMINI_EMBED_RERANK", "1").strip().lower(),
        os.environ.get("GEMINI_EMBEDDING_MODEL", ""),
        "overall_from_criteria" if env_bool("CONSULT_REVIEW_OVERALL_FROM_CRITERIA", True) else "model_overall",
        "hybrid_overall" if env_bool("CONSULT_OVERALL_HYBRID", True) else "llm_overall",
    ]
    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _consult_cache_get(key: str) -> dict | None:
    if not _consult_cache_enabled():
        return None
    with _consult_cache_lock:
        val = _consult_review_cache.get(key)
        if val is None:
            return None
        result = copy.deepcopy(val)
    result["cached_result"] = True
    return result


def _consult_cache_put(key: str, value: dict) -> None:
    if not _consult_cache_enabled():
        return
    cap = max(8, env_int("CONSULT_REVIEW_CACHE_MAX", 32 if env_bool("RENDER", False) else 256))
    stored = copy.deepcopy(value)
    if not _consult_response_include_html():
        stored.pop("report_html", None)
        stored.pop("report_markdown", None)
    with _consult_cache_lock:
        if key not in _consult_review_cache:
            _consult_cache_order.append(key)
        _consult_review_cache[key] = stored
        while len(_consult_cache_order) > cap:
            old = _consult_cache_order.pop(0)
            _consult_review_cache.pop(old, None)


def _consult_clinical_rules_pipeline(
    full_text: str,
    demographics_meta: dict,
    merged_icd: list[str] | None,
    category_slugs: list[str],
) -> dict | None:
    """MVP: извлечение фактов КЗ, подбор карточек протоколов, детерминированные правила."""
    if not env_bool("CONSULT_RULE_CHECK", True):
        return None
    try:
        from clinical_knowledge import (
            extract_consult_facts_heuristic,
            match_protocol_cards,
            run_rule_checker,
        )
    except ImportError:
        return None

    facts = extract_consult_facts_heuristic(full_text, demographics_meta=demographics_meta)
    try:
        from clinical_knowledge.consult_analysis import facts_from_document
        from clinical_knowledge.consult_parser import parse_consultation

        doc = parse_consultation(full_text, demographics_meta=demographics_meta)
        if doc.extraction_quality.parsed_sections_count > 0 or doc.diagnoses:
            facts = facts_from_document(doc)
    except Exception:
        pass
    cons = facts.setdefault("consultation", {})
    icd_merged = [str(c).upper() for c in (merged_icd or []) if c]
    cons["icd10"] = list(dict.fromkeys((cons.get("icd10") or []) + icd_merged))

    from clinical_knowledge.consult_parser import _detect_specialty
    from clinical_knowledge.protocol_match import annotate_applicability
    from clinical_knowledge.rubric_extractors import specialty_to_rubric

    doctor_rubric = specialty_to_rubric(_detect_specialty(full_text[:1500]) or _detect_specialty(full_text))
    specialty = (os.environ.get("CONSULT_RULE_CHECK_SPECIALTY") or "").strip() or None
    if not specialty and doctor_rubric in ALLOWED_SPECIALTY_SLUGS:
        specialty = doctor_rubric
    if not specialty and category_slugs:
        allowed = [sl for sl in category_slugs if sl in ALLOWED_SPECIALTY_SLUGS]
        if len(allowed) == 1:
            specialty = allowed[0]

    try:
        matched = match_protocol_cards(facts, specialty_slug=specialty, limit=8)
        patient = facts.get("patient_context") or {}
        matched = [
            m
            for m in annotate_applicability(matched, patient)
            if m.get("applicability") != "not_applicable"
        ]
        try:
            from clinical_knowledge.rich_rules_supplement import rich_table_rules_for_paths

            rich_extra = rich_table_rules_for_paths(
                [str(m.get("source_path") or "") for m in matched if m.get("source_path")]
            )
        except Exception:
            rich_extra = []
        rules = run_rule_checker(
            facts,
            matched_protocols=matched,
            extra_rules=rich_extra or None,
        )
        if rich_extra:
            rules["rich_table_rules_count"] = len(rich_extra)
    except Exception as exc:
        return {
            "consult_facts": facts,
            "matched_protocols": [],
            "rules_check": {"error": str(exc)[:240], "rules": []},
            "specialty_scope": specialty or "all_catalog",
        }

    return {
        "consult_facts": facts,
        "matched_protocols": matched,
        "rules_check": rules,
        "specialty_scope": specialty or "all_catalog",
    }


async def _read_consult_upload_bytes(uf: UploadFile, index: int, default_ext: str = ".txt") -> tuple[str, bytes]:
    """FastAPI 0.115+: await read(); sync uf.file.read() иногда пустой на Render."""
    raw_fn = ((uf.filename or "").strip()) or f"zaklyuchenie_{index + 1}{default_ext}"
    data = await uf.read()
    if not data and uf.file is not None:
        try:
            uf.file.seek(0)
            data = uf.file.read()
        except Exception:
            data = b""
    return raw_fn, data or b""


def _parse_consult_review_uploads_from_items(
    items: list[tuple[str, bytes]],
) -> tuple[str, list[dict], list[str], list[str]]:
    """Извлечь текст из уже прочитанных байтов файлов КЗ."""
    max_n = max(1, min(25, env_int("CONSULT_REVIEW_MAX_FILES", 1)))
    if len(items) > max_n:
        raise HTTPException(
            status_code=400,
            detail=f"Можно не более {max_n} файлов за один запрос.",
        )
    max_mb = env_float("CONSULT_REVIEW_MAX_MB", 15.0)
    lim_b = int(max_mb * 1024 * 1024)
    blocks: list[str] = []
    consult_docs_meta: list[dict] = []
    pdf_warnings: list[str] = []
    doc_texts_for_cache: list[str] = []

    for i, (raw_fn, data) in enumerate(items):
        if len(data) > lim_b:
            raise HTTPException(
                status_code=400,
                detail=f"Файл «{raw_fn}» превышает {max_mb} МБ",
            )
        try:
            txt, warns = extract_consult_text_from_bytes(data, raw_fn)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Ошибка чтения «{raw_fn}»: {e!s}",
            ) from e
        txt = txt.strip()
        if not txt:
            hint = ""
            if warns:
                for w in warns:
                    ws = (w or "").strip()
                    if ws:
                        hint = f" Подсказка: {ws}"
                        break
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Не удалось извлечь текст из «{raw_fn}». "
                    "Для PDF загрузите файл с текстовым слоем или читаемый скан (OCR); "
                    f"для DOCX/TXT - непустой файл.{hint}"
                ),
            )
        for w in warns or []:
            pdf_warnings.append(f"{raw_fn}: {w}")

        doc_texts_for_cache.append(_normalize_for_cache(txt))
        consult_docs_meta.append(
            {
                "index": i + 1,
                "filename": raw_fn,
                "extraction_chars": len(txt),
                "format": _consult_extension(raw_fn) or _sniff_consult_format(data, ""),
            }
        )

        shown_name = raw_fn.replace("\r", "").replace("\n", " ").strip()
        if len(shown_name) > 220:
            shown_name = shown_name[:217].rstrip() + "…"
        blocks.append(f"=== ЗАКЛЮЧЕНИЕ {i + 1} ({shown_name}) ===\n\n" + txt)

    full_text = "\n\n".join(blocks).strip()
    max_store = int(os.environ.get("CONSULT_REVIEW_MAX_TEXT_CHARS", "120000"))
    if len(full_text) > max_store:
        full_text = (
            full_text[:max_store].rstrip()
            + "\n\n[…тексты объединённых файлов обрезаны для обработки]"
        )
    return full_text, consult_docs_meta, pdf_warnings, doc_texts_for_cache


async def _parse_consult_review_uploads_async(
    files: list[UploadFile],
) -> tuple[str, list[dict], list[str], list[str]]:
    items: list[tuple[str, bytes]] = []
    for i, uf in enumerate(files):
        items.append(await _read_consult_upload_bytes(uf, i))
    return _parse_consult_review_uploads_from_items(items)


def _parse_consult_review_uploads(
    files: list[UploadFile],
) -> tuple[str, list[dict], list[str], list[str]]:
    """Sync-обёртка для TestClient и локальных скриптов."""
    default_ext = ".txt"
    items: list[tuple[str, bytes]] = []
    for i, uf in enumerate(files):
        raw_fn = ((uf.filename or "").strip()) or f"zaklyuchenie_{i + 1}{default_ext}"
        data = b""
        if uf.file is not None:
            try:
                uf.file.seek(0)
                data = uf.file.read()
            except Exception:
                data = b""
        items.append((raw_fn, data or b""))
    return _parse_consult_review_uploads_from_items(items)


def _consult_review_from_parsed_uploads(
    *,
    full_text: str,
    n_files: int,
    consult_docs_meta: list[dict],
    pdf_warnings: list[str],
    doc_texts_for_cache: list[str],
    category_slugs: str,
    on_progress=None,
) -> dict:
    from consult_review_pipeline import run_consult_review_pipeline

    content_signature = "\n||\n".join(doc_texts_for_cache)
    return run_consult_review_pipeline(
        full_text=full_text,
        n_files=n_files,
        consult_docs_meta=consult_docs_meta,
        pdf_warnings=pdf_warnings,
        content_signature=content_signature,
        category_slugs=category_slugs,
        on_progress=on_progress,
    )


def _consult_review_from_uploads(
    files: list[UploadFile],
    category_slugs: str,
    on_progress=None,
) -> dict:
    full_text, consult_docs_meta, pdf_warnings, doc_texts_for_cache = _parse_consult_review_uploads(
        files
    )
    return _consult_review_from_parsed_uploads(
        full_text=full_text,
        n_files=len(files),
        consult_docs_meta=consult_docs_meta,
        pdf_warnings=pdf_warnings,
        doc_texts_for_cache=doc_texts_for_cache,
        category_slugs=category_slugs,
        on_progress=on_progress,
    )


@app.get("/api/methodist/status")
def api_methodist_status() -> dict:
    """Публичный флаг: настроен ли кабинет методиста на сервере."""
    from clinical_knowledge.feedback_store import (
        methodist_auth_enabled,
        methodist_default_reviewer,
        methodist_ui_auto_login,
    )
    from clinical_knowledge.methodist_ai_review import methodist_ai_review_enabled
    from clinical_knowledge.gemini_model_config import methodist_gemini_model_name

    has_key = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    ai_model, model_warn = methodist_gemini_model_name()

    return {
        "enabled": methodist_auth_enabled(),
        "auto_login": methodist_ui_auto_login() and methodist_auth_enabled(),
        "default_reviewer": methodist_default_reviewer(),
        "ai_review_enabled": methodist_ai_review_enabled() and has_key,
        "ai_review_model": ai_model if methodist_ai_review_enabled() and has_key else None,
        "ai_review_model_warn": model_warn,
    }


@app.get("/api/methodist/bootstrap")
def api_methodist_bootstrap() -> dict:
    """Автовход для on-prem: токен и инициалы из env (только при METHODIST_UI_AUTO_LOGIN=1)."""
    from clinical_knowledge.feedback_store import (
        methodist_auth_enabled,
        methodist_default_reviewer,
        methodist_token_expected,
        methodist_ui_auto_login,
    )

    if not methodist_auth_enabled():
        raise HTTPException(status_code=503, detail="METHODIST_TOKEN не настроен на сервере.")
    if not methodist_ui_auto_login():
        raise HTTPException(status_code=404, detail="Автовход отключён (METHODIST_UI_AUTO_LOGIN).")
    token = methodist_token_expected()
    reviewer = methodist_default_reviewer()
    if not reviewer:
        raise HTTPException(
            status_code=503,
            detail="Задайте METHODIST_REVIEWER в .env для автовхода.",
        )
    return {"token": token, "reviewer": reviewer}


@app.get("/api/methodist/session")
def api_methodist_session(request: "Request") -> dict:
    """Проверка токена методиста (без записи feedback)."""
    _require_methodist_auth(request)
    from clinical_knowledge.feedback_store import methodist_default_reviewer

    reviewer = (request.headers.get("x-methodist-reviewer") or "").strip()
    if not reviewer:
        reviewer = methodist_default_reviewer()
    return {"ok": True, "reviewer": reviewer}


@app.get("/api/methodist/stats")
def api_methodist_stats(request: "Request") -> dict:
    """Агрегированная статистика ML/feedback для дашборда методиста."""
    _require_methodist_auth(request)
    from clinical_knowledge.methodist_stats import build_methodist_dashboard_stats

    return build_methodist_dashboard_stats()


@app.get("/api/methodist/patient-monetization")
def api_methodist_patient_monetization_get(request: "Request") -> dict:
    """Настройки монетизации B2C для кабинета методиста."""
    _require_methodist_auth(request)
    from clinical_knowledge.patient_monetization_config import monetization_admin_view

    return {"ok": True, "config": monetization_admin_view()}


@app.put("/api/methodist/patient-monetization")
def api_methodist_patient_monetization_put(
    request: "Request",
    body: PatientMonetizationPatchIn,
) -> dict:
    """Сохранить настройки монетизации B2C."""
    _require_methodist_auth(request)
    from clinical_knowledge.patient_monetization_config import (
        monetization_admin_view,
        save_patient_monetization_config,
    )

    patch = body.model_dump(exclude_none=True)
    reviewer = (request.headers.get("x-methodist-reviewer") or "").strip()
    saved = save_patient_monetization_config(patch, reviewer=reviewer)
    return {"ok": True, "config": monetization_admin_view(), "saved": saved}


def _onco_settings_state() -> dict:
    from clinical_knowledge.patient_flags import patient_onco_questions_enabled

    return {
        "consult_advisory": _consult_onco_risk_advisory_enabled(),
        "patient_b2c": patient_onco_questions_enabled(),
    }


@app.get("/api/methodist/onco-settings")
def api_methodist_onco_settings_get(request: "Request") -> dict:
    """Текущее состояние тумблеров онконастороженности (рантайм)."""
    _require_methodist_auth(request)
    return {"ok": True, "settings": _onco_settings_state()}


@app.post("/api/methodist/onco-settings")
def api_methodist_onco_settings_post(request: "Request", body: dict) -> dict:
    """Включить/выключить онко-блоки в рантайме (consult B2B-advisory и patient B2C).

    Меняет переменные окружения процесса (не персистентно между рестартами).
    """
    _require_methodist_auth(request)
    if "consult_advisory" in body:
        os.environ["CONSULT_ONCO_RISK_ADVISORY_ENABLED"] = "1" if body.get("consult_advisory") else "0"
    if "patient_b2c" in body:
        os.environ["PATIENT_ONCO_QUESTIONS_ENABLED"] = "1" if body.get("patient_b2c") else "0"
    return {"ok": True, "settings": _onco_settings_state()}


@app.get("/api/methodist/queue")
def api_methodist_queue(
    request: "Request",
    limit: int = Query(50, ge=5, le=200),
    domain: str | None = Query(None, description="kz (default) | search"),
) -> dict:
    """Очередь active learning: priority, pending, suspicious."""
    _require_methodist_auth(request)
    from clinical_knowledge.methodist_queue import build_methodist_queue

    dom = (domain or "").strip().lower() or None
    return build_methodist_queue(limit=limit, domain=dom)


@app.get("/api/methodist/patient-quality")
def api_methodist_patient_quality(request: "Request") -> dict:
    """Live-статистика B2C, последний ночной отчёт и черновики snippet-паков."""
    _require_methodist_auth(request)
    from clinical_knowledge.patient_nightly_quality import build_methodist_patient_quality_view

    return build_methodist_patient_quality_view()


@app.post("/api/methodist/patient-quality/refresh")
def api_methodist_patient_quality_refresh(request: "Request") -> dict:
    """Пересобрать ночной отчёт B2C (агрегация + LLM/эвристика, без email)."""
    _require_methodist_auth(request)
    from clinical_knowledge.patient_nightly_quality import (
        build_methodist_patient_quality_view,
        run_patient_nightly_quality,
    )

    run_patient_nightly_quality(send_email=False)
    out = build_methodist_patient_quality_view()
    out["refreshed"] = True
    return out


@app.get("/api/methodist/protocol-search")
def api_methodist_protocol_search(
    request: "Request",
    q: str = Query("", min_length=2, max_length=120),
    limit: int = Query(10, ge=1, le=20),
) -> dict:
    """Autocomplete протоколов каталога для разметки retrieval_fix (methodist only)."""
    _require_methodist_auth(request)
    from clinical_knowledge.methodist_protocol_search import search_catalog_protocols

    query = q.strip()
    return {"query": query, "items": search_catalog_protocols(query, limit=limit)}


@app.get("/api/protocol-summary-nav")
def api_protocol_summary_nav(
    path: str = Query(..., min_length=3, max_length=512),
    query: str = Query("", max_length=12000),
    icd: str = Query("", max_length=256),
    rich_fallback: bool = Query(True, description="Fallback на rich-чанки если нет Summary"),
) -> dict:
    """Оглавление Protocol Summary или rich-чанков для навигации по протоколу."""
    from clinical_knowledge.protocol_nav_cache import resolve_protocol_nav_cached

    icd_codes = [c.strip() for c in icd.split(",") if c.strip()] if icd.strip() else None
    _require_rag_loaded()
    return resolve_protocol_nav_cached(
        path.strip(),
        query=query,
        icd_codes=icd_codes,
        allow_rich_fallback=bool(rich_fallback),
    )


@app.get("/api/protocol-summary-excerpt")
def api_protocol_summary_excerpt(
    path: str = Query(..., min_length=3, max_length=512),
    condition_id: str = Query(..., min_length=1, max_length=128),
    section_id: str = Query(..., min_length=2, max_length=128),
) -> dict:
    """Цитаты из Summary или rich-чанков по разделу (без LLM) - шаг 7 воронки."""
    from clinical_knowledge.rich_chunk_search import build_rich_section_excerpt
    from clinical_knowledge.protocol_summary.nav import build_section_excerpt
    from clinical_knowledge.search_funnel import _resolve_protocol_nav

    _require_rag_loaded()
    nav = _resolve_protocol_nav(path.strip(), query="", icd_codes=None)
    if nav.get("source") == "rich_chunks":
        return build_rich_section_excerpt(
            nav,
            condition_id=condition_id.strip(),
            section_id=section_id.strip(),
        )
    allowed = {"criteria", "exams", "treatment", "red_flags", "follow_up"}
    sid = section_id.strip()
    if sid not in allowed:
        raise HTTPException(status_code=400, detail=f"section_id must be one of: {', '.join(sorted(allowed))}")
    return build_section_excerpt(path.strip(), condition_id=condition_id.strip(), section_id=sid)


@app.get("/api/protocol-source-text")
def api_protocol_source_text(
    path: str = Query(..., min_length=3, max_length=512),
) -> dict:
    """Полный текст протокола по разделам (source_text) для viewer с поиском и подсветкой."""
    from clinical_knowledge.protocol_links import normalize_protocol_path
    from clinical_knowledge.protocol_summary.source_text import resolve_protocol_source_text

    pth = normalize_protocol_path(path.strip())
    if not pth:
        raise HTTPException(status_code=404, detail="Протокол не найден")
    full = (ROOT / pth).resolve()
    try:
        full.relative_to((ROOT / "minzdrav_protocols").resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Протокол не найден") from exc
    # rich-чанки дают настоящий chunk_type и списки сущностей -> качественнее view.
    rich_chunks: list[dict] = []
    try:
        _require_rag_loaded(max_wait_sec=max(3.0, env_float("RAG_LOAD_WAIT_LITE_SEC", 28.0)))
        rich_chunks = get_rich_chunks_for_path(pth)
    except Exception:
        rich_chunks = []
    out = resolve_protocol_source_text(pth, rich_chunks=rich_chunks or None)
    out["build_version"] = BUILD_VERSION
    return out


@app.get("/api/protocol-search-intents")
def api_protocol_search_intents() -> dict:
    """Спеки intent для навигации (синхрон с clinical_knowledge/protocol_search_intents.py)."""
    from clinical_knowledge.protocol_search_intents import specs_for_api

    return {"ok": True, "intents": specs_for_api(), "build_version": BUILD_VERSION}


@app.get("/api/protocol-semantic-search")
def api_protocol_semantic_search(
    path: str = Query(..., min_length=3, max_length=512),
    q: str = Query(..., min_length=2, max_length=1200),
    top_k: int = Query(12, ge=1, le=24),
) -> dict:
    """Семантический поиск внутри одного протокола (vector + intent + lex)."""
    from clinical_knowledge.protocol_semantic_search import search_protocol_semantic

    out = search_protocol_semantic(path.strip(), q.strip(), top_k=top_k)
    out["build_version"] = BUILD_VERSION
    return out


class ProtocolOverviewIn(BaseModel):
    path: str = Field(min_length=3, max_length=512)
    q: str = Field(min_length=2, max_length=1200)
    title: str = Field(default="", max_length=512)


@app.post("/api/protocol-overview")
def api_protocol_overview(body: ProtocolOverviewIn) -> dict:
    """AI Overview по протоколу: краткий ответ + цитаты из top-K чанков."""
    from clinical_knowledge.protocol_semantic_search import build_protocol_overview

    out = build_protocol_overview(
        body.path.strip(),
        body.q.strip(),
        title=(body.title or "").strip(),
    )
    out["build_version"] = BUILD_VERSION
    return out


@app.get("/api/protocol-brief-bundle")
def api_protocol_brief_bundle(
    path: str = Query(..., min_length=3, max_length=512),
    condition_id: str = Query(..., min_length=1, max_length=128),
    query: str = Query("", max_length=12000),
    icd: str = Query("", max_length=256),
) -> dict:
    """Развёрнутая сводка + KZ-brief + prefetch матрицы (без LLM)."""
    from clinical_knowledge.protocol_kz_brief import resolve_protocol_brief_bundle_cached

    icd_codes = [c.strip() for c in icd.split(",") if c.strip()] if icd.strip() else None
    _require_rag_loaded(
        max_wait_sec=max(5.0, env_float("RAG_LOAD_WAIT_LITE_SEC", 28.0)),
    )
    return resolve_protocol_brief_bundle_cached(
        path.strip(),
        condition_id=condition_id.strip(),
        query=query,
        icd_codes=icd_codes,
    )


@app.post("/api/brief-feedback")
def api_brief_feedback(body: dict) -> dict:
    """Телеметрия полезности сводки (без авторизации методиста)."""
    import json
    from datetime import datetime, timezone

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Ожидается JSON-объект")
    rating = str(body.get("rating") or "").strip().lower()
    if rating not in ("useful", "insufficient", "wrong"):
        raise HTTPException(status_code=400, detail="rating: useful | insufficient | wrong")
    event = {
        "event_type": "brief_feedback",
        "ts": datetime.now(timezone.utc).isoformat(),
        "rating": rating,
        "path": str(body.get("path") or "")[:512],
        "condition_id": str(body.get("condition_id") or "")[:128],
        "block_id": str(body.get("block_id") or "")[:64],
        "build_version": BUILD_VERSION,
    }
    log_dir = ROOT / "data" / "ml" / "feedback"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "brief_feedback.jsonl"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    return {"ok": True}


@app.get("/api/methodist/analysis/{analysis_id}")
def api_methodist_analysis(request: "Request", analysis_id: str) -> dict:
    """Снимок сохранённого прогона (api_result) без пересчёта."""
    _require_methodist_auth(request)
    from clinical_knowledge.methodist_analysis import get_methodist_analysis

    payload = get_methodist_analysis(analysis_id.strip())
    if not payload:
        raise HTTPException(status_code=404, detail="Снимок analysis_id не найден.")
    return payload


class MethodistAiReviewIn(BaseModel):
    analysis_id: str = Field(min_length=8, description="UUID прогона из kz_analysis")


class MethodistSearchAiReviewIn(BaseModel):
    query: str = Field(min_length=1, max_length=12000)
    llm_json: dict[str, Any] = Field(default_factory=dict)
    retrieval: list[dict[str, Any]] = Field(default_factory=list)
    icd_codes: list[str] | None = None
    retrieve_only: bool = False
    funnel_context: dict[str, Any] | None = None
    audience_inferred: str | None = None


@app.post("/api/methodist/search-ai-review")
def api_methodist_search_ai_review(request: "Request", body: MethodistSearchAiReviewIn) -> dict:
    """ИИ оценивает выдачу поиска; методист только одобряет или правит."""
    _require_methodist_auth(request)
    from clinical_knowledge.methodist_ai_review import methodist_ai_review_enabled
    from clinical_knowledge.methodist_search_ai_review import (
        build_deterministic_search_ai_review,
        persist_search_ai_artifact,
        run_methodist_search_ai_review,
    )

    if not methodist_ai_review_enabled():
        raise HTTPException(status_code=404, detail="METHODIST_AI_REVIEW отключён на сервере.")

    icd_codes = list(body.icd_codes or [])
    payload = {
        "query": body.query.strip(),
        "llm_json": body.llm_json or {},
        "retrieval": body.retrieval or [],
        "icd_codes": icd_codes,
        "retrieve_only": bool(body.retrieve_only),
        "funnel_context": body.funnel_context if isinstance(body.funnel_context, dict) else None,
        "audience_inferred": (body.audience_inferred or "").strip() or None,
    }
    llm_json = payload["llm_json"]
    if isinstance(llm_json, dict) and body.retrieval and not llm_json.get("protocols"):
        rows = dedupe_retrieval_by_basename(body.retrieval)
        llm_json = dict(llm_json)
        llm_json["protocols"] = _build_protocols_from_retrieval(rows)
        payload["llm_json"] = llm_json

    ai_review: dict
    fallback = False
    fallback_reason = ""
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        ai_review = build_deterministic_search_ai_review(payload)
        fallback = True
        fallback_reason = "GOOGLE_API_KEY не настроен"
    else:
        try:
            ai_review = run_methodist_search_ai_review(payload)
        except HTTPException as e:
            if e.status_code in (401, 403, 404):
                raise
            ai_review = build_deterministic_search_ai_review(payload)
            fallback = True
            fallback_reason = str(e.detail)[:200]
        except (ValueError, Exception) as e:
            ai_review = build_deterministic_search_ai_review(payload)
            fallback = True
            fallback_reason = str(e)[:200]

    artifact_id = persist_search_ai_artifact(
        payload, ai_review, fallback=fallback, fallback_reason=fallback_reason
    )

    return {
        "ok": True,
        "ai_review": ai_review,
        "fallback": fallback,
        "fallback_reason": fallback_reason if fallback else None,
        "artifact_id": artifact_id,
    }


class MethodistSearchProbeIn(BaseModel):
    limit: int = Field(default=15, ge=1, le=120, description="Сколько кейсов из fixture прогнать")
    group: str | None = Field(default=None, max_length=80, description="Фильтр по group в fixture")


@app.post("/api/methodist/search-probe")
def api_methodist_search_probe(request: "Request", body: MethodistSearchProbeIn) -> dict:
    """Batch-прогон fixture probe поиска протоколов (как scripts/run_methodist_search_probe.py)."""
    _require_methodist_auth(request)
    from clinical_knowledge.methodist_search_probe_runner import run_probe_batch

    _require_rag_loaded()
    reports, summary = run_probe_batch(limit=body.limit, group=(body.group or "").strip() or None)
    summary["build_version"] = BUILD_VERSION
    return {
        "ok": True,
        "summary": summary,
        "reports": reports,
    }


@app.post("/api/methodist/ai-review")
def api_methodist_ai_review(request: "Request", body: MethodistAiReviewIn) -> dict:
    """Этап 2: LLM оценивает результат детерминированного анализа; методист только одобряет."""
    _require_methodist_auth(request)
    from clinical_knowledge.feedback_store import load_analysis_snapshot, load_secure_kz_text
    from clinical_knowledge.methodist_ai_review import methodist_ai_review_enabled, run_methodist_ai_review

    if not methodist_ai_review_enabled():
        raise HTTPException(status_code=404, detail="METHODIST_AI_REVIEW отключён на сервере.")
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise HTTPException(status_code=503, detail="GOOGLE_API_KEY не настроен для AI-оценки.")

    snap = load_analysis_snapshot(body.analysis_id.strip())
    if not snap:
        raise HTTPException(status_code=404, detail="Снимок analysis_id не найден. Повторите анализ в режиме методиста.")
    result = snap.get("api_result")
    if not isinstance(result, dict):
        raise HTTPException(status_code=404, detail="Некорректный снимок анализа.")

    text_hash = str(snap.get("text_hash") or result.get("text_hash") or "")
    full_text = load_secure_kz_text(text_hash) or ""
    if not full_text.strip():
        full_text = str(snap.get("text_excerpt") or "")

    try:
        ai_review = run_methodist_ai_review(result, full_text)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e)[:240]) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка AI-оценки: {str(e)[:200]}") from e

    from clinical_knowledge.gemini_model_config import methodist_gemini_model_name

    resolved, model_warn = methodist_gemini_model_name()
    if model_warn:
        ai_review["model_warn"] = model_warn
    ai_review["model_used"] = resolved

    return {
        "ok": True,
        "analysis_id": body.analysis_id.strip(),
        "ai_review": ai_review,
    }


@app.get("/api/ml/feedback/export")
def api_ml_feedback_export(
    request: "Request",
    since: str | None = Query(
        None,
        description="ISO-дата (YYYY-MM-DD) или datetime; только события с ts ≥ since",
        max_length=40,
    ),
) -> "Response":
    """Скачать feedback/*.jsonl (без текста КЗ) для export_training_feedback на Mac."""
    _require_methodist_auth(request)
    from clinical_knowledge.feedback_store import build_feedback_export_tar_gz

    data, manifest = build_feedback_export_tar_gz(since=since)
    exported_at = str(manifest.get("exported_at") or "export")
    fname = f"ml_feedback_{exported_at[:10]}.tar.gz"
    return Response(
        content=data,
        media_type="application/gzip",
        headers={
            "Content-Disposition": f'attachment; filename="{fname}"',
            "X-Feedback-Event-Count": str(manifest.get("event_count", 0)),
        },
    )


@app.post("/api/ml/feedback")
def api_ml_feedback(request: "Request", body: dict) -> dict:
    """Append-only запись события разметки методиста (JSONL)."""
    _require_methodist_auth(request)
    from clinical_knowledge.feedback_store import append_feedback_event, expand_analysis_review_events

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Ожидается JSON-объект")
    reviewer_hdr = (request.headers.get("x-methodist-reviewer") or "").strip()
    if reviewer_hdr and not body.get("reviewer"):
        body = {**body, "reviewer": reviewer_hdr}
    et = (body.get("event_type") or "").strip()
    if not et:
        raise HTTPException(status_code=400, detail="Поле event_type обязательно")
    try:
        if et == "analysis_review":
            ids: list[str] = []
            for ev in expand_analysis_review_events(body):
                ids.append(append_feedback_event(ev))
            return {"ok": True, "event_id": ids[0] if ids else ""}
        event_id = append_feedback_event(body)
        return {"ok": True, "event_id": event_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/search/feedback")
def api_search_feedback(request: "Request", body: dict) -> dict:
    """Лёгкий фидбэк врача по подбору протокола (подошёл / не тот). Без методист-авторизации."""
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Ожидается JSON-объект")
    from clinical_knowledge.search_telemetry import log_search_feedback

    try:
        event_id = log_search_feedback(
            query=str(body.get("query") or ""),
            verdict=str(body.get("verdict") or ""),
            rejected_path=body.get("rejected_path"),
            chosen_path=body.get("chosen_path"),
            top_paths=body.get("top_paths") if isinstance(body.get("top_paths"), list) else None,
            icd_codes=body.get("icd_codes") if isinstance(body.get("icd_codes"), list) else None,
            source=str(body.get("source") or "doctor_search"),
        )
        return {"ok": True, "event_id": event_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/methodist/search-feedback")
def api_methodist_search_feedback(request: "Request") -> dict:
    """Очередь разметки: агрегат фидбэка врачей по подбору протоколов (methodist only)."""
    _require_methodist_auth(request)
    from clinical_knowledge.search_telemetry import (
        aggregate_search_feedback,
        iter_search_feedback_events,
    )

    events = iter_search_feedback_events()
    return aggregate_search_feedback(events)


@app.post("/api/consult-compliance-screen")
def api_consult_compliance_screen(request: "Request", body: ConsultComplianceScreenIn) -> dict:
    """Быстрый L0: структурный разбор + send_gate (<2 с, без LLM-критериев и RAG)."""
    from clinical_knowledge.consult_screen import run_compliance_screen

    if not (body.text or "").strip() and not body.bundle:
        raise HTTPException(status_code=400, detail="Укажите text или bundle (FHIR BY).")
    try:
        t0 = time.perf_counter()
        result = run_compliance_screen(
            text=body.text,
            bundle=body.bundle,
            consultation_id=body.consultation_id,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        raw_text = _consult_text_from_screen_body(body)
        return _maybe_methodist_autolog(
            request,
            result,
            tier="L0",
            full_text=raw_text,
            consultation_id=body.consultation_id,
            latency_ms=latency_ms,
            sandbox=body.sandbox,
            body_methodist_mode=body.methodist_mode,
            category_slugs=getattr(body, "category_slugs", "") or "",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _consult_review_from_tier_or_pipeline(
    *,
    tier: str,
    text: str | None,
    bundle: dict | None,
    consultation_id: str,
    category_slugs: str,
    require_rag_for_l2: bool = True,
    l2_narrative: bool = False,
) -> dict:
    from clinical_knowledge.consult_tiering import resolve_tier, run_consult_by_tier
    from clinical_knowledge.fhir_bundle_adapter import bundle_to_consultation_text
    from consult_review_pipeline import run_consult_review_pipeline

    if resolve_tier(tier) == "L2" and _consult_render_l2_skip_llm():
        routed = run_consult_by_tier(
            tier="L1",
            text=text,
            bundle=bundle,
            consultation_id=consultation_id,
            category_slugs=category_slugs,
        )
        routed["review_tier"] = "L2"
        _annotate_render_l2_limited(routed)
        return routed

    routed = run_consult_by_tier(
        tier=tier,
        text=text,
        bundle=bundle,
        consultation_id=consultation_id,
        category_slugs=category_slugs,
    )
    if not routed.get("delegate_full_pipeline"):
        return routed
    if require_rag_for_l2 and not _consult_render_l2_lite_enabled():
        _require_rag_loaded()
    full_text = routed.get("text") or ""
    if not full_text and bundle:
        full_text = bundle_to_consultation_text(bundle)
    meta = (
        [{"filename": "fhir_bundle.json", "source": "fhir"}]
        if bundle
        else [{"filename": "mis_text.txt", "source": "json"}]
    )
    result = run_consult_review_pipeline(
        full_text=full_text,
        n_files=1,
        consult_docs_meta=meta,
        pdf_warnings=[],
        content_signature=full_text[:8000],
        category_slugs=category_slugs,
        fhir_bundle=bundle if bundle else None,
        l2_narrative=l2_narrative,
    )
    result["review_tier"] = "L2"
    return result


@app.post("/api/consult-review/tier")
def api_consult_review_tier(request: "Request", body: ConsultReviewTierIn) -> dict:
    """L0/L1/L2: скрининг, structured или полный pipeline (МИС, массовый поток)."""
    if not (body.text or "").strip() and not body.bundle:
        raise HTTPException(status_code=400, detail="Укажите text или bundle (FHIR BY).")
    try:
        t0 = time.perf_counter()
        result = _consult_review_from_tier_or_pipeline(
            tier=body.tier,
            text=body.text,
            bundle=body.bundle,
            consultation_id=body.consultation_id,
            category_slugs=body.category_slugs,
            require_rag_for_l2=True,
            l2_narrative=bool(body.l2_narrative),
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        raw_text = (body.text or "").strip()
        if not raw_text and body.bundle:
            from clinical_knowledge.fhir_bundle_adapter import bundle_to_consultation_text

            raw_text = bundle_to_consultation_text(body.bundle)
        tier = (body.tier or result.get("review_tier") or "L2").upper()
        return _maybe_methodist_autolog(
            request,
            result,
            tier=tier,
            full_text=raw_text,
            consultation_id=body.consultation_id,
            latency_ms=latency_ms,
            sandbox=body.sandbox,
            body_methodist_mode=body.methodist_mode,
            category_slugs=body.category_slugs or "",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/consult-review/l2-narrative")
def api_consult_review_l2_narrative(body: ConsultL2NarrativeIn) -> dict:
    """L2+: пояснение методиста по evidence pack (без повторного полного L2)."""
    if not body.evidence_pack:
        raise HTTPException(status_code=400, detail="Укажите evidence_pack из результата L2.")
    try:
        t0 = time.perf_counter()
        summary = _consult_l2_narrative_synthesize(
            evidence_pack=body.evidence_pack,
            block_gaps=body.block_gaps,
            structured_summary=body.structured_summary,
        )
        return {
            "ok": True,
            "summary_ru": summary,
            "l2_mode": "narrative",
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)[:200]) from e


@app.post("/api/consult-review/json")
def api_consult_review_json(request: "Request", body: ConsultReviewJsonIn) -> dict:
    """Полный конвейер проверки КЗ из текста или FHIR BY Bundle (интеграция «Айболит»)."""
    if not (body.text or "").strip() and not body.bundle:
        raise HTTPException(status_code=400, detail="Укажите text или bundle (FHIR BY).")
    try:
        t0 = time.perf_counter()
        result = _consult_review_from_tier_or_pipeline(
            tier=body.tier,
            text=body.text,
            bundle=body.bundle,
            consultation_id="json",
            category_slugs=body.category_slugs,
            require_rag_for_l2=True,
            l2_narrative=bool(body.l2_narrative),
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        raw_text = (body.text or "").strip()
        if not raw_text and body.bundle:
            from clinical_knowledge.fhir_bundle_adapter import bundle_to_consultation_text

            raw_text = bundle_to_consultation_text(body.bundle)
        tier = (body.tier or result.get("review_tier") or "L2").upper()
        return _maybe_methodist_autolog(
            request,
            result,
            tier=tier,
            full_text=raw_text,
            consultation_id="json",
            latency_ms=latency_ms,
            sandbox=body.sandbox,
            body_methodist_mode=body.methodist_mode,
            category_slugs=body.category_slugs or "",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/consult-validate-bundle")
def api_consult_validate_bundle(body: ConsultValidateBundleIn) -> dict:
    """Чек-лист готовности Bundle к импорту в ЦИСЗ (программа испытаний МИС v.1.3-4)."""
    from clinical_knowledge.cisz_readiness import evaluate_cisz_readiness

    if not isinstance(body.bundle, dict) or body.bundle.get("resourceType") != "Bundle":
        raise HTTPException(status_code=400, detail="Ожидается FHIR Bundle (resourceType=Bundle).")
    cisz = evaluate_cisz_readiness(bundle=body.bundle, scenario=body.scenario)
    return {"ok": cisz.get("ok", True), "cisz_readiness": cisz}


@app.post("/api/consult-review")
async def api_consult_review(
    request: "Request",
    files: Annotated[
        list[UploadFile],
        File(description="Один файл после приёма: КЗ, медицинский осмотр или консультация (PDF, TXT, DOCX, RTF, ODT, HTML и др.)"),
    ],
    category_slugs: str = Form(
        "",
        description=(
            "Необязательно: через запятую идентификаторы рубрик каталога (тот же slug, что в адресе раздела сайта Минздрава "
            'РБ, например pulmonologiya-ftiziatriya)'
        ),
    ),
    tier: str = Form(
        "",
        description="L0/L1/L2 - уровень проверки (по умолчанию L1)",
    ),
) -> dict:
    """Загрузка одного или нескольких файлов заключений → отбор фрагментов протоколов → JSON-оценка.

    Не медико-правовая экспертиза; ориентир для методиста при настроенном сервере обработки текста.
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="Не переданы файлы: загрузите хотя бы один файл заключения.",
        )
    selected_tier = (tier or "L1").strip().upper()
    full_text, consult_docs_meta, pdf_warnings, doc_texts_for_cache = (
        await _parse_consult_review_uploads_async(files)
    )
    _ensure_consult_rag_ready()
    if selected_tier in ("L0", "L1"):
        t0 = time.perf_counter()
        result = await _run_consult_review_blocking(
            _consult_review_from_tier_or_pipeline,
            tier=selected_tier,
            text=full_text,
            bundle=None,
            consultation_id="upload",
            category_slugs=category_slugs,
            require_rag_for_l2=False,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        if pdf_warnings:
            result["pdf_warnings"] = pdf_warnings
        if consult_docs_meta:
            result["consult_documents"] = consult_docs_meta
        return _maybe_methodist_autolog(
            request,
            result,
            tier=selected_tier,
            full_text=full_text,
            consultation_id="upload",
            latency_ms=latency_ms,
            category_slugs=category_slugs,
        )
    if _consult_render_l2_skip_llm():
        t0 = time.perf_counter()
        result = await _run_consult_review_blocking(
            _consult_review_from_tier_or_pipeline,
            tier="L2",
            text=full_text,
            bundle=None,
            consultation_id="upload",
            category_slugs=category_slugs,
            require_rag_for_l2=False,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        if pdf_warnings:
            result["pdf_warnings"] = pdf_warnings
        if consult_docs_meta:
            result["consult_documents"] = consult_docs_meta
        return _maybe_methodist_autolog(
            request,
            result,
            tier="L2",
            full_text=full_text,
            consultation_id="upload",
            latency_ms=latency_ms,
            category_slugs=category_slugs,
        )
    if not _consult_render_l2_lite_enabled():
        _require_rag_loaded()
    t0 = time.perf_counter()
    result = await _run_consult_review_blocking(
        _consult_review_from_tier_or_pipeline,
        tier="L2",
        text=full_text,
        bundle=None,
        consultation_id="upload",
        category_slugs=category_slugs,
        require_rag_for_l2=False,
    )
    if pdf_warnings:
        result["pdf_warnings"] = pdf_warnings
    if consult_docs_meta:
        result["consult_documents"] = consult_docs_meta
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return _maybe_methodist_autolog(
        request,
        result,
        tier="L2",
        full_text=full_text,
        consultation_id="upload",
        latency_ms=latency_ms,
        category_slugs=category_slugs,
    )


@app.post("/api/consult-review/stream")
async def api_consult_review_stream(
    request: "Request",
    files: Annotated[
        list[UploadFile],
        File(description="Один файл после приёма: КЗ, медосмотр или консультация"),
    ],
    category_slugs: str = Form(""),
    tier: str = Form(""),
    l2_narrative: str = Form(""),
):
    """SSE-поток прогресса проверки КЗ: события progress (pct, partial) и done (result)."""
    from consult_review_pipeline import (
        iter_consult_review_pipeline,
        sse_encode_done,
        sse_encode_error,
        sse_encode_progress,
    )

    if not files:
        raise HTTPException(status_code=400, detail="Не переданы файлы.")

    try:
        full_text, consult_docs_meta, pdf_warnings, doc_texts_for_cache = (
            await _parse_consult_review_uploads_async(files)
        )
    except HTTPException as e:
        detail = e.detail if isinstance(e.detail, str) else str(e.detail)

        def err_gen():
            yield sse_encode_error(detail, e.status_code)

        return StreamingResponse(err_gen(), media_type="text/event-stream")

    content_signature = "\n||\n".join(doc_texts_for_cache)
    selected_tier = (tier or "L1").strip().upper()
    stream_t0 = time.perf_counter()

    if selected_tier in ("L0", "L1"):

        def tier_gen():
            try:
                t0 = time.perf_counter()
                result = _consult_review_from_tier_or_pipeline(
                    tier=selected_tier,
                    text=full_text,
                    bundle=None,
                    consultation_id="upload",
                    category_slugs=category_slugs,
                    require_rag_for_l2=False,
                )
                latency_ms = int((time.perf_counter() - t0) * 1000)
                if pdf_warnings:
                    result["pdf_warnings"] = pdf_warnings
                if consult_docs_meta:
                    result["consult_documents"] = consult_docs_meta
                result = _maybe_methodist_autolog(
                    request,
                    result,
                    tier=selected_tier,
                    full_text=full_text,
                    consultation_id="upload",
                    latency_ms=latency_ms,
                    category_slugs=category_slugs,
                )
                yield sse_encode_progress(
                    "extract",
                    100,
                    f"Готово ({selected_tier}, {len(full_text)} симв.)",
                    result,
                )
                yield sse_encode_done(result)
            except HTTPException as e:
                detail = e.detail if isinstance(e.detail, str) else str(e.detail)
                yield sse_encode_error(detail, e.status_code)
            except Exception as e:
                yield sse_encode_error(str(e)[:400], 500)

        return StreamingResponse(
            tier_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if not _consult_render_l2_lite_enabled():
        _require_rag_loaded()

    def event_gen():
        sent_done = False
        yield sse_encode_progress(
            "extract",
            12,
            f"Текст извлечён ({len(full_text)} симв., {len(files)} файл.)",
            {
                "consult_documents": consult_docs_meta,
                "extraction_chars": len(full_text),
                "documents_count": len(files),
            },
        )
        try:
            for kind, payload in iter_consult_review_pipeline(
                full_text=full_text,
                n_files=len(files),
                consult_docs_meta=consult_docs_meta,
                pdf_warnings=pdf_warnings,
                content_signature=content_signature,
                category_slugs=category_slugs,
                l2_narrative=str(l2_narrative or "").strip().lower() in ("1", "true", "yes", "on"),
            ):
                if kind == "progress":
                    p = payload
                    yield sse_encode_progress(
                        p["stage"],
                        p["pct"],
                        p["label_ru"],
                        p.get("partial"),
                    )
                elif kind == "done":
                    latency_ms = int((time.perf_counter() - stream_t0) * 1000)
                    payload = _maybe_methodist_autolog(
                        request,
                        payload,
                        tier="L2",
                        full_text=full_text,
                        consultation_id="upload",
                        latency_ms=latency_ms,
                        category_slugs=category_slugs,
                    )
                    yield sse_encode_done(payload)
                    sent_done = True
                    return
        except HTTPException as e:
            detail = e.detail if isinstance(e.detail, str) else str(e.detail)
            yield sse_encode_error(detail, e.status_code)
            sent_done = True
        except Exception as e:
            yield sse_encode_error(str(e)[:400], 500)
            sent_done = True
        finally:
            if not sent_done:
                yield sse_encode_error(
                    "Сессия прервана до завершения анализа. Повторите запрос - будет попытка без потока.",
                    500,
                )

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _patient_review_enabled() -> bool:
    return env_bool("PATIENT_REVIEW_ENABLED", True)


def _patient_html_response() -> "Response":
    """patient.html с подставленной BUILD_VERSION (без устаревшего SW-кэша)."""
    p = ROOT / "patient.html"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Страница patient.html не найдена")
    html = p.read_text(encoding="utf-8").replace("__BUILD_VERSION__", BUILD_VERSION)
    html = html.replace("__PATIENT_FILE_ACCEPT__", consult_review_file_accept_attr())
    max_files = env_int("PATIENT_REVIEW_MAX_FILES", 1)
    html = html.replace("__PATIENT_FORMATS_HINT__", consult_review_formats_hint_ru(max_files=max_files))
    return Response(
        content=html,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


def _run_patient_review_core(
    *,
    text: str,
    consultation_id: str = "patient",
    demographics_meta: dict | None = None,
    lab_text: str | None = None,
    product_tier: str = "P1",
    catalog_tier_id: str | None = None,
    question_tone: str | None = None,
    kz_filename: str = "",
    lab_filename: str = "",
) -> dict:
    from clinical_knowledge.patient_review import run_patient_review

    _ensure_consult_rag_ready()
    return run_patient_review(
        text=text,
        consultation_id=consultation_id,
        demographics_meta=demographics_meta,
        lab_text=lab_text,
        product_tier=product_tier,
        catalog_tier_id=catalog_tier_id,
        question_tone=question_tone,
        kz_filename=kz_filename,
        lab_filename=lab_filename,
    )


def _first_upload_filename(files: list | None) -> str:
    if not files:
        return ""
    fn = getattr(files[0], "filename", None) or ""
    return str(fn).strip()


def _attach_patient_review_telemetry(
    result: dict,
    *,
    kz_text: str,
    lab_upload: bool = False,
    latency_ms: int | None = None,
) -> dict:
    if not isinstance(result, dict) or result.get("upload_mismatch"):
        return result
    pr = result.get("patient_report")
    if not isinstance(pr, dict):
        return result
    from clinical_knowledge.patient_feedback_store import record_patient_review_snapshot

    fp = record_patient_review_snapshot(
        kz_text=kz_text,
        report=pr,
        build_version=BUILD_VERSION,
        latency_ms=latency_ms,
        has_lab_upload=lab_upload,
    )
    result["review_fingerprint"] = fp
    return result


def _patient_product_tier_from_catalog(tier_id: str | None) -> str:
    from clinical_knowledge.patient_clinic_config import resolve_tier

    tier = resolve_tier(tier_id)
    return str(tier.get("review_tier") or "P1").upper()


_patient_idempotency_cache: dict[str, tuple[float, dict]] = {}
_PATIENT_IDEMPOTENCY_TTL_SEC = 120.0


def _patient_idempotency_get(key: str | None) -> dict | None:
    k = (key or "").strip()
    if not k:
        return None
    row = _patient_idempotency_cache.get(k)
    if not row:
        return None
    ts, payload = row
    if time.time() - ts > _PATIENT_IDEMPOTENCY_TTL_SEC:
        _patient_idempotency_cache.pop(k, None)
        return None
    return dict(payload)


def _patient_idempotency_put(key: str | None, payload: dict) -> None:
    k = (key or "").strip()
    if not k or not isinstance(payload, dict):
        return
    if len(_patient_idempotency_cache) > 500:
        cutoff = time.time() - _PATIENT_IDEMPOTENCY_TTL_SEC
        stale = [kk for kk, (ts, _) in _patient_idempotency_cache.items() if ts < cutoff]
        for kk in stale:
            _patient_idempotency_cache.pop(kk, None)
    _patient_idempotency_cache[k] = (time.time(), dict(payload))


def _require_patient_payment(payment_token: str | None, tier_id: str | None) -> None:
    from clinical_knowledge.patient_payment import payment_required, verify_payment_token

    if not payment_required():
        return
    if not verify_payment_token(payment_token, tier_id=tier_id):
        raise HTTPException(
            status_code=402,
            detail="Требуется оплата. Создайте сессию: POST /api/patient/payment/session",
        )


async def _parse_patient_lab_uploads_async(
    files: list,
) -> tuple[str, list[str]]:
    """Извлечь текст из бланков анализов (отдельно от КЗ)."""
    if not files:
        return "", []
    max_n = env_int("PATIENT_LAB_MAX_FILES", 3)
    items: list[tuple[str, bytes]] = []
    for i, uf in enumerate(files[:max_n]):
        items.append(await _read_consult_upload_bytes(uf, i, default_ext=".pdf"))
    if not items:
        return "", []
    blocks: list[str] = []
    warnings: list[str] = []
    for raw_fn, data in items:
        try:
            from clinical_knowledge.patient_lab_ocr import extract_lab_text_from_bytes

            txt, warns = extract_lab_text_from_bytes(data, raw_fn)
        except HTTPException:
            raise
        except Exception as e:
            warnings.append(f"{raw_fn}: {e!s}")
            continue
        txt = (txt or "").strip()
        if txt:
            blocks.append(f"=== АНАЛИЗ ({raw_fn}) ===\n{txt}")
        for w in warns or []:
            warnings.append(f"{raw_fn}: {w}")
    return "\n\n".join(blocks).strip(), warnings


@app.get("/api/patient/status")
@app.get("/api/patient/config")
@app.get("/api/patient/health")
def api_patient_status() -> dict:
    """Статус B2C-контура для мобильного приложения и patient.html."""
    from clinical_knowledge.patient_flags import (
        patient_no_history_mode_enabled,
        patient_plain_terms_enabled,
        patient_protocol_age_filter_enabled,
        patient_rag_retrieval_enabled,
        patient_question_safety_enabled,
        patient_report_v2_enabled,
        patient_safe_quotes_enabled,
        patient_show_protocol_technical_block,
        patient_visit_sheet_pdf_enabled,
    )
    from clinical_knowledge.patient_payment import payment_required, tier_catalog_public
    from clinical_knowledge.patient_question_tone import question_tones_for_api
    from clinical_knowledge.patient_monetization_config import monetization_public_view

    mon = monetization_public_view()
    return {
        "ok": True,
        "enabled": _patient_review_enabled(),
        "review_tier": "P1",
        "report_schema_version": 2,
        "build_version": BUILD_VERSION,
        "brand_name": "Protocol",
        "upload": "POST /api/patient/review",
        "upload_stream": "POST /api/patient/review/stream",
        "lab_upload": "optional lab_files in multipart",
        "question_tone_field": "question_tone",
        "question_tones": question_tones_for_api(),
        "default_question_tone": "serious",
        "patient_rag_retrieval": patient_rag_retrieval_enabled(),
        "feature_flags": {
            "PATIENT_REPORT_V2_ENABLED": patient_report_v2_enabled(),
            "PATIENT_PROTOCOL_AGE_FILTER_ENABLED": patient_protocol_age_filter_enabled(),
            "PATIENT_SAFE_QUOTES_ENABLED": patient_safe_quotes_enabled(),
            "PATIENT_QUESTION_SAFETY_ENABLED": patient_question_safety_enabled(),
            "PATIENT_PLAIN_TERMS_ENABLED": patient_plain_terms_enabled(),
            "PATIENT_VISIT_SHEET_PDF_ENABLED": patient_visit_sheet_pdf_enabled(),
            "PATIENT_NO_HISTORY_MODE_ENABLED": patient_no_history_mode_enabled(),
            "PATIENT_SHOW_PROTOCOL_TECHNICAL_BLOCK": patient_show_protocol_technical_block(),
            "PATIENT_RAG_RETRIEVAL_ENABLED": patient_rag_retrieval_enabled(),
        },
        "upload_formats": {
            "extensions": list(consult_review_allowed_extensions()),
            "accept": consult_review_file_accept_attr(),
            "max_files": env_int("PATIENT_REVIEW_MAX_FILES", 1),
            "hint_ru": consult_review_formats_hint_ru(
                max_files=env_int("PATIENT_REVIEW_MAX_FILES", 1),
            ),
        },
        "payment_required": payment_required(),
        "monetization": mon,
        "tiers": mon.get("tiers") or tier_catalog_public(),
        "disclaimer": (
            "Ориентировочная сверка с клиническими протоколами Минздрава РБ; "
            "не диагноз и не замена очного приёма."
        ),
    }


@app.get("/api/patient/clinic")
def api_patient_clinic(clinic_id: str = "") -> dict:
    from clinical_knowledge.patient_clinic_config import clinic_public_view, resolve_clinic

    clinic = resolve_clinic(clinic_id)
    if not clinic:
        return {"ok": True, "clinic": None}
    return {"ok": True, "clinic": clinic_public_view(clinic)}


@app.get("/api/patient/tiers")
def api_patient_tiers() -> dict:
    from clinical_knowledge.patient_payment import tier_catalog_public

    return {"ok": True, "tiers": tier_catalog_public()}


@app.post("/api/patient/payment/session")
def api_patient_payment_session(body: PatientPaymentSessionIn) -> dict:
    from clinical_knowledge.patient_payment import create_payment_session

    return create_payment_session(tier_id=body.tier_id, clinic_id=body.clinic_id)


@app.post("/api/patient/analytics")
def api_patient_analytics(body: PatientAnalyticsIn) -> dict:
    from clinical_knowledge.patient_analytics import record_patient_event

    return record_patient_event(
        event=body.event,
        clinic_id=body.clinic_id,
        tier_id=body.tier_id,
        meta=body.meta,
        text_hash=body.text_hash,
        build_version=BUILD_VERSION,
    )


@app.post("/api/patient/account/session")
def api_patient_account_session() -> dict:
    from clinical_knowledge.patient_account import create_guest_session

    return create_guest_session()


@app.post("/api/patient/account/sync")
def api_patient_account_sync(body: PatientAccountSyncIn) -> dict:
    from clinical_knowledge.patient_account import sync_history

    return sync_history(body.session_token, body.history)


@app.get("/api/patient/account/history")
def api_patient_account_history(session_token: str = "") -> dict:
    from clinical_knowledge.patient_account import get_history

    return get_history(session_token)


@app.post("/api/patient/review/json")
async def api_patient_review_json(body: PatientReviewJsonIn, request: "Request") -> dict:
    """B2C: проверка текста КЗ (tier P1) без загрузки файла."""
    if not _patient_review_enabled():
        raise HTTPException(status_code=503, detail="Проверка для пациентов временно недоступна.")
    idem = (request.headers.get("Idempotency-Key") or "").strip()
    cached = _patient_idempotency_get(idem)
    if cached is not None:
        out = dict(cached)
        out["idempotent_replay"] = True
        return out
    from clinical_knowledge.patient_review import patient_demographics_from_form

    demo = patient_demographics_from_form(age_years=body.age_years, sex=body.sex)
    _require_patient_payment(body.payment_token, body.tier_id)
    product_tier = _patient_product_tier_from_catalog(body.tier_id)
    t0 = time.perf_counter()
    result = await _run_consult_review_blocking(
        _run_patient_review_core,
        text=body.text.strip(),
        consultation_id="patient-json",
        demographics_meta=demo,
        product_tier=product_tier,
        catalog_tier_id=(body.tier_id or "").strip() or None,
    )
    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    result["build_version"] = BUILD_VERSION
    _patient_idempotency_put(idem, result)
    return result


@app.post("/api/patient/review")
async def api_patient_review(
    request: "Request",
    files: Annotated[
        list[UploadFile],
        File(description="Один файл после приёма: КЗ, медосмотр или консультация (PDF, фото или Word)"),
    ],
    lab_files: Annotated[
        list[UploadFile] | None,
        File(description="Необязательно: фото/PDF бланков анализов"),
    ] = None,
    age_years: str = Form("", description="Возраст пациента (необязательно)"),
    sex: str = Form("", description="Пол: male/female (необязательно)"),
    consent: str = Form("", description="1 - согласие на обработку документа"),
    clinic_id: str = Form("", description="White-label clinic id"),
    tier_id: str = Form("", description="product tier id"),
    payment_token: str = Form("", description="Оплата (если PATIENT_PAYMENT_REQUIRED=1)"),
    question_tone: str = Form("serious", description="Тон вопросов врачу: serious|official|playful"),
) -> dict:
    """B2C: загрузка КЗ пациентом → отчёт P1 (без ЦИСЗ и send_gate)."""
    if not _patient_review_enabled():
        raise HTTPException(status_code=503, detail="Проверка для пациентов временно недоступна.")
    if (consent or "").strip() not in ("1", "true", "yes", "on"):
        raise HTTPException(
            status_code=400,
            detail="Нужно согласие на обработку загруженного документа.",
        )
    idem = (request.headers.get("Idempotency-Key") or "").strip()
    cached = _patient_idempotency_get(idem)
    if cached is not None:
        out = dict(cached)
        out["idempotent_replay"] = True
        return out
    if not files:
        raise HTTPException(status_code=400, detail="Загрузите фото или PDF заключения.")
    max_files = env_int("PATIENT_REVIEW_MAX_FILES", 1)
    if len(files) > max_files:
        raise HTTPException(
            status_code=400,
            detail=f"Не более {max_files} файлов за один запрос.",
        )

    from clinical_knowledge.patient_review import patient_demographics_from_form

    full_text, consult_docs_meta, pdf_warnings, _doc_texts = (
        await _parse_consult_review_uploads_async(files)
    )
    lab_text = ""
    lab_warnings: list[str] = []
    if lab_files:
        lab_text, lab_warnings = await _parse_patient_lab_uploads_async(lab_files)
    demo = patient_demographics_from_form(age_years=age_years, sex=sex)
    _require_patient_payment(payment_token, tier_id)
    product_tier = _patient_product_tier_from_catalog(tier_id)
    t0 = time.perf_counter()
    result = await _run_consult_review_blocking(
        _run_patient_review_core,
        text=full_text,
        consultation_id="patient-upload",
        demographics_meta=demo,
        lab_text=lab_text or None,
        product_tier=product_tier,
        catalog_tier_id=(tier_id or "").strip() or None,
        question_tone=(question_tone or "").strip() or None,
        kz_filename=_first_upload_filename(files),
        lab_filename=_first_upload_filename(lab_files),
    )
    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    _attach_patient_review_telemetry(
        result,
        kz_text=full_text,
        lab_upload=bool(lab_text),
        latency_ms=result["latency_ms"],
    )
    if pdf_warnings:
        result["pdf_warnings"] = pdf_warnings
    if lab_warnings:
        result["lab_warnings"] = lab_warnings
    if consult_docs_meta:
        result["document_count"] = len(consult_docs_meta)
    result["build_version"] = BUILD_VERSION
    if clinic_id.strip():
        result["clinic_id"] = clinic_id.strip()
    if tier_id.strip():
        result["tier_id"] = tier_id.strip()
    _patient_idempotency_put(idem, result)
    return result


@app.post("/api/patient/review/stream")
async def api_patient_review_stream(
    files: Annotated[
        list[UploadFile],
        File(description="Фото или PDF: КЗ, медосмотр или консультация"),
    ],
    lab_files: Annotated[list[UploadFile] | None, File()] = None,
    age_years: str = Form(""),
    sex: str = Form(""),
    consent: str = Form(""),
    clinic_id: str = Form(""),
    tier_id: str = Form(""),
    payment_token: str = Form(""),
    question_tone: str = Form("serious"),
):
    """B2C: SSE-прогресс проверки КЗ."""
    from consult_review_pipeline import sse_encode_done, sse_encode_error, sse_encode_progress
    from clinical_knowledge.patient_review import iter_patient_review_progress, patient_demographics_from_form

    if not _patient_review_enabled():
        raise HTTPException(status_code=503, detail="Проверка для пациентов временно недоступна.")
    if (consent or "").strip() not in ("1", "true", "yes", "on"):
        raise HTTPException(status_code=400, detail="Нужно согласие на обработку документа.")
    if not files:
        raise HTTPException(status_code=400, detail="Загрузите фото или PDF заключения.")
    _require_patient_payment(payment_token, tier_id)
    product_tier = _patient_product_tier_from_catalog(tier_id)

    full_text, _, pdf_warnings, _ = await _parse_consult_review_uploads_async(files)
    lab_text = ""
    if lab_files:
        lab_text, _ = await _parse_patient_lab_uploads_async(lab_files)
    demo = patient_demographics_from_form(age_years=age_years, sex=sex)

    def event_gen():
        yield sse_encode_progress("extract", 20, f"Текст извлечён ({len(full_text)} симв.)")
        try:
            for kind, payload in iter_patient_review_progress(
                text=full_text,
                consultation_id="patient-stream",
                demographics_meta=demo,
                lab_text=lab_text or None,
                product_tier=product_tier,
                question_tone=(question_tone or "").strip() or None,
                kz_filename=_first_upload_filename(files),
                lab_filename=_first_upload_filename(lab_files),
            ):
                if kind == "progress":
                    yield sse_encode_progress(
                        payload["stage"],
                        payload["pct"],
                        payload["label_ru"],
                    )
                elif kind == "done":
                    payload["build_version"] = BUILD_VERSION
                    _attach_patient_review_telemetry(
                        payload,
                        kz_text=full_text,
                        lab_upload=bool(lab_text),
                    )
                    if pdf_warnings:
                        payload["pdf_warnings"] = pdf_warnings
                    if clinic_id.strip():
                        payload["clinic_id"] = clinic_id.strip()
                    if tier_id.strip():
                        payload["tier_id"] = tier_id.strip()
                    yield sse_encode_done(payload)
                    return
                elif kind == "error":
                    yield sse_encode_error(str(payload.get("detail") or "Ошибка"), int(payload.get("status") or 500))
                    return
        except HTTPException as e:
            detail = e.detail if isinstance(e.detail, str) else str(e.detail)
            yield sse_encode_error(detail, e.status_code)
        except Exception as e:
            yield sse_encode_error(str(e)[:400], 500)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Статика (index.html, protocols.json, PDF) - регистрировать после API-маршрутов.
# Иначе GET / даёт 404 «Not Found» на Render при открытии корня в браузере.
if (ROOT / "index.html").is_file():

    @app.get("/", include_in_schema=False)
    def _serve_index_html() -> FileResponse:
        """Без долгого кэша HTML: после деплоя сразу подхватывается новый JS/разметка."""
        return FileResponse(
            path=str(ROOT / "index.html"),
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/methodist", include_in_schema=False)
    def _redirect_methodist() -> RedirectResponse:
        return RedirectResponse(url="/?mode=methodist", status_code=302)

    @app.get("/consult_review.html", include_in_schema=False)
    def _serve_consult_review_html() -> FileResponse:
        p = ROOT / "consult_review.html"
        if not p.is_file():
            raise HTTPException(status_code=404, detail="Страница consult_review.html не найдена")
        return FileResponse(
            path=str(p),
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/patient", include_in_schema=False)
    def _redirect_patient() -> RedirectResponse:
        return RedirectResponse(url="/patient.html", status_code=302)

    @app.get("/check", include_in_schema=False)
    def _redirect_check() -> RedirectResponse:
        return RedirectResponse(url="/patient-check.html", status_code=302)

    @app.get("/patient-check.html", include_in_schema=False)
    def _serve_patient_check_html() -> FileResponse:
        p = ROOT / "patient-check.html"
        if not p.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(path=str(p), media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-cache"})

    @app.get("/patient-tokens.css", include_in_schema=False)
    def _serve_patient_tokens_css() -> FileResponse:
        p = ROOT / "patient-tokens.css"
        if not p.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(
            path=str(p),
            media_type="text/css; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/patient-ui.js", include_in_schema=False)
    def _serve_patient_ui_js() -> FileResponse:
        p = ROOT / "patient-ui.js"
        if not p.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(
            path=str(p),
            media_type="application/javascript; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/proto-viewer.html", include_in_schema=False)
    def _serve_proto_viewer_html() -> Response:
        p = ROOT / "proto-viewer.html"
        if not p.is_file():
            raise HTTPException(status_code=404, detail="Страница proto-viewer.html не найдена")
        html = p.read_text(encoding="utf-8").replace("__BUILD_VERSION__", BUILD_VERSION)
        return Response(
            content=html,
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )

    @app.get("/patient.html", include_in_schema=False)
    def _serve_patient_html() -> Response:
        return _patient_html_response()

    @app.get("/patient-manifest.webmanifest", include_in_schema=False)
    def _serve_patient_manifest() -> FileResponse:
        p = ROOT / "patient-manifest.webmanifest"
        if not p.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(path=str(p), media_type="application/manifest+json")

    @app.get("/patient-sw.js", include_in_schema=False)
    def _serve_patient_sw() -> FileResponse:
        p = ROOT / "patient-sw.js"
        if not p.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(
            path=str(p),
            media_type="application/javascript; charset=utf-8",
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )

    @app.get("/onco-risk.html", include_in_schema=False)
    def _serve_onco_risk_html() -> Response:
        p = ROOT / "onco-risk.html"
        if not p.is_file():
            raise HTTPException(status_code=404, detail="Страница onco-risk.html не найдена")
        html = p.read_text(encoding="utf-8").replace("__BUILD_VERSION__", BUILD_VERSION)
        return Response(
            content=html,
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )

    app.mount(
        "/",
        SafeStaticFiles(directory=str(ROOT), html=True),
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

    # Render/прод: слушаем 0.0.0.0 и порт из $PORT, иначе деплой не пройдёт port scan.
    # Локально без $PORT - 127.0.0.1:8787 (можно переопределить HOST/PORT).
    _port = int(os.environ.get("PORT") or "8787")
    _host = os.environ.get("HOST") or ("0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    print(f"Starting uvicorn on {_host}:{_port}", flush=True)
    uvicorn.run(app, host=_host, port=_port, log_level="info")
