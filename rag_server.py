#!/usr/bin/env python3
"""
Локальный RAG: отбор фрагментов из chunks.json и ответ по ним (ориентир — тексты клинических протоколов).

Запуск: pip install -r requirements-rag.txt, скопировать .env.example в .env и задать ключ API.
Переменные — из .env / .env.local (python-dotenv). См. комментарии в .env.example.

Фронт (index.html) вызывает POST /api/assist; ключ к API не передаётся в браузер.
"""
from __future__ import annotations

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
PROTOCOLS_PATH = ROOT / "protocols.json"

_chunks: list[dict] = []
_protocols_by_path: dict[str, dict] = {}
_routing: dict = {}
_model = None

SYSTEM_JSON = """Ты помощник врача по клиническим протоколам Минздрава Республики Беларусь.
Фрагменты PDF ниже могут быть неполными. Не выдумывай факты вне фрагментов.
Если в запросе есть блок «=== Контекст пациента ===» (возраст, пол и т.д.) и «=== Жалобы и вопрос ===», учитывай контекст при выборе детских vs взрослых протоколов и в формулировке summary.
Если возраст явно взрослый (например «49 лет», ≥18 лет) — не включай в protocols детские КП: в списке должны остаться только path из входных фрагментов; если фрагменты только детские (маловероятно), опирайся на них осторожно и не выдавай детский протокол как основной без пометки.
Клиническая калибровка (обязательно):
- В summary и differential сначала отражай типичные и частые причины (ОРВИ, фарингит, тонзиллит, ларингит), если они согласуются с запросом и фрагментами.
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
Соблюдай клиническую калибровку: частые диагнозы первыми; эпиглоттит и др. редкие неотложные — только при красных флагах или прямо в фрагментах; при нормальном дыхании не веди с эпиглоттита.
Предыдущая попытка оборвалась по длине. Сделай ещё короче:
- summary: РОВНО 2 коротких предложения, ВМЕСТЕ максимум 220 символов, последний символ — точка.
- match_reason: до 55 символов на протокол.
- differential: 2 коротких пункта; questions_for_patient: [] если есть протокол с confidence_score 1.0, иначе 2 коротких вопроса.
Сохрани все path из фрагментов. Не обрывай слова."""


def load_data() -> None:
    global _chunks, _protocols_by_path, _routing
    if not CHUNKS_PATH.is_file():
        raise SystemExit(f"Нет {CHUNKS_PATH}. Запустите: python3 build_chunks.py")
    _chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if PROTOCOLS_PATH.is_file():
        for row in json.loads(PROTOCOLS_PATH.read_text(encoding="utf-8")):
            _protocols_by_path[row["path"]] = row
    rp = ROOT / "symptom_routing.json"
    if rp.is_file():
        _routing = json.loads(rp.read_text(encoding="utf-8"))
    else:
        _routing = {}


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
    """Текст для лексического RAG: только блок «Жалобы и вопрос», без длинного контекста."""
    sep = "=== Жалобы и вопрос ==="
    if sep in full_query:
        part = full_query.split(sep, 1)[1].strip()
        return part if part else full_query.strip()
    return full_query.strip()


def retrieve(
    query: str,
    max_chunks: int | None = None,
    max_per_path: int = 2,
    routing_query: str | None = None,
) -> list[dict]:
    """Лексический отбор + множители из symptom_routing.json (если RAG_ROUTING=1).

    query — короткий текст для подсчёта совпадений с чанками (обычно только жалобы).
    routing_query — полный запрос для правил возраста/рубрик; если None, берётся query.
    """
    if max_chunks is None:
        max_chunks = int(os.environ.get("RAG_MAX_CHUNKS", "6"))
    use_routing = os.environ.get("RAG_ROUTING", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    rq = routing_query if routing_query is not None else query
    qtok = set(tokenize_ru(query))
    if not qtok:
        return []
    scored: list[tuple[float, float, float, dict]] = []
    for ch in _chunks:
        text = (ch.get("text") or "") + " " + (ch.get("title") or "")
        low = text.lower()
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


@app.get("/health")
def health() -> dict:
    has_key = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    return {
        "ok": True,
        "chunks": len(_chunks),
        "protocols": len(_protocols_by_path),
        "gemini_configured": has_key,
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
    retrieved = retrieve(q_rag, routing_query=q)
    if not retrieved:
        raise HTTPException(status_code=400, detail="Пустой отбор — уточните запрос")

    retrieved, audience_inferred, audience_fallback = filter_retrieval_by_audience(
        retrieved, q, _routing
    )

    lines = []
    for i, r in enumerate(retrieved, 1):
        cat = ""
        p = r["path"]
        if p in _protocols_by_path:
            cat = _protocols_by_path[p].get("category") or ""
        lines.append(
            f"[{i}] path={p}\n"
            f"рубрика={cat}\n"
            f"тип_фрагмента={r['kind']}\n"
            f"текст:\n{r['excerpt']}\n"
        )
    context = "\n---\n".join(lines)

    user_block = f"Запрос пользователя:\n{q}\n\nФрагменты протоколов:\n{context}"
    full_prompt = SYSTEM_JSON + "\n\n---\n\n" + user_block
    prompt_limit = int(os.environ.get("GEMINI_PROMPT_MAX_CHARS", "28000"))
    if len(full_prompt) > prompt_limit:
        full_prompt = full_prompt[: prompt_limit - 80] + "\n…[обрезано для лимита контекста]"

    model = get_gemini()
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

    return {
        "query": q,
        "retrieval": retrieved,
        "audience_inferred": audience_inferred,
        "retrieval_audience_fallback": audience_fallback,
        "llm_text": text,
        "llm_json": parsed,
        "gemini_finish_reason": finish,
        "gemini_retry_used": retry_used,
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
