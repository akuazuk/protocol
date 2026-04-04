#!/usr/bin/env python3
"""
Локальный RAG: отбор фрагментов из chunks.json + ответ Gemini (диагноз/протоколы — как ориентир по текстам КП).

Запуск:
  export GOOGLE_API_KEY="ваш_ключ"
  # опционально: export GEMINI_MODEL=gemini-2.5-flash
  # опционально: export GEMINI_CALL_TIMEOUT=120
  pip install -r requirements-rag.txt
  uvicorn rag_server:app --host 127.0.0.1 --port 8787 --reload

По умолчанию модель gemini-2.0-flash; вызов к API ограничен по времени (см. GEMINI_CALL_TIMEOUT).

Фронт (index.html) дергает POST /api/assist — ключ в браузер не передаётся.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError as e:
    raise SystemExit(f"Установите: pip install -r requirements-rag.txt ({e})") from e

CHUNKS_PATH = ROOT / "chunks.json"
PROTOCOLS_PATH = ROOT / "protocols.json"

_chunks: list[dict] = []
_protocols_by_path: dict[str, dict] = {}
_model = None

SYSTEM_JSON = """Ты помощник врача по клиническим протоколам Минздрава Республики Беларусь.
Ниже пользователь даёт жалобы/ситуацию и автоматически отобранные фрагменты PDF (могут быть неполными). Не выдумывай факты вне фрагментов.
Ответь СТРОГО валидным JSON (без markdown-ограждений) со схемой:
{
  "summary": "кратко: что может соответствовать жалобе (2–4 предложения, осторожно формулируй)",
  "protocols": [
    {"path": "относительный путь к pdf как в данных", "title": "читаемое название", "match_reason": "почему релевантен", "confidence": "низкая|средняя|высокая"}
  ],
  "differential": ["1–3 дифференциальных направления для обсуждения с врачом"],
  "questions_for_patient": ["1–3 уточняющих вопроса"],
  "disclaimer": "Информация из протоколов; не замена очной консультации и не медицинское заключение."
}
Поле protocols — только из path, присутствующих во входных фрагментах. Если данных мало — снизь confidence и скажи об этом в summary."""


def load_data() -> None:
    global _chunks, _protocols_by_path
    if not CHUNKS_PATH.is_file():
        raise SystemExit(f"Нет {CHUNKS_PATH}. Запустите: python3 build_chunks.py")
    _chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if PROTOCOLS_PATH.is_file():
        for row in json.loads(PROTOCOLS_PATH.read_text(encoding="utf-8")):
            _protocols_by_path[row["path"]] = row


def tokenize_ru(s: str) -> list[str]:
    s = s.lower().replace("ё", "е")
    return [t for t in re.findall(r"[а-яa-z]{2,}", s) if len(t) >= 2]


def retrieve(query: str, max_chunks: int = 12, max_per_path: int = 3) -> list[dict]:
    """Простой лексический отбор без тяжёлых зависимостей."""
    qtok = set(tokenize_ru(query))
    if not qtok:
        return []
    scored: list[tuple[float, dict]] = []
    for ch in _chunks:
        text = (ch.get("text") or "") + " " + (ch.get("title") or "")
        low = text.lower()
        score = 0.0
        for t in qtok:
            if t in low:
                score += 1.0 + min(len(t), 10) * 0.02
        if score > 0:
            scored.append((score, ch))
    scored.sort(key=lambda x: -x[0])

    per_path: dict[str, int] = {}
    out: list[dict] = []
    for sc, ch in scored:
        p = ch.get("path") or ""
        if per_path.get(p, 0) >= max_per_path:
            continue
        per_path[p] = per_path.get(p, 0) + 1
        out.append(
            {
                "path": p,
                "title": ch.get("title") or "",
                "kind": ch.get("kind") or "general",
                "score": round(sc, 3),
                "excerpt": (ch.get("text") or "")[:900],
            }
        )
        if len(out) >= max_chunks:
            break
    return out


GEMINI_CALL_TIMEOUT = float(os.environ.get("GEMINI_CALL_TIMEOUT", "90"))


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
    name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
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


def _generate_blocking(model, full_prompt: str):
    return model.generate_content(
        full_prompt,
        generation_config={
            "temperature": 0.25,
            "max_output_tokens": 2048,
        },
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
                detail=f"Таймаут Gemini ({int(GEMINI_CALL_TIMEOUT)} с). Проверьте сеть или GEMINI_MODEL.",
            ) from e


load_data()
app = FastAPI(title="Protocol RAG", version="1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AssistIn(BaseModel):
    query: str = Field(..., min_length=2, max_length=4000)


@app.get("/health")
def health() -> dict:
    has_key = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    return {
        "ok": True,
        "chunks": len(_chunks),
        "protocols": len(_protocols_by_path),
        "gemini_configured": has_key,
    }


@app.post("/api/assist")
def api_assist(body: AssistIn) -> dict:
    q = body.query.strip()
    retrieved = retrieve(q)
    if not retrieved:
        raise HTTPException(status_code=400, detail="Пустой отбор — уточните запрос")

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
    if len(full_prompt) > 28000:
        full_prompt = full_prompt[:27900] + "\n…[обрезано для лимита]"

    model = get_gemini()
    try:
        resp = generate_gemini(model, full_prompt)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini: {e!s}") from e

    pf = getattr(resp, "prompt_feedback", None)
    if pf is not None and getattr(pf, "block_reason", None):
        raise HTTPException(
            status_code=502,
            detail=f"Запрос отклонён моделью: {pf.block_reason}",
        )

    text = _extract_gemini_text(resp)
    if not text:
        raise HTTPException(
            status_code=502,
            detail="Пустой ответ Gemini (блокировка контента или сбой). Попробуйте другую формулировку.",
        )

    parsed = None
    raw = text
    t = text
    if "```" in t:
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.M)
        t = re.sub(r"\s*```\s*$", "", t, flags=re.M)
    try:
        parsed = json.loads(t)
    except json.JSONDecodeError:
        parsed = None

    return {
        "query": q,
        "retrieval": retrieved,
        "llm_text": raw,
        "llm_json": parsed,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8787)
