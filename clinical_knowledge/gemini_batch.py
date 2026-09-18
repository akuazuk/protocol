"""Gemini Batch API (50% off). Best-effort: любой сбой -> None, вызывающий идёт sequential."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from clinical_knowledge.gemini_lite import (
    json_generation_config_dict,
    json_max_output_tokens,
)


BATCH_CREATE_URL = "https://generativelanguage.googleapis.com/v1beta/batches"


def use_night_batch() -> bool:
    raw = (os.environ.get("GEMINI_NIGHT_USE_BATCH") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def rest_generation_config(
    *,
    max_output_tokens: int | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """REST-имена полей (camelCase) для Batch / generateContent."""
    local = json_generation_config_dict(
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    return {
        "temperature": local["temperature"],
        "maxOutputTokens": local["max_output_tokens"],
        "candidateCount": local["candidate_count"],
        "responseMimeType": local["response_mime_type"],
        "thinkingConfig": {"thinkingBudget": int(local["thinking_config"]["thinking_budget"])},
    }


def inlined_request(prompt: str, *, max_output_tokens: int | None = None) -> dict[str, Any]:
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": rest_generation_config(max_output_tokens=max_output_tokens),
    }


def batch_create_body(model: str, prompts: list[str], *, display_name: str) -> dict[str, Any]:
    name = model if model.startswith("models/") else f"models/{model}"
    max_out = json_max_output_tokens()
    return {
        "batch": {
            "displayName": display_name,
            "model": name,
            "inputConfig": {
                "requests": {
                    "requests": [
                        inlined_request(prompt, max_output_tokens=max_out) for prompt in prompts
                    ]
                }
            },
        }
    }


def extract_text_from_generate_content_response(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    candidates = payload.get("candidates") or []
    parts_out: list[str] = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        content = cand.get("content") or {}
        for part in content.get("parts") or []:
            if isinstance(part, dict) and part.get("text"):
                parts_out.append(str(part["text"]))
    return "".join(parts_out).strip()


def usage_from_generate_content_response(payload: Any) -> tuple[int, int]:
    if not isinstance(payload, dict):
        return 0, 0
    meta = payload.get("usageMetadata") or payload.get("usage_metadata") or {}
    if not isinstance(meta, dict):
        return 0, 0
    prompt = int(meta.get("promptTokenCount") or meta.get("prompt_token_count") or 0)
    completion = int(
        meta.get("candidatesTokenCount") or meta.get("candidates_token_count") or 0
    )
    thoughts = int(meta.get("thoughtsTokenCount") or meta.get("thoughts_token_count") or 0)
    return prompt, completion + thoughts


def _http_json(
    url: str,
    *,
    api_key: str,
    payload: dict[str, Any] | None = None,
    method: str = "GET",
    timeout: float = 60.0,
) -> dict[str, Any]:
    data = None
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _batch_name(created: dict[str, Any]) -> str:
    batch = created.get("batch") if isinstance(created.get("batch"), dict) else created
    name = str(batch.get("name") or created.get("name") or "")
    return name


def _batch_state(body: dict[str, Any]) -> str:
    batch = body.get("batch") if isinstance(body.get("batch"), dict) else body
    meta = batch.get("metadata") if isinstance(batch.get("metadata"), dict) else {}
    return str(
        batch.get("state")
        or meta.get("state")
        or body.get("done")
        or ""
    ).upper()


def _inlined_responses(body: dict[str, Any]) -> list[Any]:
    batch = body.get("batch") if isinstance(body.get("batch"), dict) else body
    dest = batch.get("output") or batch.get("dest") or body.get("response") or {}
    if not isinstance(dest, dict):
        return []
    inline = (
        dest.get("inlinedResponses")
        or dest.get("inlined_responses")
        or dest.get("responses")
        or []
    )
    if isinstance(inline, dict):
        inline = inline.get("inlinedResponses") or inline.get("responses") or []
    return list(inline) if isinstance(inline, list) else []


def try_batch_generate(
    model: str,
    prompts: list[str],
    *,
    api_key: str,
    display_name: str = "mo-night-grade",
    poll_seconds: float | None = None,
) -> list[dict[str, Any]] | None:
    """Список {text, prompt_tokens, completion_tokens} или None (вызывающий идёт sequential)."""
    if not prompts or not api_key:
        return None
    timeout = poll_seconds
    if timeout is None:
        timeout = float(os.environ.get("GEMINI_BATCH_POLL_SECONDS", "900"))
    try:
        created = _http_json(
            BATCH_CREATE_URL,
            api_key=api_key,
            payload=batch_create_body(model, prompts, display_name=display_name),
        )
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    name = _batch_name(created)
    if not name:
        # Некоторые ответы сразу содержат inlined responses.
        inline = _inlined_responses(created)
        if len(inline) == len(prompts):
            return [_parse_inline_row(row) for row in inline]
        return None
    url = (
        name
        if name.startswith("http")
        else f"https://generativelanguage.googleapis.com/v1beta/{name.lstrip('/')}"
    )
    deadline = time.time() + timeout
    body = created
    while time.time() < deadline:
        state = _batch_state(body)
        if "SUCCEED" in state or "COMPLETE" in state or body.get("done") is True:
            rows = _inlined_responses(body)
            if len(rows) != len(prompts):
                return None
            return [_parse_inline_row(row) for row in rows]
        if any(token in state for token in ("FAIL", "CANCEL", "ERROR", "EXPIRED")):
            return None
        time.sleep(5)
        try:
            body = _http_json(url, api_key=api_key, method="GET")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None
    return None


def _parse_inline_row(row: Any) -> dict[str, Any]:
    payload = row
    if isinstance(row, dict):
        payload = row.get("response") or row.get("generateContentResponse") or row
    text = extract_text_from_generate_content_response(payload)
    prompt_tokens, completion_tokens = usage_from_generate_content_response(payload)
    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
