"""Token and cost accounting for MO LLM evaluations."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parent.parent
PRICING_PATH = ROOT / "config" / "llm_pricing.yaml"


@lru_cache(maxsize=1)
def load_pricing() -> dict[str, Any]:
    return yaml.safe_load(PRICING_PATH.read_text(encoding="utf-8")) or {}


def calculate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = (load_pricing().get("prices_per_million_tokens") or {}).get(model)
    if not prices:
        raise ValueError(f"llm_price_missing:{model}")
    long_context = prompt_tokens > int(prices.get("long_context_threshold") or 10**18)
    input_rate = float(
        prices.get("long_context_input") if long_context else prices["input"]
    )
    output_rate = float(
        prices.get("long_context_output") if long_context else prices["output"]
    )
    return round(
        prompt_tokens * input_rate / 1_000_000
        + completion_tokens * output_rate / 1_000_000,
        8,
    )


def response_usage(response: Any) -> tuple[int, int]:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return 0, 0
    prompt = getattr(metadata, "prompt_token_count", None)
    completion = getattr(metadata, "candidates_token_count", None)
    if isinstance(metadata, Mapping):
        prompt = metadata.get("prompt_token_count", prompt)
        completion = metadata.get("candidates_token_count", completion)
    return int(prompt or 0), int(completion or 0)


def record_llm_usage(
    db_path: Path,
    *,
    run_id: str,
    tier: str,
    model: str,
    case_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    status: str,
    retry_count: int = 0,
) -> dict[str, Any]:
    from .mo_daily import initialize_warehouse

    initialize_warehouse(db_path)
    now = datetime.now(timezone.utc).isoformat()
    usage_id = hashlib.sha256(
        f"{run_id}:{tier}:{model}:{case_id}:{now}".encode("utf-8")
    ).hexdigest()[:32]
    try:
        cost = calculate_cost_usd(model, prompt_tokens, completion_tokens)
        pricing_status = "priced"
    except ValueError:
        cost = 0.0
        pricing_status = "unpriced"
    payload = {
        "usage_id": usage_id,
        "run_id": run_id,
        "usage_date": now[:10],
        "tier": tier,
        "model": model,
        "case_id": case_id,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "cost_usd": cost,
        "latency_ms": int(latency_ms),
        "status": status if pricing_status == "priced" else f"{status}:{pricing_status}",
        "retry_count": int(retry_count),
        "created_at": now,
    }
    with sqlite3.connect(db_path) as db:
        db.execute(
            """INSERT INTO fact_llm_usage
               (usage_id,run_id,usage_date,tier,model,case_id,prompt_tokens,
                completion_tokens,cost_usd,latency_ms,status,retry_count,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            tuple(payload.values()),
        )
        db.execute(
            """UPDATE fact_mo_case SET llm_cost_usd=(
                 SELECT COALESCE(SUM(cost_usd),0) FROM fact_llm_usage
                 WHERE case_id=?
               ) WHERE mis_id=? OR visit_id=?""",
            (case_id, case_id, case_id),
        )
    return json.loads(json.dumps(payload))
