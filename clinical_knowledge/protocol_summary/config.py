"""Конфигурация Protocol Summary Cards из env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

AnalysisMode = Literal["legacy", "summary", "hybrid"]
ReviewStatusMin = Literal["draft", "reviewed", "approved"]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_mode() -> AnalysisMode:
    raw = (os.environ.get("PROTOCOL_SUMMARY_MODE") or "legacy").strip().lower()
    if raw in ("legacy", "summary", "hybrid"):
        return raw  # type: ignore[return-value]
    return "legacy"


def _env_review_min() -> ReviewStatusMin:
    raw = (os.environ.get("PROTOCOL_SUMMARY_MIN_REVIEW_STATUS") or "draft").strip().lower()
    if raw in ("draft", "reviewed", "approved"):
        return raw  # type: ignore[return-value]
    return "draft"


@dataclass(frozen=True)
class ProtocolSummaryConfig:
    enabled: bool
    mode: AnalysisMode
    strict_validation: bool
    fallback_to_legacy: bool
    compare_with_legacy: bool
    generate_rules: bool
    generate_rag: bool
    min_review_status: ReviewStatusMin
    data_root: str

    @classmethod
    def from_env(cls) -> ProtocolSummaryConfig:
        root = os.environ.get("PROTOCOL_SUMMARY_DATA_ROOT") or "data/protocol_summaries"
        return cls(
            enabled=_env_bool("PROTOCOL_SUMMARY_ENABLED", False),
            mode=_env_mode(),
            strict_validation=_env_bool("PROTOCOL_SUMMARY_STRICT_VALIDATION", True),
            fallback_to_legacy=_env_bool("PROTOCOL_SUMMARY_FALLBACK_TO_LEGACY", True),
            compare_with_legacy=_env_bool("PROTOCOL_SUMMARY_COMPARE_WITH_LEGACY", True),
            generate_rules=_env_bool("PROTOCOL_SUMMARY_GENERATE_RULES", True),
            generate_rag=_env_bool("PROTOCOL_SUMMARY_GENERATE_RAG", True),
            min_review_status=_env_review_min(),
            data_root=root,
        )


protocol_summary_config = ProtocolSummaryConfig.from_env()
