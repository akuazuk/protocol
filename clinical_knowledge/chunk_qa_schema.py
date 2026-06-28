"""Схема ответа LLM для QA rich-чанков."""
from __future__ import annotations

from typing import Any, Literal

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - optional at import time
    BaseModel = object  # type: ignore[misc, assignment]

Verdict = Literal["ok", "fix", "drop", "merge_with_next"]
Obligation = Literal["required", "recommended", "optional", "contraindicated"]


class ChunkQaEntities(BaseModel):
    exam: list[str] = Field(default_factory=list)
    drug: list[str] = Field(default_factory=list)
    condition: list[str] = Field(default_factory=list)


class ChunkQaResult(BaseModel):
    chunk_id: str
    verdict: Verdict = "ok"
    corrected_chunk_type: str | None = None
    corrected_section_title: str | None = None
    clean_text: str | None = None
    obligation: Obligation | None = None
    entities: ChunkQaEntities = Field(default_factory=ChunkQaEntities)
    noise_reasons: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str = ""


class ProtocolSectionRow(BaseModel):
    section_number: str = ""
    section_title: str = ""
    chunk_type: str = "body"
    page_from: int | None = None


class ProtocolSectionsQaResult(BaseModel):
    doc_id: str
    sections: list[ProtocolSectionRow] = Field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""


def parse_chunk_qa_result(data: dict[str, Any]) -> ChunkQaResult | None:
    try:
        return ChunkQaResult.model_validate(data)
    except Exception:
        return None


def parse_protocol_sections_result(data: dict[str, Any]) -> ProtocolSectionsQaResult | None:
    try:
        return ProtocolSectionsQaResult.model_validate(data)
    except Exception:
        return None
