"""LLM-судья action-очереди МО: контракт этапов A/B (чистые функции).

Не пишет в primary warehouse. См. docs/plans/2026-08-05-mo-llm-action-queue-judge-v1.md.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

ENGINE = "mo_llm_action_judge_v1"
SCHEMA_VERSION = 1

VERDICTS = frozenset({"good", "acceptable", "review", "poor", "critical"})
# Частая путаница модели: fit ICD вместо verdict.
VERDICT_ALIASES = {
    "strong": "good",
    "adequate": "acceptable",
    "weak": "poor",
    "ok": "good",
    "fail": "poor",
    "failed": "poor",
    "bad": "poor",
}
OUTCOMES = frozenset({"ok", "gap", "contradiction", "unknown"})
SEVERITIES = frozenset({"P0", "P1", "P2", "P3", "none"})
AUDIENCES = frozenset({"pediatric", "adult", "unknown"})
ICD_FITS = frozenset({"weak", "adequate", "strong", "unknown"})
FOLLOW_KINDS = frozenset({"on_worsening_only", "scheduled", "unclear", "absent"})
AGE_SOURCES = frozenset({"visit_meta", "text", "unknown"})
COMPLETENESS_BLOCKS = (
    "complaints",
    "anamnesis",
    "objective_status",
    "exam_data",
    "diagnosis",
    "exam_recommendations",
    "treatment_recommendations",
)

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)


def _clip(text: str, n: int = 200) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def _as_pct(v: Any) -> int | None:
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return n if 0 <= n <= 100 else None


def _as_conf(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if 0.0 <= x <= 1.0 else None


def _normalize_verdict(raw: Any) -> str:
    verdict = str(raw or "").strip().lower()
    if verdict in VERDICTS:
        return verdict
    return VERDICT_ALIASES.get(verdict, verdict)


def extract_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("пустой ответ модели")
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("в ответе нет JSON-объекта")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("корень JSON должен быть объектом")
    return obj


def _validate_finding(item: Any, *, idx: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"findings[{idx}] не объект")
    sev = str(item.get("severity") or "none").strip()
    if sev not in SEVERITIES:
        raise ValueError(f"findings[{idx}].severity недопустим: {sev}")
    return {
        "code": str(item.get("code") or f"finding_{idx}").strip()[:80],
        "severity": sev,
        "text_ru": _clip(str(item.get("text_ru") or ""), 400),
        "evidence": _clip(str(item.get("evidence") or ""), 200),
    }


def _validate_completeness(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("completeness обязателен")
    score = _as_pct(raw.get("score_pct"))
    if score is None:
        raise ValueError("completeness.score_pct 0-100")
    verdict = _normalize_verdict(raw.get("verdict"))
    if verdict not in VERDICTS:
        raise ValueError(f"completeness.verdict недопустим: {verdict}")
    blocks_in = raw.get("blocks") if isinstance(raw.get("blocks"), dict) else {}
    blocks_out: dict[str, Any] = {}
    for name in COMPLETENESS_BLOCKS:
        block = blocks_in.get(name) if isinstance(blocks_in.get(name), dict) else {}
        blocks_out[name] = {
            "present": bool(block.get("present")),
            "adequate": bool(block.get("adequate")),
            "note": _clip(str(block.get("note") or ""), 160),
        }
    missing = [
        _clip(str(x), 40)
        for x in (raw.get("missing_blocks") or [])
        if str(x).strip() in COMPLETENESS_BLOCKS
    ][:12]
    if not missing:
        missing = [name for name, block in blocks_out.items() if not block["present"]]
    return {
        "score_pct": score,
        "verdict": verdict,
        "blocks": blocks_out,
        "missing_blocks": missing,
        "summary_ru": _clip(str(raw.get("summary_ru") or ""), 400),
    }


def validate_stage_a(raw: dict[str, Any], *, case_id: str | None = None) -> dict[str, Any]:
    if str(raw.get("stage") or "") != "A":
        raise ValueError("stage должен быть A")
    completeness = _validate_completeness(raw.get("completeness"))
    dx = raw.get("diagnosis_assessment")
    if not isinstance(dx, dict):
        raise ValueError("diagnosis_assessment обязателен")
    score = _as_pct(dx.get("score_pct"))
    if score is None:
        raise ValueError("diagnosis_assessment.score_pct 0-100")
    verdict = _normalize_verdict(dx.get("verdict"))
    if verdict not in VERDICTS:
        raise ValueError(f"diagnosis_assessment.verdict недопустим: {verdict}")
    patient = raw.get("patient") if isinstance(raw.get("patient"), dict) else {}
    audience = str(patient.get("audience") or "unknown").strip()
    if audience not in AUDIENCES:
        audience = "unknown"
    age_source = str(patient.get("age_source") or "unknown").strip()
    if age_source not in AGE_SOURCES:
        age_source = "unknown"
    age = patient.get("age_years")
    try:
        age_i = int(age) if age is not None and str(age).strip() != "" else None
    except (TypeError, ValueError):
        age_i = None
    chain_out: list[dict[str, Any]] = []
    for i, link in enumerate(raw.get("chain") or []):
        if not isinstance(link, dict):
            continue
        outcome = str(link.get("outcome") or "unknown").strip()
        if outcome not in OUTCOMES:
            outcome = "unknown"
        chain_out.append(
            {
                "from": str(link.get("from") or "")[:40],
                "to": str(link.get("to") or "")[:40],
                "outcome": outcome,
                "note": _clip(str(link.get("note") or ""), 240),
            }
        )
    icd_in = dx.get("icd") if isinstance(dx.get("icd"), dict) else {}
    fit = str(icd_in.get("fit") or "unknown").strip()
    if fit not in ICD_FITS:
        fit = "unknown"
    conf = _as_conf(raw.get("confidence"))
    if conf is None:
        raise ValueError("confidence 0-1 обязателен")
    findings = [
        _validate_finding(f, idx=i) for i, f in enumerate(raw.get("findings") or []) if isinstance(f, dict)
    ]
    blocks = completeness["blocks"]
    inputs_src = raw.get("inputs_used") if isinstance(raw.get("inputs_used"), dict) else {}
    cid = str(raw.get("case_id") or case_id or "").strip()
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "A",
        "engine": ENGINE,
        "case_id": cid,
        "visit_id": str(raw.get("visit_id") or cid).strip(),
        "mis_id": str(raw.get("mis_id") or "").strip(),
        "patient": {
            "age_years": age_i,
            "audience": audience,
            "age_source": age_source,
        },
        "completeness": completeness,
        "inputs_used": {
            "complaints": bool(inputs_src.get("complaints", blocks["complaints"]["present"])),
            "anamnesis": bool(inputs_src.get("anamnesis", blocks["anamnesis"]["present"])),
            "exams": bool(inputs_src.get("exams", blocks["exam_data"]["present"])),
            "objective_status": bool(
                inputs_src.get("objective_status", blocks["objective_status"]["present"])
            ),
            "diagnosis": bool(inputs_src.get("diagnosis", blocks["diagnosis"]["present"])),
        },
        "diagnosis_assessment": {
            "score_pct": score,
            "verdict": verdict,
            "blocked_by_incomplete": bool(dx.get("blocked_by_incomplete", False)),
            "summary_ru": _clip(str(dx.get("summary_ru") or ""), 500),
            "supported_by": [_clip(str(x), 120) for x in (dx.get("supported_by") or [])[:8]],
            "not_supported_by": [_clip(str(x), 120) for x in (dx.get("not_supported_by") or [])[:8]],
            "icd": {
                "code": _clip(str(icd_in.get("code") or ""), 16),
                "text": _clip(str(icd_in.get("text") or ""), 200),
                "fit": fit,
            },
        },
        "chain": chain_out[:12],
        "findings": findings[:16],
        "conclusion_ru": _clip(str(raw.get("conclusion_ru") or ""), 600),
        "confidence": conf,
        "needs_human": bool(raw.get("needs_human", True)),
    }


def validate_stage_b(raw: dict[str, Any], *, case_id: str | None = None) -> dict[str, Any]:
    if str(raw.get("stage") or "") != "B":
        raise ValueError("stage должен быть B")
    plan = raw.get("plan_assessment")
    if not isinstance(plan, dict):
        raise ValueError("plan_assessment обязателен")
    score = _as_pct(plan.get("score_pct"))
    if score is None:
        raise ValueError("plan_assessment.score_pct 0-100")
    verdict = _normalize_verdict(plan.get("verdict"))
    if verdict not in VERDICTS:
        raise ValueError(f"plan_assessment.verdict недопустим: {verdict}")

    def _block(name: str) -> dict[str, Any]:
        block = plan.get(name) if isinstance(plan.get(name), dict) else {}
        b_score = _as_pct(block.get("score_pct"))
        if b_score is None:
            raise ValueError(f"plan_assessment.{name}.score_pct 0-100")
        b_verdict = _normalize_verdict(block.get("verdict"))
        if b_verdict not in VERDICTS:
            raise ValueError(f"plan_assessment.{name}.verdict недопустим: {b_verdict}")
        out: dict[str, Any] = {
            "score_pct": b_score,
            "verdict": b_verdict,
            "summary_ru": _clip(str(block.get("summary_ru") or ""), 400),
        }
        if name == "exam_recommendations":
            out["present"] = [ _clip(str(x), 120) for x in (block.get("present") or [])[:10] ]
            out["missing_suggested"] = [
                _clip(str(x), 120) for x in (block.get("missing_suggested") or [])[:10]
            ]
        elif name == "treatment_recommendations":
            out["present"] = [ _clip(str(x), 120) for x in (block.get("present") or [])[:10] ]
            out["concerns"] = [ _clip(str(x), 80) for x in (block.get("concerns") or [])[:10] ]
        elif name == "follow_up":
            kind = str(block.get("kind") or "unclear").strip()
            out["kind"] = kind if kind in FOLLOW_KINDS else "unclear"
        return out

    conf = _as_conf(raw.get("confidence"))
    if conf is None:
        raise ValueError("confidence 0-1 обязателен")
    findings = [
        _validate_finding(f, idx=i) for i, f in enumerate(raw.get("findings") or []) if isinstance(f, dict)
    ]
    ref = raw.get("stage_a_ref") if isinstance(raw.get("stage_a_ref"), dict) else {}
    cid = str(raw.get("case_id") or case_id or "").strip()
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "B",
        "engine": ENGINE,
        "case_id": cid,
        "visit_id": str(raw.get("visit_id") or cid).strip(),
        "mis_id": str(raw.get("mis_id") or "").strip(),
        "stage_a_ref": {
            "diagnosis_score_pct": _as_pct(ref.get("diagnosis_score_pct")),
            "diagnosis_verdict": str(ref.get("diagnosis_verdict") or "")[:20],
            "key_gaps": [ _clip(str(x), 80) for x in (ref.get("key_gaps") or [])[:8] ],
        },
        "plan_assessment": {
            "exam_recommendations": _block("exam_recommendations"),
            "treatment_recommendations": _block("treatment_recommendations"),
            "follow_up": _block("follow_up"),
            "score_pct": score,
            "verdict": verdict,
        },
        "findings": findings[:16],
        "conclusion_ru": _clip(str(raw.get("conclusion_ru") or ""), 600),
        "confidence": conf,
        "needs_human": bool(raw.get("needs_human", True)),
    }


def stage_a_digest(stage_a: dict[str, Any]) -> dict[str, Any]:
    dx = stage_a.get("diagnosis_assessment") or {}
    completeness = stage_a.get("completeness") or {}
    gaps = [f.get("code") for f in (stage_a.get("findings") or []) if f.get("code")]
    for name in completeness.get("missing_blocks") or []:
        gaps.append(f"missing:{name}")
    for link in stage_a.get("chain") or []:
        if link.get("outcome") in {"gap", "contradiction"} and link.get("note"):
            gaps.append(str(link.get("from")) + "->" + str(link.get("to")))
    return {
        "completeness_score_pct": completeness.get("score_pct"),
        "completeness_verdict": completeness.get("verdict"),
        "missing_blocks": list(completeness.get("missing_blocks") or [])[:8],
        "diagnosis_score_pct": dx.get("score_pct"),
        "diagnosis_verdict": dx.get("verdict"),
        "blocked_by_incomplete": bool(dx.get("blocked_by_incomplete")),
        "key_gaps": gaps[:8],
        "conclusion_ru": stage_a.get("conclusion_ru") or "",
        "patient": stage_a.get("patient") or {},
    }


def build_prompt_a(case_pack: dict[str, Any]) -> str:
    """case_pack: clinical slots + meta (без сырого result целиком)."""
    meta = case_pack.get("meta") or {}
    slots = case_pack.get("slots") or {}
    block_hint = {"present": True, "adequate": True, "note": ""}
    schema_hint = {
        "schema_version": SCHEMA_VERSION,
        "stage": "A",
        "engine": ENGINE,
        "case_id": meta.get("case_id"),
        "visit_id": meta.get("visit_id"),
        "mis_id": meta.get("mis_id"),
        "patient": {
            "age_years": None,
            "audience": "pediatric|adult|unknown",
            "age_source": "visit_meta|text|unknown",
        },
        "completeness": {
            "score_pct": 0,
            "verdict": "good|acceptable|review|poor|critical",
            "blocks": {name: dict(block_hint) for name in COMPLETENESS_BLOCKS},
            "missing_blocks": [],
            "summary_ru": "",
        },
        "inputs_used": {
            "complaints": True,
            "anamnesis": True,
            "exams": False,
            "objective_status": True,
            "diagnosis": True,
        },
        "diagnosis_assessment": {
            "score_pct": 0,
            "verdict": "good|acceptable|review|poor|critical",
            "blocked_by_incomplete": False,
            "summary_ru": "",
            "supported_by": [],
            "not_supported_by": [],
            "icd": {"code": "", "text": "", "fit": "weak|adequate|strong|unknown"},
        },
        "chain": [
            {
                "from": "complaints",
                "to": "diagnosis",
                "outcome": "ok|gap|contradiction|unknown",
                "note": "",
            }
        ],
        "findings": [{"code": "", "severity": "P0|P1|P2|P3|none", "text_ru": "", "evidence": ""}],
        "conclusion_ru": "",
        "confidence": 0.0,
        "needs_human": True,
    }
    return "\n".join(
        [
            "Ты клинический методист. Этап A: полнота заполнения МО + согласованность диагноза.",
            "Сначала оцени completeness по блокам (жалобы, анамнез, статус, обследования, диагноз, рекомендации).",
            "Затем diagnosis fit (жалобы/анамнез/статус/обследования ↔ диагноз(+МКБ)).",
            "Если критически пусты жалобы/статус/диагноз - blocked_by_incomplete=true, не угадывай Dx.",
            "Не оценивай адекватность плана лечения/дообследования (это этап B) - только наличие блоков.",
            "Не выдумывай факты. Если поля пустые - present=false, adequate=false.",
            "Ответ: один JSON по схеме, без markdown.",
            "Схема:",
            json.dumps(schema_hint, ensure_ascii=False),
            "",
            "Мета:",
            json.dumps(meta, ensure_ascii=False),
            "",
            "Слоты КЗ:",
            json.dumps(slots, ensure_ascii=False),
        ]
    )


def build_prompt_b(case_pack: dict[str, Any], digest_a: dict[str, Any]) -> str:
    meta = case_pack.get("meta") or {}
    slots = case_pack.get("slots") or {}
    plan_slots = {
        "exam_recommendations": slots.get("exam_recommendations") or slots.get("exam_recs") or "",
        "treatment_recommendations": slots.get("treatment_recommendations") or slots.get("treatment_recs") or "",
        "follow_up": slots.get("follow_up") or slots.get("dispensary_info") or "",
        "diagnosis": slots.get("clinical_diagnosis") or slots.get("diagnosis") or "",
        "complaints": slots.get("complaints") or "",
        "objective_status": slots.get("objective_status") or "",
    }
    schema_hint = {
        "schema_version": SCHEMA_VERSION,
        "stage": "B",
        "engine": ENGINE,
        "case_id": meta.get("case_id"),
        "stage_a_ref": digest_a,
        "plan_assessment": {
            "exam_recommendations": {
                "score_pct": 0,
                "verdict": "good|acceptable|review|poor|critical",
                "present": [],
                "missing_suggested": [],
                "summary_ru": "",
            },
            "treatment_recommendations": {
                "score_pct": 0,
                "verdict": "good|acceptable|review|poor|critical",
                "present": [],
                "concerns": [],
                "summary_ru": "",
            },
            "follow_up": {
                "score_pct": 0,
                "verdict": "good|acceptable|review|poor|critical",
                "kind": "on_worsening_only|scheduled|unclear|absent",
                "summary_ru": "",
            },
            "score_pct": 0,
            "verdict": "good|acceptable|review|poor|critical",
        },
        "findings": [{"code": "", "severity": "P0|P1|P2|P3|none", "text_ru": "", "evidence": ""}],
        "conclusion_ru": "",
        "confidence": 0.0,
        "needs_human": True,
    }
    return "\n".join(
        [
            "Ты клинический методист. Этап B: адекватность плана обследования и лечения.",
            "Учти итог этапа A (stage_a_ref). Не переоценивай диагноз заново.",
            "Оцени: рекомендации по обследованию, лечению, follow-up.",
            "Ответ: один JSON по схеме, без markdown.",
            "Схема:",
            json.dumps(schema_hint, ensure_ascii=False),
            "",
            "Итог этапа A:",
            json.dumps(digest_a, ensure_ascii=False),
            "",
            "Мета:",
            json.dumps(meta, ensure_ascii=False),
            "",
            "Слоты плана/клиники:",
            json.dumps(plan_slots, ensure_ascii=False),
        ]
    )


EXAMPLE_STAGE_A: dict[str, Any] = {
    "schema_version": 1,
    "stage": "A",
    "engine": ENGINE,
    "case_id": "3646270",
    "visit_id": "3646270",
    "mis_id": "898517",
    "patient": {"age_years": 9, "audience": "pediatric", "age_source": "text"},
    "completeness": {
        "score_pct": 70,
        "verdict": "review",
        "blocks": {
            "complaints": {"present": True, "adequate": True, "note": ""},
            "anamnesis": {"present": True, "adequate": False, "note": "нет динамики"},
            "objective_status": {"present": True, "adequate": True, "note": ""},
            "exam_data": {"present": False, "adequate": False, "note": "пусто"},
            "diagnosis": {"present": True, "adequate": True, "note": ""},
            "exam_recommendations": {"present": True, "adequate": True, "note": ""},
            "treatment_recommendations": {"present": True, "adequate": True, "note": ""},
        },
        "missing_blocks": ["exam_data"],
        "summary_ru": "Клиника заполнена; блок данных обследований пуст.",
    },
    "inputs_used": {
        "complaints": True,
        "anamnesis": True,
        "exams": False,
        "objective_status": True,
        "diagnosis": True,
    },
    "diagnosis_assessment": {
        "score_pct": 35,
        "verdict": "poor",
        "blocked_by_incomplete": False,
        "summary_ru": "Диагноз не закрывает суставную находку и хроническую хромоту.",
        "supported_by": ["болезненность мышцы бедра"],
        "not_supported_by": ["отёк колена", "хромота 3 месяца"],
        "icd": {"code": "M60", "text": "Миозит", "fit": "weak"},
    },
    "chain": [
        {
            "from": "complaints",
            "to": "diagnosis",
            "outcome": "gap",
            "note": "Хромота/сустав не в Dx",
        }
    ],
    "findings": [
        {
            "code": "finding_not_in_diagnosis",
            "severity": "P1",
            "text_ru": "Отёк колена не в диагнозе",
            "evidence": "отёк правого коленного сустава",
        }
    ],
    "conclusion_ru": "Полнота средняя; диагноз слабо согласован с клиникой.",
    "confidence": 0.78,
    "needs_human": True,
}

EXAMPLE_STAGE_B: dict[str, Any] = {
    "schema_version": 1,
    "stage": "B",
    "engine": ENGINE,
    "case_id": "3646270",
    "visit_id": "3646270",
    "mis_id": "898517",
    "stage_a_ref": {
        "diagnosis_score_pct": 35,
        "diagnosis_verdict": "poor",
        "key_gaps": ["finding_not_in_diagnosis"],
    },
    "plan_assessment": {
        "exam_recommendations": {
            "score_pct": 20,
            "verdict": "poor",
            "present": [],
            "missing_suggested": ["УЗИ коленного сустава"],
            "summary_ru": "Обследований нет.",
        },
        "treatment_recommendations": {
            "score_pct": 40,
            "verdict": "review",
            "present": ["ибупрофен"],
            "concerns": ["underworkup_before_therapy"],
            "summary_ru": "Симптоматика без дообследования.",
        },
        "follow_up": {
            "score_pct": 15,
            "verdict": "poor",
            "kind": "on_worsening_only",
            "summary_ru": "Только при ухудшении.",
        },
        "score_pct": 25,
        "verdict": "poor",
    },
    "findings": [
        {
            "code": "underworkup_chronic_red_flag",
            "severity": "P1",
            "text_ru": "Нет imaging/labs",
            "evidence": "при отрицательной динамике",
        }
    ],
    "conclusion_ru": "Сначала дообследование, затем терапия.",
    "confidence": 0.74,
    "needs_human": True,
}


def _judge_data_roots() -> list[Path]:
    roots: list[Path] = []
    configured = (os.environ.get("MO_DATA_ROOT") or "").strip()
    if configured:
        roots.append(Path(configured))
    roots.append(Path(__file__).resolve().parents[1] / "data" / "medical_exams")
    var = Path("/var/data/medical_exams")
    if var.is_dir():
        roots.append(var)
    return roots


def judge_jsonl_path(day: str, *, root: Path | None = None) -> Path:
    y, m, d = str(day)[:10].split("-")
    base = root or _judge_data_roots()[0]
    return base / "llm_action_judge" / y / m / d / "judges.jsonl"


def load_llm_action_judge_row(
    case_id: str,
    *,
    visit_date: str,
    roots: list[Path] | None = None,
) -> dict[str, Any] | None:
    """Найти shadow-строку judges.jsonl по case_id за дату визита."""
    cid = str(case_id or "").strip()
    day = str(visit_date or "").strip()[:10]
    if not cid or len(day) != 10:
        return None
    for root in roots or _judge_data_roots():
        path = judge_jsonl_path(day, root=root)
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        continue
                    keys = {
                        str(row.get("case_id") or "").strip(),
                        str(row.get("visit_id") or "").strip(),
                        str(row.get("mis_id") or "").strip(),
                    }
                    if cid in keys:
                        return row
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _kpi_payload(score_pct: Any, verdict: Any, summary_ru: Any) -> dict[str, Any]:
    return {
        "score_pct": _as_pct(score_pct),
        "verdict": str(verdict or "").strip() or None,
        "summary_ru": _clip(str(summary_ru or ""), 400),
    }


def summarize_llm_action_judge_for_ui(row: dict[str, Any] | None) -> dict[str, Any]:
    """Компактный payload для case detail: 3 KPI + выводы (shadow)."""
    if not row:
        return {
            "available": False,
            "shadow": True,
            "engine": ENGINE,
            "reason": "LLM-оценка action-очереди ещё не готова для этого случая",
        }
    if row.get("error") and not row.get("stage_a") and not row.get("stage_b"):
        return {
            "available": False,
            "shadow": True,
            "engine": ENGINE,
            "reason": _clip(str(row.get("error") or "ошибка прогона"), 240),
            "error": _clip(str(row.get("error") or ""), 240),
        }
    stage_a = row.get("stage_a") if isinstance(row.get("stage_a"), dict) else {}
    stage_b = row.get("stage_b") if isinstance(row.get("stage_b"), dict) else {}
    completeness = stage_a.get("completeness") if isinstance(stage_a.get("completeness"), dict) else {}
    diagnosis = (
        stage_a.get("diagnosis_assessment")
        if isinstance(stage_a.get("diagnosis_assessment"), dict)
        else {}
    )
    plan = stage_b.get("plan_assessment") if isinstance(stage_b.get("plan_assessment"), dict) else {}
    findings: list[dict[str, Any]] = []
    for src in (stage_a.get("findings") or [], stage_b.get("findings") or []):
        for item in src:
            if not isinstance(item, dict):
                continue
            findings.append(
                {
                    "code": str(item.get("code") or "")[:80],
                    "severity": str(item.get("severity") or "none")[:8],
                    "text_ru": _clip(str(item.get("text_ru") or ""), 240),
                    "evidence": _clip(str(item.get("evidence") or ""), 160),
                }
            )
            if len(findings) >= 8:
                break
        if len(findings) >= 8:
            break
    missing = [_clip(str(x), 40) for x in (completeness.get("missing_blocks") or [])[:8]]
    return {
        "available": True,
        "shadow": True,
        "engine": ENGINE,
        "case_id": str(row.get("case_id") or row.get("visit_id") or "").strip(),
        "mis_id": str(row.get("mis_id") or "").strip(),
        "date": str(row.get("date") or "").strip()[:10],
        "models": {"a": row.get("model_a"), "b": row.get("model_b")},
        "kpis": {
            "completeness": _kpi_payload(
                completeness.get("score_pct"),
                completeness.get("verdict"),
                completeness.get("summary_ru"),
            ),
            "diagnosis": _kpi_payload(
                diagnosis.get("score_pct"),
                diagnosis.get("verdict"),
                diagnosis.get("summary_ru"),
            ),
            "recommendations": _kpi_payload(
                plan.get("score_pct"),
                plan.get("verdict"),
                plan.get("summary_ru")
                or (
                    (plan.get("exam_recommendations") or {}).get("summary_ru")
                    if isinstance(plan.get("exam_recommendations"), dict)
                    else ""
                ),
            ),
        },
        "conclusions": {
            "completeness_ru": _clip(str(completeness.get("summary_ru") or ""), 400),
            "diagnosis_ru": _clip(str(diagnosis.get("summary_ru") or stage_a.get("conclusion_ru") or ""), 400),
            "recommendations_ru": _clip(
                str(plan.get("summary_ru") or stage_b.get("conclusion_ru") or ""),
                400,
            ),
            "missing_blocks": missing,
            "blocked_by_incomplete": bool(diagnosis.get("blocked_by_incomplete")),
            "findings": findings,
        },
        "error": _clip(str(row.get("error") or ""), 240) or None,
    }


def load_llm_action_judge_for_case(
    case_id: str,
    *,
    visit_date: str,
    roots: list[Path] | None = None,
) -> dict[str, Any]:
    return summarize_llm_action_judge_for_ui(
        load_llm_action_judge_row(case_id, visit_date=visit_date, roots=roots)
    )
