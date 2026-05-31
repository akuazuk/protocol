"""Draft-генерация Protocol Summary из protocol_cards + chunks (478 PDF)."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config as _cfg_mod
from .loader import export_summary_json
from .schema import (
    ConditionSummary,
    CriteriaBlock,
    CriterionItem,
    DrugGroupItem,
    DrugTreatmentItem,
    ExamRequirement,
    ExtractionMetadata,
    FollowUpRequirement,
    NonDrugTreatmentItem,
    ProtocolApplicability,
    ProtocolRubric,
    ProtocolSource,
    ProtocolSummary,
    RedFlagItem,
    SummarySourceRef,
    TreatmentBlock,
)
from .summary_to_rag import write_summary_rag_jsonl
from .summary_quality import write_summary_quality_report
from .validator import validate_protocol_summary, write_validation_report

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "output" / "registry" / "protocol_cards.jsonl"
CHUNKS = ROOT / "output" / "chunks" / "chunks.jsonl"

_EXAM_RE = re.compile(
    r"\b(ЭГДС|ФГДС|эзофагогастродуоденоскоп|УЗИ|КТ|МРТ|колоноскоп|"
    r"ОАК|ЭКГ|биопси|анализ кров|анти-?DNA|ANA|Helicobacter|HPV)\b",
    re.I,
)
_RED_FLAG_RE = [
    (re.compile(r"опухолев|злокачествен|не\s+исключить\s+инваз|подозрени[ея]\s+на\s+зло", re.I),
     "possible_malignancy", "critical"),
    (re.compile(r"флеботромб|тромбоз|ТЭЛА|эмболи", re.I), "thrombosis", "high"),
    (re.compile(r"анaphyl|анафилак|тяжел.*инфек", re.I), "severe_infection", "high"),
    (re.compile(r"системн.*аутоиммун|волчанк", re.I), "systemic_autoimmune", "medium"),
    (re.compile(r"гепатотокс|нефротокс|лекарственн.*безопас", re.I), "drug_safety", "medium"),
    (re.compile(r"неотложн|экстренн.*направ|госпитализац.*необход", re.I), "urgent_referral", "high"),
]
_FOLLOW_UP_RE = re.compile(
    r"(контроль|повторн.*консульт|через\s+\d+\s+(?:дн|нед|мес)|динамическ.*наблюден)",
    re.I,
)
_DRUG_DOSE_RE = re.compile(
    r"([а-яёa-z\-]{4,30})\s+(\d+(?:[.,]\d+)?\s*(?:мг|мкг|г|мл))\s*([^.;]{0,40})?",
    re.I,
)
_GENERIC_TITLES = frozenset({"клинический протокол", "клинический протокол.", "протокол"})


def _slug(s: str, max_len: int = 72) -> str:
    t = re.sub(r"[^a-z0-9а-яё]+", "_", (s or "").lower()).strip("_")
    return t[:max_len] or "protocol"


def _norm_icd(code: str) -> str:
    return re.sub(r"\s+", "", (code or "").upper().strip())


def _title_from_path(path: str) -> str:
    stem = Path(path).stem
    t = re.sub(r"[_]+", " ", stem)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:240] or "Клинический протокол"


def _protocol_id_from_path(path: str) -> str:
    parts = Path(path).parts
    slug = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    return _slug(slug.replace(".pdf", ""), 80)


def _chunk_ref(protocol_id: str, ch: dict[str, Any], quote: str = "") -> SummarySourceRef:
    txt = quote or (ch.get("text") or "")[:300]
    return SummarySourceRef(
        protocol_id=protocol_id,
        local_path=ch.get("source_path"),
        page_start=ch.get("page_from"),
        page_end=ch.get("page_to"),
        section_title=ch.get("section_title"),
        section_type=ch.get("chunk_type"),
        quote=txt.strip()[:400] if txt.strip() else "…",
    )


def _population_from_cards(cards: list[dict[str, Any]]) -> list[str]:
    pops: set[str] = set()
    for c in cards:
        p = str(c.get("population") or "any").lower()
        if p == "child":
            pops.add("child")
        elif p == "adult":
            pops.add("adult")
        else:
            pops.update(["adult", "child"])
    return sorted(pops) or ["unknown"]


def _merge_approval(cards: list[dict[str, Any]]) -> dict[str, Any]:
    for c in cards:
        ap = c.get("approval") or {}
        if isinstance(ap, dict) and (ap.get("number") or ap.get("date")):
            return ap
    return {}


def _conditions_from_cards_and_chunks(
    protocol_id: str,
    cards: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    pdf_title: str,
) -> list[ConditionSummary]:
    icd_to_name: dict[str, str] = defaultdict(str)
    for c in cards:
        for code in c.get("icd10_primary") or []:
            icd = _norm_icd(str(code))
            if icd:
                title = str(c.get("title") or "").strip()
                if title.lower() not in _GENERIC_TITLES:
                    icd_to_name[icd] = title
                elif not icd_to_name[icd]:
                    icd_to_name[icd] = pdf_title
        for code in (c.get("icd10_all") or [])[:12]:
            icd = _norm_icd(str(code))
            if icd and icd not in icd_to_name:
                icd_to_name[icd] = pdf_title

    for ch in chunks:
        for code in ch.get("icd10_codes") or []:
            icd = _norm_icd(str(code))
            if icd and re.match(r"^[A-Z]\d", icd) and icd not in icd_to_name:
                icd_to_name[icd] = pdf_title

    if not icd_to_name:
        icd_to_name[_slug(pdf_title)[:12].upper() or "COND"] = pdf_title

    blob = " ".join(ch.get("text") or "" for ch in chunks)[:50000]
    conditions: list[ConditionSummary] = []

    for icd, name in sorted(icd_to_name.items())[:24]:
        if not re.match(r"^[A-Z]\d", icd):
            continue
        cid = _slug(f"{icd}_{name}")[:56]
        cond_chunks = [
            ch for ch in chunks
            if icd in {_norm_icd(str(x)) for x in (ch.get("icd10_codes") or [])}
            or icd.lower() in (ch.get("text") or "").lower()
        ] or chunks[:40]

        exams: dict[str, ExamRequirement] = {}
        for ch in cond_chunks:
            for proc in ch.get("procedures") or []:
                if isinstance(proc, str) and len(proc) > 2:
                    key = proc.lower()[:40]
                    if key not in exams:
                        exams[key] = ExamRequirement(
                            name=proc.strip(),
                            exam_type="instrumental",
                            requirement_level="recommended",
                            source_ref=_chunk_ref(protocol_id, ch, proc),
                        )
            for m in _EXAM_RE.finditer(ch.get("text") or ""):
                label = m.group(1).upper()
                if label == "ФГДС":
                    label = "ЭГДС"
                key = label.lower()
                if key not in exams:
                    exams[key] = ExamRequirement(
                        name=label,
                        exam_type="instrumental" if label in ("ЭГДС", "КТ", "МРТ", "УЗИ") else "laboratory",
                        requirement_level="recommended",
                        source_ref=_chunk_ref(protocol_id, ch, m.group(0)),
                    )

        drugs: list[DrugTreatmentItem] = []
        drug_groups: list[DrugGroupItem] = []
        seen_drugs: set[str] = set()
        for ch in cond_chunks:
            for d in ch.get("drugs") or []:
                if not isinstance(d, str) or len(d) < 3:
                    continue
                k = d.lower()[:40]
                if k in seen_drugs:
                    continue
                seen_drugs.add(k)
                drugs.append(
                    DrugTreatmentItem(
                        drug_name=d.strip(),
                        source_ref=_chunk_ref(protocol_id, ch, d),
                    ),
                )
            for m in _DRUG_DOSE_RE.finditer(ch.get("text") or ""):
                k = m.group(1).lower()
                if k in seen_drugs:
                    continue
                seen_drugs.add(k)
                drugs.append(
                    DrugTreatmentItem(
                        drug_name=m.group(1).strip(),
                        dose_text=m.group(2).strip(),
                        frequency_text=(m.group(3) or "").strip() or None,
                        source_ref=_chunk_ref(protocol_id, ch, m.group(0)),
                    ),
                )

        non_drug: list[NonDrugTreatmentItem] = []
        if re.search(r"диет|режим|физическ.*актив|отказ от курен", blob, re.I):
            sample = next((ch for ch in cond_chunks if re.search(r"диет|режим", ch.get("text") or "", re.I)), cond_chunks[0] if cond_chunks else None)
            if sample:
                non_drug.append(
                    NonDrugTreatmentItem(
                        text="немедикаментозные рекомендации (диета, режим)",
                        source_ref=_chunk_ref(protocol_id, sample, "диета, режим"),
                    ),
                )

        red_flags: list[RedFlagItem] = []
        for ch in cond_chunks:
            txt = ch.get("text") or ""
            for rx, rtype, sev in _RED_FLAG_RE:
                m = rx.search(txt)
                if m:
                    key = m.group(0).lower()[:40]
                    if not any(r.text.lower()[:30] == key[:30] for r in red_flags):
                        red_flags.append(
                            RedFlagItem(
                                text=m.group(0).strip(),
                                red_flag_type=rtype,  # type: ignore[arg-type]
                                severity=sev,  # type: ignore[arg-type]
                                expected_actions=["дообследование", "маршрутизация"],
                                cap_if_unhandled=45 if sev == "critical" else None,
                                source_ref=_chunk_ref(protocol_id, ch, m.group(0)),
                            ),
                        )

        follow_up: list[FollowUpRequirement] = []
        for ch in cond_chunks:
            txt = ch.get("text") or ""
            for m in _FOLLOW_UP_RE.finditer(txt):
                phrase = m.group(0).strip()
                if len(phrase) < 8:
                    continue
                if any(f.text[:20] == phrase[:20] for f in follow_up):
                    continue
                follow_up.append(
                    FollowUpRequirement(
                        text=phrase[:200],
                        source_ref=_chunk_ref(protocol_id, ch, phrase),
                    ),
                )
                if len(follow_up) >= 5:
                    break

        clinical = CriteriaBlock(
            required=[
                CriterionItem(
                    text=f"отражение нозологии {name} (МКБ {icd})",
                    operator="present",
                    evidence_targets=["diagnosis", "complaints"],
                    source_ref=_chunk_ref(protocol_id, cond_chunks[0], name) if cond_chunks else SummarySourceRef(
                        protocol_id=protocol_id, page_start=1, section_title="Диагноз", quote=name,
                    ),
                ),
            ],
        )

        conditions.append(
            ConditionSummary(
                condition_id=cid,
                name=name[:200],
                icd10_codes=[icd],
                required_exams=list(exams.values())[:12],
                conditional_exams=[],
                treatment=TreatmentBlock(
                    non_drug=non_drug[:6],
                    drug_groups=drug_groups[:6],
                    drugs=drugs[:12],
                ) if (drugs or non_drug or drug_groups) else None,
                follow_up=follow_up[:6],
                red_flags=red_flags[:8],
                clinical_criteria=clinical,
            ),
        )
    return conditions or [
        ConditionSummary(
            condition_id=_slug(pdf_title),
            name=pdf_title[:200],
            icd10_codes=[],
            clinical_criteria=CriteriaBlock(
                required=[
                    CriterionItem(
                        text=f"соответствие протоколу «{pdf_title[:80]}»",
                        operator="present",
                        evidence_targets=["diagnosis"],
                        source_ref=SummarySourceRef(
                            protocol_id=protocol_id, page_start=1, section_title="Протокол", quote=pdf_title[:200],
                        ),
                    ),
                ],
            ),
        ),
    ]


def build_draft_from_pdf_group(
    source_path: str,
    cards: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> ProtocolSummary:
    """Один ProtocolSummary на PDF (несколько conditions внутри)."""
    protocol_id = _protocol_id_from_path(source_path)
    pdf_title = _title_from_path(source_path)
    first = cards[0]
    approval = _merge_approval(cards)
    rubric_slug = str(first.get("specialty_slug") or "")
    rubric_name = str(first.get("specialty_ru") or rubric_slug or "Каталог")

    year = None
    if approval.get("date"):
        try:
            year = int(str(approval["date"])[:4])
        except ValueError:
            pass

    conditions = _conditions_from_cards_and_chunks(protocol_id, cards, chunks, pdf_title)

    return ProtocolSummary(
        protocol_id=protocol_id,
        extraction_status="auto_extracted",
        review_status="not_reviewed",
        source=ProtocolSource(
            title=pdf_title,
            url=first.get("source_url"),
            local_path=source_path,
            document_sha256=first.get("sha256"),
            approval_date=approval.get("date"),
            approval_number=str(approval.get("number") or "") or None,
            valid_from=approval.get("valid_from"),
            document_year=year,
        ),
        rubric=ProtocolRubric(name=rubric_name, slug=rubric_slug or None, specialty_slugs=[rubric_slug] if rubric_slug else []),
        applicability=ProtocolApplicability(population=_population_from_cards(cards)),  # type: ignore[arg-type]
        conditions=conditions,
        extraction_metadata=ExtractionMetadata(
            extracted_at=datetime.now(timezone.utc).isoformat(),
            extractor="builder.heuristic_v2",
            extractor_version="2.0",
            source_document_sha256=first.get("sha256"),
            notes=[f"aggregated from {len(cards)} protocol_cards, {len(chunks)} chunks"],
        ),
    )


def _load_cards_grouped(
    *,
    rubric: str | None = None,
    limit_pdfs: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if not REGISTRY.is_file():
        return {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with REGISTRY.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                card = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rubric:
                slug = str(card.get("specialty_slug") or card.get("category_slug") or "")
                if rubric not in slug:
                    continue
            path = str(card.get("source_path") or "")
            if path:
                groups[path].append(card)
    if limit_pdfs:
        keys = sorted(groups.keys())[:limit_pdfs]
        return {k: groups[k] for k in keys}
    return dict(groups)


def _load_chunks_by_path() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if not CHUNKS.is_file():
        return out
    with CHUNKS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ch = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = str(ch.get("source_path") or ch.get("local_path") or "")
            if p:
                out.setdefault(p, []).append(ch)
    return out


def _data_root() -> Path:
    root = Path(_cfg_mod.protocol_summary_config.data_root)
    if not root.is_absolute():
        root = ROOT / root
    return root


def publish_summaries(summaries: list[ProtocolSummary]) -> dict[str, int]:
    """Копирует валидные draft в yaml/ и json/."""
    root = _data_root()
    yaml_dir = root / "yaml"
    json_dir = root / "json"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    stats = {"yaml": 0, "json": 0, "skipped": 0}
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None  # type: ignore
    for s in summaries:
        vr = s.validation or validate_protocol_summary(s)
        if vr.status == "invalid":
            stats["skipped"] += 1
            continue
        export_summary_json(s, json_dir)
        stats["json"] += 1
        if yaml:
            (yaml_dir / f"{s.protocol_id}.yaml").write_text(
                yaml.safe_dump(s.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            stats["yaml"] += 1
    return stats


def build_protocol_summaries(
    *,
    limit: int | None = None,
    rubric: str | None = None,
    write_yaml: bool = True,
    validate: bool = True,
    publish: bool = True,
    export_rag: bool = True,
) -> list[ProtocolSummary]:
    """Batch: один summary на PDF → drafts/, publish → yaml/json/."""
    groups = _load_cards_grouped(rubric=rubric, limit_pdfs=limit)
    chunks_map = _load_chunks_by_path()
    root = _data_root()
    drafts_dir = root / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[ProtocolSummary] = []
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None  # type: ignore

    for path in sorted(groups.keys()):
        cards = groups[path]
        chunks = chunks_map.get(path, [])
        summary = build_draft_from_pdf_group(path, cards, chunks)
        if validate:
            result = validate_protocol_summary(summary)
            summary.validation = result
            write_validation_report(summary, result)
        summaries.append(summary)
        if write_yaml and yaml:
            (drafts_dir / f"{summary.protocol_id}.yaml").write_text(
                yaml.safe_dump(summary.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

    if publish and summaries:
        publish_summaries(summaries)
    if export_rag and summaries:
        write_summary_rag_jsonl(summaries)
    if summaries:
        write_summary_quality_report()
    return summaries
