"""Корпусный отчёт качества Protocol Summary."""
from __future__ import annotations

from pathlib import Path

from .config import protocol_summary_config
from .loader import load_protocol_summaries, load_summary_rules
from .summary_to_rules import summary_to_protocol_rules
from .validator import validate_protocol_summary

ROOT = Path(__file__).resolve().parents[2]


def build_summary_quality_report() -> str:
    summaries = load_protocol_summaries(usable_only=False)
    valid = 0
    approved = 0
    conditions = 0
    icd_set: set[str] = set()
    exams = 0
    treatments = 0
    red_flags = 0
    rules = 0
    rules_with_quote = 0

    for s in summaries:
        vr = validate_protocol_summary(s)
        if vr.status in ("valid", "valid_with_warnings"):
            valid += 1
        if s.review_status == "approved":
            approved += 1
        conditions += len(s.conditions)
        for c in s.conditions:
            icd_set.update(c.icd10_codes)
            exams += len(c.required_exams) + len(c.conditional_exams)
            if c.treatment:
                treatments += len(c.treatment.drugs) + len(c.treatment.drug_groups) + len(c.treatment.non_drug)
            red_flags += len(c.red_flags)
        for pr in summary_to_protocol_rules(s):
            rules += 1
            if pr.source.quote:
                rules_with_quote += 1

    loaded_rules = len(load_summary_rules(usable_only=False))
    lines = [
        "# Protocol Summary Quality Report",
        "",
        f"- protocols with summary files: **{len(summaries)}**",
        f"- valid (incl. warnings): **{valid}**",
        f"- approved: **{approved}**",
        f"- conditions extracted: **{conditions}**",
        f"- unique ICD-10 codes: **{len(icd_set)}**",
        f"- exam requirements: **{exams}**",
        f"- treatment items: **{treatments}**",
        f"- red flags: **{red_flags}**",
        f"- generated rules: **{max(rules, loaded_rules)}**",
        f"- rules with quote: **{rules_with_quote}**",
        "",
    ]
    return "\n".join(lines)


def write_summary_quality_report(out_path: Path | None = None) -> Path:
    root = Path(protocol_summary_config.data_root)
    if not root.is_absolute():
        root = ROOT / root
    out_path = out_path or (root / "summary_quality_report.md")
    out_path.write_text(build_summary_quality_report(), encoding="utf-8")
    return out_path
