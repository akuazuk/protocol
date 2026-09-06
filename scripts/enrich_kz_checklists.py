#!/usr/bin/env python3
"""Э1: детерминированное обогащение протоколов - сборка kz_checklist из полей карточки.

Работает напрямую на JSON карточках (data/protocol_summaries/json/*.json), без pydantic
и без LLM. Каждый пункт checklist выводится из уже привязанных к цитате полей протокола
(source-anchored по построению). Пропускаем стоматологию (отдельный трек).

  python3 scripts/enrich_kz_checklists.py            # применить и записать
  python3 scripts/enrich_kz_checklists.py --dry-run  # только отчёт, без записи

См. docs/plans/2026-07-22-kz-scoring-methodology-v1.md §11.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_DIR = ROOT / "data" / "protocol_summaries" / "json"
REPORT = ROOT / "data" / "ml" / "reports" / "protocol_enrichment_kz_checklist.md"

# Стоматология - отдельный трек (исключена из охвата Э1).
EXCLUDE_RUBRIC_PREFIXES = ("stomatologiya",)


def _load_builder():
    path = ROOT / "clinical_knowledge" / "protocol_summary" / "kz_checklist_builder.py"
    spec = importlib.util.spec_from_file_location("kz_checklist_builder", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _excluded(summary: dict) -> bool:
    pid = str(summary.get("protocol_id") or "").lower()
    slug = str(((summary.get("rubric") or {}) or {}).get("slug") or "").lower()
    for pref in EXCLUDE_RUBRIC_PREFIXES:
        if pid.startswith(pref) or slug.startswith(pref):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="перезаписать уже существующий kz_checklist")
    args = ap.parse_args()

    builder = _load_builder()
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    files = sorted(JSON_DIR.glob("*.json"))
    stats = Counter()
    cond_total = cond_scope = cond_filled_now = cond_already = cond_no_source = 0
    bucket_items = Counter()
    changed_files = 0

    for fp in files:
        try:
            summary = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stats["parse_error"] += 1
            continue
        if _excluded(summary):
            stats["excluded_protocols"] += 1
            continue
        stats["protocols_in_scope"] += 1
        file_changed = False
        for cond in summary.get("conditions") or []:
            if not isinstance(cond, dict):
                continue
            cond_total += 1
            cond_scope += 1
            existing = cond.get("kz_checklist")
            if builder.checklist_is_nonempty(existing) and not args.overwrite:
                cond_already += 1
                continue
            if not builder.condition_has_source_fields(cond):
                cond_no_source += 1
                continue
            checklist = builder.build_kz_checklist(cond)
            if not builder.checklist_is_nonempty(checklist):
                cond_no_source += 1
                continue
            cond["kz_checklist"] = checklist
            for b in ("must_have", "should_have", "conditional", "warnings"):
                bucket_items[b] += len(checklist.get(b) or [])
            cond_filled_now += 1
            file_changed = True

        if file_changed:
            note = f"kz_checklist=auto_derived_v1@{now}"
            meta = summary.setdefault("extraction_metadata", {})
            notes = meta.setdefault("notes", [])
            if note not in notes:
                notes.append(note)
            if not args.dry_run:
                fp.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            changed_files += 1

    coverage = 100.0 * (cond_already + cond_filled_now) / max(cond_scope, 1)
    lines = [
        "# Обогащение протоколов: kz_checklist (Э1, детерминированно)",
        "",
        f"Дата: {now}  |  режим: {'DRY-RUN' if args.dry_run else 'APPLY'}",
        "",
        (f"- Протоколов в охвате: **{stats['protocols_in_scope']}** "
        f"(исключено стоматологии: {stats['excluded_protocols']})"),
        f"- Файлов изменено: **{changed_files}**",
        f"- Conditions в охвате: **{cond_scope}**",
        f"- Заполнено сейчас: **{cond_filled_now}**",
        f"- Уже было заполнено: {cond_already}",
        f"- Нет исходных полей (оставлены пустыми): {cond_no_source}",
        f"- **Покрытие kz_checklist: {coverage:.1f}%**",
        "",
        "## Пункты по корзинам (сумма по заполненным)",
        f"- must_have: {bucket_items['must_have']}",
        f"- should_have: {bucket_items['should_have']}",
        f"- conditional: {bucket_items['conditional']}",
        f"- warnings: {bucket_items['warnings']}",
        "",
        "Каждый пункт выведен из полей протокола с source_ref (quote) - без LLM, без выдумки.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n(report -> {REPORT})" if not args.dry_run else "\n(dry-run: report not written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
