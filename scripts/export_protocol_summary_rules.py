#!/usr/bin/env python3
"""Export ProtocolRule list from summary cards to JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.protocol_summary.loader import load_protocol_summaries  # noqa: E402
from clinical_knowledge.protocol_summary.summary_to_rules import (  # noqa: E402
    protocol_rule_to_legacy_dict,
    summary_to_protocol_rules,
)


def main() -> int:
    out = ROOT / "data" / "protocol_summaries" / "exported_rules.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rules = []
    for s in load_protocol_summaries(usable_only=False):
        for pr in summary_to_protocol_rules(s):
            rules.append(protocol_rule_to_legacy_dict(pr))
    out.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(rules)} rules → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
