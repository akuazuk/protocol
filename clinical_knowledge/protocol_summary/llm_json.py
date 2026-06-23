"""Parse JSON from LLM responses."""
from __future__ import annotations

import json
import re
from typing import Any


def parse_json_loose(text: str) -> dict[str, Any] | list[Any] | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return data
    except json.JSONDecodeError:
        m = re.search(r"[\[{][\s\S]*[\]}]", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
