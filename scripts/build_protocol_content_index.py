#!/usr/bin/env python3
"""Собрать data/catalog/protocol_content_index.json из Protocol Summary."""
from __future__ import annotations

from clinical_knowledge.protocol_content_index import (
    clear_content_index_cache,
    write_content_index,
    _load_packaged_index,
)


def main() -> int:
    dest = write_content_index()
    clear_content_index_cache()
    print(f"wrote {dest} keys={len(_load_packaged_index())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
