"""Unit tests: rceth parse + status (без сети)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinical_knowledge.rceth_sync.parse import (
    merge_manifest_rows,
    parse_detail_card,
    parse_result_counts,
    parse_search_results,
)
from clinical_knowledge.rceth_sync.status import (
    public_rceth_sync_payload,
    read_status,
    resolve_live_status,
    write_status,
    write_sync_summary,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "rceth"


def test_parse_search_results_sample():
    html = (FIX / "search_results_sample.html").read_text(encoding="utf-8")
    rec, pages = parse_result_counts(html)
    assert rec == 3 and pages == 1
    rows = parse_search_results(html)
    by_id = {r["reg_id"]: r for r in rows}
    assert "21_04_3138" in by_id
    assert by_id["21_04_3138"]["has_s_pdf"] is True
    assert by_id["21_04_3138"]["url_s"].endswith("_s.pdf")
    assert by_id["21_04_3138"]["trade_name_ru"] == "ФЕНИБУТ"
    assert by_id["11349_24"]["has_s_pdf"] is True
    assert by_id["old_01"]["status"] == "expired"
    active = [r for r in rows if r["status"] == "active"]
    assert len(active) == 2


def test_oxlp_section_split_fixtures():
    from clinical_knowledge.rceth_sync.label_parse import build_label_record, split_oxlp_sections

    ibu = (FIX / "oxlp_ibuprofen_sample.txt").read_text(encoding="utf-8")
    split = split_oxlp_sections(ibu)
    assert split["parse"]["ok"] is True
    assert split["parse"]["needs_human"] is False
    assert any("боли" in p.lower() for p in split["sections"]["indications_4_1"])
    assert split["sections"]["contraindications_4_3"]
    assert split["sections"]["posology_4_2"]
    assert split["sections"]["interactions_4_5"]

    phen = (FIX / "oxlp_phenibut_sample.txt").read_text(encoding="utf-8")
    label = build_label_record(
        reg_id="21_04_3138",
        text=phen,
        meta={"trade_name_ru": "ФЕНИБУТ", "inn": "Phenibut", "url_s": "/NDfiles/instr/21_04_3138_s.pdf"},
    )
    assert label["parse"]["ok"] is True
    assert label["pdf_s"]["url"].endswith("_s.pdf")
    assert "нервная" in " ".join(label["sections"]["indications_4_1"]).lower()


def test_oxlp_needs_human_without_core_sections():
    from clinical_knowledge.rceth_sync.label_parse import split_oxlp_sections

    split = split_oxlp_sections("Нет нумерованных разделов ОХЛП в этом тексте.")
    assert split["parse"]["ok"] is False
    assert split["parse"]["needs_human"] is True
    assert split["sections"]["indications_4_1"] == []


def test_parse_detail_card_fixture():
    html = (FIX / "detail_21_04_3138.html").read_text(encoding="utf-8")
    card = parse_detail_card(html, "21_04_3138")
    assert card["reg_id"] == "21_04_3138"
    assert card["has_s_pdf"] is True
    assert "N06" in (card.get("atc") or "") or card.get("inn")
    assert card["url_s"].endswith("21_04_3138_s.pdf")


def test_merge_prefers_active():
    rows = [
        {"reg_id": "x", "status": "expired", "url_s": ""},
        {"reg_id": "x", "status": "active", "url_s": "/NDfiles/instr/x_s.pdf"},
    ]
    merged = merge_manifest_rows(rows)
    assert len(merged) == 1
    assert merged[0]["status"] == "active"
    assert merged[0]["url_s"].endswith("_s.pdf")


def test_status_roundtrip(tmp_path: Path):
    write_status(
        phase="crawl",
        status="running",
        done=2,
        total=10,
        message="letter=а",
        current_reg_id="",
        root=tmp_path,
    )
    live = read_status(tmp_path)
    assert live and live["status"] == "running"
    assert live["progress"]["done"] == 2
    assert live["phase"] == "crawl"
    write_status(
        phase="download",
        status="done",
        done=10,
        total=10,
        message="ok",
        root=tmp_path,
    )
    live2 = read_status(tmp_path)
    assert live2["status"] == "done"
    assert "finished_at" in live2
    path = write_sync_summary(
        {"manifest_count": 5, "with_s_pdf": 3, "downloaded": 3, "failed": 0},
        root=tmp_path,
        day="2026-08-14",
    )
    assert path.name == "rceth_sync_2026-08-14.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["manifest_count"] == 5
    pub = public_rceth_sync_payload(data, live2, history=[data])
    assert pub["ok"] is True
    assert pub["latest"]["with_s_pdf"] == 3
    assert pub["running"] is False
    assert pub["history"]


def test_resolve_live_status_marks_dead_pid_interrupted():
    from datetime import datetime, timezone

    live = {
        "status": "running",
        "phase": "parse",
        "progress": {"done": 0, "total": 50},
        "message": "parse 10000_12_16_17_24",
        "updated_at": "2026-08-14T09:12:40Z",
        "pid": 99999999,
    }
    view, running, top = resolve_live_status(
        live,
        now=datetime(2026, 8, 14, 10, 0, 0, tzinfo=timezone.utc),
        stale_sec=180,
    )
    assert running is False
    assert top == "interrupted"
    assert view["stale"] is True
    assert view["stale_reason"] == "process_gone"
    pub = public_rceth_sync_payload(
        {"sync_day": "2026-08-14", "manifest_count": 1, "with_s_pdf": 1, "downloaded": 1},
        live,
        history=[],
    )
    # public payload re-resolves; dead pid → not running
    assert pub["running"] is False
    assert pub["status"] == "interrupted"
    assert pub["live"]["stale"] is True


def test_resolve_live_status_marks_stale_heartbeat():
    from datetime import datetime, timezone

    live = {
        "status": "running",
        "phase": "parse",
        "progress": {"done": 0, "total": 50},
        "message": "parse hung",
        "updated_at": "2026-08-14T09:00:00Z",
        # no pid → heartbeat age decides
    }
    view, running, top = resolve_live_status(
        live,
        now=datetime(2026, 8, 14, 9, 10, 0, tzinfo=timezone.utc),
        stale_sec=180,
    )
    assert running is False
    assert top == "interrupted"
    assert view["stale_reason"] == "heartbeat_stale"


def test_page_pairs_requires_query_string_and_postback():
    from clinical_knowledge.rceth_sync.http_client import page_pairs_from_html

    html = '<input id="QueryStringFind" name="QueryStringFind" type="hidden" value="QQQ" />'
    pairs = page_pairs_from_html(html, 2)
    as_dict = dict(pairs)
    assert as_dict["QueryStringFind"] == "QQQ"
    assert as_dict["IsPostBack"] == "true"
    assert as_dict["PropSubmit"] == "FOpt_PageN"
    assert as_dict["ValueSubmit"] == "2"
    with pytest.raises(ValueError):
        page_pairs_from_html("<html></html>", 2)


def test_refbank_client_retries_timeout(monkeypatch):
    from clinical_knowledge.rceth_sync import http_client as hc

    calls = {"n": 0}

    class Boom:
        def open(self, req, timeout=None):
            calls["n"] += 1
            raise TimeoutError("slow")

    client = hc.RefbankClient(throttle_sec=0, timeout=5, retries=3, insecure_ssl=True)
    client._opener = Boom()  # type: ignore[assignment]
    with pytest.raises(TimeoutError):
        client.request("https://www.rceth.by/Refbank/")
    assert calls["n"] == 3
