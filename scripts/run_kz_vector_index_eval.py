#!/usr/bin/env python3
"""Полный прогон КЗ + проверка vector index на Render (L1 + метрики retrieval/semantic)."""
from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import env_load

    env_load.load_project_env(ROOT)
except ImportError:
    pass

from clinical_knowledge.patient_upload_classifier import is_b2c_lab_filename

DEFAULT_BASE = os.environ.get("RENDER_URL", "https://protocol-bimy.onrender.com")
SUPPORTED = {".pdf", ".txt", ".md", ".docx", ".rtf", ".odt", ".html"}


def _import_batch_module():
    import importlib.util

    path = ROOT / "scripts" / "run_clients_consult_render_batch.py"
    spec = importlib.util.spec_from_file_location("kz_batch", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _discover(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED and p.name.lower() != "readme.md"
    )


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    try:
        import certifi

        ctx.load_verify_locations(certifi.where())
    except ImportError:
        pass
    return ctx


def _get(base: str, path: str, *, timeout: int = 60) -> dict:
    req = urllib.request.Request(f"{base.rstrip('/')}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _semantic_probe(base: str, path: str, query: str) -> dict:
    enc_path = urllib.parse.quote(path, safe="")
    enc_q = urllib.parse.quote(query[:400], safe="")
    try:
        return _get(base, f"/api/protocol-semantic-search?path={enc_path}&q={enc_q}", timeout=90)
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


def _extract_probe_query(text: str) -> str:
    """Короткий клинический запрос из КЗ для semantic probe."""
    blob = (text or "").replace("\r", "\n")
    for pat in (
        r"(?i)жалоб\w*[^\n]{0,120}",
        r"(?i)диагноз[^\n]{0,120}",
        r"(?i)рекомендац\w*[^\n]{0,120}",
        r"(?i)назнач\w*[^\n]{0,120}",
    ):
        m = re.search(pat, blob)
        if m:
            q = re.sub(r"\s+", " ", m.group(0)).strip()
            if len(q) >= 8:
                return q[:200]
    words = re.findall(r"[а-яёa-z]{4,}", blob.lower())[:12]
    return " ".join(words)[:160] if words else "лечение диагностика"


def _classify_file(path: Path) -> str:
    return "b2c_analysis" if is_b2c_lab_filename(path.stem) else "kz"


def run_case_with_retry(
    path: Path,
    *,
    base: str,
    token: str,
    tier: str,
    retries: int = 3,
    pause_sec: float = 8.0,
) -> dict:
    batch = _import_batch_module()
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return batch.run_case(path, base=base, token=token, ai_review="off", tier=tier)
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in (502, 503, 504) and attempt < retries:
                time.sleep(pause_sec * attempt)
                continue
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < retries and "502" in str(exc):
                time.sleep(pause_sec * attempt)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("run_case_with_retry failed")
    return "b2c_analysis" if is_b2c_lab_filename(path.stem) else "kz"


def _build_improvement_plan(summary: dict, reports: list[dict], index: dict) -> list[str]:
    plan: list[str] = []
    if not index.get("loaded") and index.get("enabled"):
        plan.append(
            "P0: vector index enabled но не в RAM - при первом запросе грузится ~1GB; "
            "на Render Standard возможен 502/OOM. Решение: mmap/lazy load или plan Pro."
        )
    elif (index.get("indexed") or 0) < 50000:
        plan.append(f"P0: indexed={index.get('indexed')} - пересобрать индекс (ожидается ~83k).")

    kz = [r for r in reports if r.get("doc_kind") == "kz"]
    no_rag = [r for r in kz if (r.get("rag_chunks_n") or 0) == 0]
    no_ret = [r for r in kz if not r.get("retrieval_top")]
    low = [r for r in kz if r.get("overall_pct") is not None and float(r["overall_pct"]) < 70]
    sem_fail = [r for r in kz if r.get("semantic_ok") is False]
    sem_lex = [r for r in kz if r.get("semantic_mode") == "lexical"]
    analysis_scored = [
        r for r in reports if r.get("doc_kind") == "b2c_analysis" and (r.get("overall_pct") or 0) > 0
    ]

    if no_rag:
        plan.append(
            f"P1: {len(no_rag)} КЗ без RAG-чанков - усилить retrieval/allowlist: "
            + ", ".join(r["case_id"] for r in no_rag[:5])
            + ("…" if len(no_rag) > 5 else "")
        )
    if no_ret:
        plan.append(
            f"P1: {len(no_ret)} КЗ без retrieval_top - проверить vector prefilter и ICD routing."
        )
    if sem_lex and index.get("loaded"):
        plan.append(
            f"P2: protocol semantic в lexical для {len(sem_lex)}/{len(kz)} КЗ - "
            "включить per-path embeddings в lazy store (PROTOCOL_SEMANTIC include_embedding)."
        )
    if sem_fail:
        plan.append(
            f"P2: semantic probe failed для {len(sem_fail)} КЗ - "
            + ", ".join(r["case_id"] for r in sem_fail[:4])
        )
    if low:
        plan.append(
            f"P2: overall<70% у {len(low)} КЗ - ручной разбор: "
            + ", ".join(r["case_id"] for r in low[:6])
        )
    if analysis_scored:
        plan.append(
            f"P1: {len(analysis_scored)} файлов A/a/А/а получили scoring - "
            "усилить gate lab_in_kz в consult-review."
        )
    if not plan:
        plan.append("Индекс и retrieval в норме по автоматическим метрикам; точечный разбор weak cases по REVIEW_QUEUE.")
    return plan


def _write_report_md(out: Path, payload: dict) -> None:
    s = payload["summary"]
    idx = s.get("vector_index") or {}
    lines = [
        "# KZ + Vector Index Eval",
        "",
        f"- **Дата:** {s.get('generated_at')}",
        f"- **Base:** {s.get('base')}",
        f"- **BUILD:** {s.get('server_version')}",
        f"- Tier: **{s.get('tier', 'L1')}**",
        "",
        "## Vector index",
        "",
        "| enabled | loaded | indexed | dim | path |",
        "|---|---|---|---|---|",
        f"| {idx.get('enabled')} | {idx.get('loaded')} | {idx.get('indexed')} | {idx.get('dim')} | `{idx.get('path')}` |",
        "",
        "## Сводка",
        "",
        f"- Всего файлов: **{s.get('total')}** (КЗ: {s.get('kz_count')}, анализы A/a/А/а: {s.get('b2c_analysis_count')})",
        f"- Успешно: **{s.get('ok')}**, ошибок: {s.get('errors')}",
        f"- КЗ avg overall: **{s.get('kz_overall_avg')}%**",
        f"- КЗ overall<70%: **{s.get('kz_overall_lt70')}**",
        f"- КЗ без RAG chunks: **{s.get('kz_no_rag')}**",
        f"- КЗ semantic lexical (не vector): **{s.get('kz_semantic_lexical')}**",
        "",
        "## План улучшений",
        "",
    ]
    for i, item in enumerate(s.get("improvement_plan") or [], 1):
        lines.append(f"{i}. {item}")
    lines.extend(["", "## Weak KZ (overall < 70%)", ""])
    for r in payload.get("weak_kz") or []:
        lines.append(
            f"- `{r.get('case_id')}` overall={r.get('overall_pct')}% "
            f"rag={r.get('rag_chunks_n')} ret={len(r.get('retrieval_top') or [])} "
            f"sem={r.get('semantic_mode')}"
        )
    lines.extend(["", "## B2C анализы (A/a/А/а)", ""])
    for r in payload.get("analysis_reports") or []:
        lines.append(
            f"- `{r.get('case_id')}` mismatch={r.get('upload_mismatch')} "
            f"wrong={r.get('wrong_document_kind')} overall={r.get('overall_pct')}"
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", type=Path, default=ROOT / "clients_consult")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", type=Path, default=ROOT / "ml" / "experiments" / f"kz_index_eval_{time.strftime('%Y-%m-%d')}")
    ap.add_argument("--kz-only", action="store_true", help="Только КЗ (без A/a/А/а)")
    ap.add_argument("--include-analysis", action="store_true", help="Явно включить прогон анализов")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tier", choices=("L1", "L2"), default="L1")
    ap.add_argument("--semantic-probe", action="store_true", default=True)
    ap.add_argument("--warm-index", action="store_true", default=False)
    ap.add_argument("--no-warm-index", action="store_false", dest="warm_index")
    ap.add_argument("--no-semantic-probe", action="store_false", dest="semantic_probe")
    ap.add_argument("--pause-sec", type=float, default=20.0, help="Пауза между КЗ (снижает OOM на Render)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    token = (os.environ.get("METHODIST_TOKEN") or os.environ.get("METHODIST_PIN") or "").strip()
    batch = _import_batch_module()
    load_text = batch._load_text

    print("Index health...", flush=True)
    try:
        health = _get(args.base, "/health/live", timeout=30)
        corpus = _get(args.base, "/api/corpus-stats", timeout=45)
        vector_index = corpus.get("vector_index") or {}
    except Exception as exc:
        print(f"FATAL: cannot reach {args.base}: {exc}", file=sys.stderr)
        return 2

    server_version = health.get("version") or corpus.get("version")
    print(f"  version={server_version} index loaded={vector_index.get('loaded')} indexed={vector_index.get('indexed')}")

    if args.warm_index:
        print("Warming vector index...", flush=True)
        try:
            _get(
                args.base,
                "/api/protocol-semantic-search?path="
                + urllib.parse.quote(
                    "minzdrav_protocols/khirurgiya/КП_Диагностика_лечение_пациентов_взрос_с_доброкач_забол_прямой_кишки_параректальной_и_копчиковой_области_амбул_пост_МЗ_01.04.2022_№22.pdf",
                    safe="",
                )
                + "&q="
                + urllib.parse.quote("какие препараты"),
                timeout=120,
            )
            corpus = _get(args.base, "/api/corpus-stats", timeout=45)
            vector_index = corpus.get("vector_index") or vector_index
            print(
                f"  after warm: loaded={vector_index.get('loaded')} indexed={vector_index.get('indexed')}",
                flush=True,
            )
        except Exception as exc:
            print(f"  warm-index skip: {exc}", flush=True)

    paths = _discover(args.folder.resolve())
    if args.kz_only:
        paths = [p for p in paths if _classify_file(p) == "kz"]
    elif not args.include_analysis:
        # по умолчанию: все файлы, но анализы помечаем отдельно
        pass
    if args.limit:
        paths = paths[: args.limit]

    reports: list[dict] = []
    errors: list[dict] = []

    for i, p in enumerate(paths, 1):
        doc_kind = _classify_file(p)
        print(f"[{i}/{len(paths)}] {p.name} ({doc_kind})...", flush=True)
        try:
            rep = run_case_with_retry(p, base=args.base, token=token, tier=args.tier)
            rep["doc_kind"] = doc_kind
            rep["expected_doc_kind"] = doc_kind

            if args.semantic_probe and doc_kind == "kz":
                text = load_text(p)
                rep["probe_query"] = _extract_probe_query(text)
                top_path = None
                for key in ("matched_protocols_full", "retrieval_top_full", "matched_protocols", "retrieval_top"):
                    vals = rep.get(key) or []
                    if vals:
                        top_path = vals[0]
                        if not str(top_path).startswith("minzdrav_protocols/"):
                            # усечённый хвост - пропускаем semantic probe
                            top_path = None
                            continue
                        break
                if top_path:
                    sem = _semantic_probe(args.base, top_path, rep["probe_query"])
                    rep["semantic_ok"] = bool(sem.get("ok"))
                    rep["semantic_mode"] = sem.get("mode")
                    rep["semantic_match_count"] = sem.get("match_count")
                    rep["semantic_vector_enabled"] = sem.get("vector_enabled")
                    rep["semantic_path"] = top_path
                else:
                    rep["semantic_ok"] = None
                    rep["semantic_mode"] = None
                    rep["semantic_match_count"] = 0

            reports.append(rep)
            print(
                f"  overall={rep.get('overall_pct')}% rag={rep.get('rag_chunks_n')} "
                f"ret={len(rep.get('retrieval_top') or [])} "
                f"sem={rep.get('semantic_mode')}",
                flush=True,
            )
        except Exception as exc:
            errors.append({"file": p.name, "doc_kind": doc_kind, "error": str(exc)[:400]})
            print(f"  ERROR: {exc}", file=sys.stderr)
        if i < len(paths):
            time.sleep(max(0.0, args.pause_sec))

    kz_reports = [r for r in reports if r.get("doc_kind") == "kz"]
    analysis_reports = [r for r in reports if r.get("doc_kind") == "b2c_analysis"]
    weak_kz = sorted(
        [r for r in kz_reports if r.get("overall_pct") is not None and float(r["overall_pct"]) < 70],
        key=lambda x: float(x.get("overall_pct") or 0),
    )

    kz_ovs = [float(r["overall_pct"]) for r in kz_reports if r.get("overall_pct") is not None]
    summary = {
        "base": args.base,
        "folder": str(args.folder),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server_version": server_version,
        "tier": args.tier,
        "vector_index": vector_index,
        "total": len(paths),
        "ok": len(reports),
        "errors": len(errors),
        "kz_count": len(kz_reports),
        "b2c_analysis_count": len(analysis_reports),
        "kz_overall_avg": round(sum(kz_ovs) / len(kz_ovs), 1) if kz_ovs else None,
        "kz_overall_lt70": len(weak_kz),
        "kz_no_rag": sum(1 for r in kz_reports if (r.get("rag_chunks_n") or 0) == 0),
        "kz_no_retrieval": sum(1 for r in kz_reports if not r.get("retrieval_top")),
        "kz_semantic_lexical": sum(1 for r in kz_reports if r.get("semantic_mode") == "lexical"),
        "kz_semantic_semantic": sum(1 for r in kz_reports if r.get("semantic_mode") == "semantic"),
        "analysis_mismatch_ok": sum(1 for r in analysis_reports if r.get("upload_mismatch")),
    }
    summary["improvement_plan"] = _build_improvement_plan(summary, reports, vector_index)

    payload = {
        "summary": summary,
        "reports": reports,
        "errors": errors,
        "weak_kz": weak_kz,
        "analysis_reports": analysis_reports,
    }
    (args.out / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report_md(args.out, payload)

    print(f"\nSaved: {args.out / 'report.json'}")
    print(f"Saved: {args.out / 'REPORT.md'}")
    print(f"KZ avg={summary.get('kz_overall_avg')}% weak={summary.get('kz_overall_lt70')} no_rag={summary.get('kz_no_rag')}")
    return 0 if reports else 1


if __name__ == "__main__":
    raise SystemExit(main())
