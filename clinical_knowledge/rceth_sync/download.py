"""Preflight NDfiles + download `_s.pdf` с resume/sha256."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from clinical_knowledge.rceth_sync.crawl import load_manifest
from clinical_knowledge.rceth_sync.http_client import BASE, RefbankClient
from clinical_knowledge.rceth_sync.paths import manifest_path, pdf_dir
from clinical_knowledge.rceth_sync.status import write_status, write_sync_summary

# Известный PDF для smoke (Фенибут специалист).
SMOKE_URL = "/NDfiles/instr/21_04_3138_s.pdf"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight_ndfiles(
    client: RefbankClient | None = None,
    *,
    urls: list[str] | None = None,
    insecure_ssl: bool = False,
    throttle_sec: float = 0.4,
    root: Path | None = None,
) -> dict[str, Any]:
    """Smoke `_s.pdf` через Range (не полный download). ok если большинство PDF."""
    client = client or RefbankClient(
        throttle_sec=throttle_sec,
        insecure_ssl=insecure_ssl,
        timeout=25.0,
        retries=3,
    )
    # небольшие/известные инструкции; не тянуть 20МБ Avastin целиком
    urls = urls or [SMOKE_URL, "/NDfiles/instr/11349_24_s.pdf", "/NDfiles/instr/19_06_2226_s.pdf"]
    write_status(
        phase="preflight",
        status="running",
        done=0,
        total=len(urls),
        message="ndfiles range smoke",
        root=root,
    )
    probes: list[dict[str, Any]] = []
    ok_n = 0
    for i, url in enumerate(urls, start=1):
        write_status(
            phase="preflight",
            status="running",
            done=i - 1,
            total=len(urls),
            message=url.rsplit("/", 1)[-1],
            current_reg_id=url.rsplit("/", 1)[-1].replace("_s.pdf", ""),
            root=root,
        )
        try:
            code, body, headers = client.get_bytes(
                url,
                range_bytes=(0, 65535),
                max_read=65536,
            )
            err = ""
        except Exception as exc:  # noqa: BLE001
            code, body, headers, err = 0, b"", {}, str(exc)[:180]
        head = body[:16]
        is_pdf = head[:4] == b"%PDF" or (headers.get("content-type") or "").lower().startswith(
            "application/pdf"
        )
        # 206 Partial / 200 OK оба допустимы
        ok = code in {200, 206} and is_pdf and len(body) > 64
        if ok:
            ok_n += 1
        probes.append(
            {
                "url": url,
                "http": code,
                "bytes": len(body),
                "is_pdf": bool(is_pdf),
                "ok": ok,
                "error": err,
            }
        )
        del body
    result = {
        "ok": ok_n >= max(1, (len(urls) + 1) // 2),
        "ok_count": ok_n,
        "total": len(urls),
        "probes": probes,
        "base": BASE,
    }
    write_status(
        phase="preflight",
        status="done" if result["ok"] else "error",
        done=len(urls),
        total=len(urls),
        message=f"ok={ok_n}/{len(urls)}",
        errors=0 if result["ok"] else 1,
        root=root,
        extra={"preflight": {"ok": result["ok"], "ok_count": ok_n, "total": len(urls)}},
    )
    return result


def local_pdf_path(reg_id: str, root: Path | None = None, kind: str = "s") -> Path:
    return pdf_dir(root) / f"{reg_id}_{kind}.pdf"


def download_s_pdfs(
    *,
    root: Path | None = None,
    limit: int | None = None,
    throttle_sec: float = 0.7,
    insecure_ssl: bool = False,
    require_preflight: bool = True,
    client: RefbankClient | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    """Скачать `_s` для строк манифеста с url_s; skip если sha уже есть."""
    client = client or RefbankClient(
        throttle_sec=throttle_sec,
        insecure_ssl=insecure_ssl,
        timeout=45.0,
        retries=retries,
    )
    rows = [r for r in load_manifest(manifest_path(root)) if r.get("url_s") or r.get("has_s_pdf")]
    if limit is not None:
        rows = rows[: max(0, limit)]
    write_status(
        phase="download",
        status="running",
        done=0,
        total=len(rows),
        message="preflight" if require_preflight else "start",
        root=root,
    )
    retries_503 = 0
    errors = 0
    if require_preflight:
        pf = preflight_ndfiles(
            client,
            insecure_ssl=insecure_ssl,
            throttle_sec=throttle_sec,
            root=root,
        )
        if not pf.get("ok"):
            write_status(
                phase="download",
                status="error",
                done=0,
                total=len(rows),
                message="preflight_failed",
                errors=1,
                root=root,
                extra={"preflight": pf},
            )
            return {"ok": False, "preflight": pf, "downloaded": 0, "failed": 0, "skipped": 0}
    else:
        pf = {"ok": True, "skipped": True}

    downloaded = 0
    skipped = 0
    failed = 0
    out_dir = pdf_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(rows, start=1):
        reg_id = str(row.get("reg_id") or "")
        url = str(row.get("url_s") or f"/NDfiles/instr/{reg_id}_s.pdf")
        dest = local_pdf_path(reg_id, root, "s")
        write_status(
            phase="download",
            status="running",
            done=i - 1,
            total=len(rows),
            message="download",
            current_reg_id=reg_id,
            errors=errors,
            retries_503=retries_503,
            root=root,
        )
        if dest.is_file() and dest.stat().st_size > 1000:
            # resume: already present
            row["pdf_s_sha256"] = sha256_file(dest)
            row["pdf_s_bytes"] = dest.stat().st_size
            skipped += 1
            continue
        ok_file = False
        last_err = ""
        for attempt in range(1, retries + 1):
            code, body, headers = client.get_bytes(url)
            if code == 503:
                retries_503 += 1
                time.sleep(min(8, attempt * 2))
                last_err = "503"
                continue
            if code != 200 or body[:4] != b"%PDF":
                last_err = f"http={code} ct={headers.get('content-type')}"
                time.sleep(attempt)
                continue
            dest.write_bytes(body)
            row["pdf_s_sha256"] = hashlib.sha256(body).hexdigest()
            row["pdf_s_bytes"] = len(body)
            downloaded += 1
            ok_file = True
            break
        if not ok_file:
            failed += 1
            errors += 1
            row["download_error"] = last_err[:200]

    # переписать манифест с sha
    from clinical_knowledge.rceth_sync.crawl import write_manifest

    all_rows = load_manifest(manifest_path(root))
    by_id = {str(r.get("reg_id")): r for r in rows}
    for r in all_rows:
        rid = str(r.get("reg_id"))
        if rid in by_id:
            r.update({k: by_id[rid][k] for k in ("pdf_s_sha256", "pdf_s_bytes", "download_error") if k in by_id[rid]})
    write_manifest(all_rows, manifest_path(root))

    summary = {
        "ok": failed == 0,
        "preflight": {k: pf[k] for k in ("ok", "ok_count", "total") if k in pf},
        "manifest_count": len(all_rows),
        "with_s_pdf": sum(1 for r in all_rows if r.get("url_s") or r.get("has_s_pdf")),
        "download_targets": len(rows),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "retries_503": retries_503,
    }
    write_status(
        phase="download",
        status="done" if failed == 0 else "done",
        done=len(rows),
        total=len(rows),
        message=f"downloaded={downloaded} skipped={skipped} failed={failed}",
        errors=errors,
        retries_503=retries_503,
        root=root,
        extra={"summary": summary},
    )
    write_sync_summary(summary, root=root)
    return summary
