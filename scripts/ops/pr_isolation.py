#!/usr/bin/env python3
"""Изоляция параллельных PR: BUILD_VERSION-only merge и пересечение файлов."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

BUILD_VERSION_LINE = re.compile(r'^BUILD_VERSION\s*=\s*"([^"]+)"\s*$', re.M)
STAMP_LINE = 'BUILD_VERSION = "__STAMP__"'
COMMENT_MARK = "<!-- protocol-pr-overlap -->"
SOFT_PATHS = frozenset({"rag_server.py"})


class UnresolvableBuildVersion(ValueError):
    """rag_server.py отличается не только строкой BUILD_VERSION."""


def stamp_neutral(text: str) -> str:
    return BUILD_VERSION_LINE.sub(STAMP_LINE, text, count=1)


def extract_version(text: str) -> str:
    match = BUILD_VERSION_LINE.search(text)
    return match.group(1) if match else ""


def extract_slug(text: str) -> str:
    version = extract_version(text)
    parts = version.split("Z-", 1)
    if len(parts) == 2 and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", parts[1]):
        return parts[1]
    return ""


def resolve_rag_server(base: str, ours: str, theirs: str) -> tuple[str, str]:
    """Трёхсторонний разбор rag_server.py, если спор только о BUILD_VERSION.

    ours = текущий HEAD во время rebase (обычно origin/main),
    theirs = коммит, который накладываем.
    Возвращает (текст до нового stamp, slug для bump).
    """
    neutral_base = stamp_neutral(base)
    neutral_ours = stamp_neutral(ours)
    neutral_theirs = stamp_neutral(theirs)
    slug = extract_slug(theirs) or extract_slug(ours) or extract_slug(base)
    if neutral_ours == neutral_theirs:
        return theirs, slug
    if neutral_theirs == neutral_base:
        return ours, slug
    if neutral_ours == neutral_base:
        return theirs, slug
    raise UnresolvableBuildVersion(
        "rag_server.py differs beyond BUILD_VERSION; resolve manually"
    )


def classify_overlap(
    our_files: list[str],
    other_files: list[str],
    *,
    our_rag_only_version: bool = False,
    other_rag_only_version: bool = False,
) -> dict[str, list[str]]:
    common = sorted(set(our_files) & set(other_files))
    soft: list[str] = []
    hard: list[str] = []
    for path in common:
        if (
            path in SOFT_PATHS
            and our_rag_only_version
            and other_rag_only_version
        ):
            soft.append(path)
        elif path in SOFT_PATHS and (our_rag_only_version or other_rag_only_version):
            soft.append(path)
        else:
            hard.append(path)
    return {"hard": hard, "soft": soft, "all": common}


def is_build_version_only_diff(unified_diff: str) -> bool:
    """True, если в unified diff rag_server.py меняется только BUILD_VERSION."""
    changed: list[str] = []
    for raw in unified_diff.splitlines():
        if raw.startswith("+++") or raw.startswith("---") or raw.startswith("@@"):
            continue
        if raw.startswith("+") or raw.startswith("-"):
            changed.append(raw[1:])
    if not changed:
        return True
    return all(BUILD_VERSION_LINE.match(line.strip()) for line in changed if line.strip())


def format_overlap_comment(
    *,
    subject_pr: int,
    peers: list[dict[str, Any]],
    merged: bool = False,
) -> str:
    lines = [
        COMMENT_MARK,
        "",
        "## Параллельные PR: пересечение файлов",
        "",
    ]
    if not peers:
        lines.append("Пересечений с другими открытыми PR нет.")
        return "\n".join(lines) + "\n"
    if merged:
        lines.append(
            f"#{subject_pr} уже в `main`. Если ниже есть **жёсткое** пересечение - "
            "дождитесь и сделайте `scripts/ops/rebase_task_onto_main.sh`. "
            "Только `BUILD_VERSION` скрипт снимет сам."
        )
    else:
        lines.append(
            "Другая вкладка/агент трогает те же пути. Не мержить вслепую. "
            "После чужого merge: `scripts/ops/rebase_task_onto_main.sh`."
        )
    lines.append("")
    for peer in peers:
        hard = ", ".join(peer.get("hard") or []) or "нет"
        soft = ", ".join(peer.get("soft") or []) or "нет"
        title = peer.get("title") or ""
        url = peer.get("url") or f"#{peer.get('number')}"
        lines.append(
            f"- [#{peer.get('number')} {title}]({url}): жёсткое: {hard}; "
            f"только версия: {soft}"
        )
    lines.append("")
    return "\n".join(lines)


def _gh_request(
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "protocol-pr-isolation",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8")) if raw else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub {method} {url} -> {exc.code}: {detail}") from exc


def _paginate(url: str, token: str) -> list[Any]:
    out: list[Any] = []
    next_url: str | None = url
    while next_url:
        req = urllib.request.Request(
            next_url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "protocol-pr-isolation",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            chunk = json.loads(resp.read().decode("utf-8"))
            if isinstance(chunk, list):
                out.extend(chunk)
            link = resp.headers.get("Link", "")
        next_url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                next_url = part[part.find("<") + 1 : part.find(">")]
    return out


def _pr_file_rows(api: str, repo: str, number: int, token: str) -> list[dict[str, Any]]:
    return _paginate(f"{api}/repos/{repo}/pulls/{number}/files?per_page=100", token)


def _rag_only_version(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if row.get("filename") != "rag_server.py":
            continue
        return is_build_version_only_diff(str(row.get("patch") or ""))
    return False


def _open_pulls(api: str, repo: str, token: str) -> list[dict[str, Any]]:
    return _paginate(f"{api}/repos/{repo}/pulls?state=open&per_page=100", token)


def _upsert_comment(api: str, repo: str, number: int, token: str, body: str) -> None:
    comments = _paginate(
        f"{api}/repos/{repo}/issues/{number}/comments?per_page=100", token
    )
    existing = next(
        (c for c in comments if COMMENT_MARK in str(c.get("body") or "")),
        None,
    )
    if existing:
        _gh_request("PATCH", str(existing["url"]), token, {"body": body})
        return
    _gh_request(
        "POST",
        f"{api}/repos/{repo}/issues/{number}/comments",
        token,
        {"body": body},
    )


def notify_from_github_event(event: dict[str, Any], token: str) -> dict[str, Any]:
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    repo = os.environ.get("GITHUB_REPOSITORY") or ""
    pr = event.get("pull_request") or {}
    number = int(pr.get("number") or 0)
    if not repo or not number or not token:
        return {"ok": False, "reason": "missing_repo_pr_or_token"}
    action = str(event.get("action") or "")
    merged = bool(pr.get("merged"))
    if action == "closed" and not merged:
        return {"ok": True, "skipped": "closed_unmerged"}

    our_rows = _pr_file_rows(api, repo, number, token)
    our_files = [str(row.get("filename") or "") for row in our_rows if row.get("filename")]
    our_rag_only = _rag_only_version(our_rows)
    peers_out: list[dict[str, Any]] = []
    for other in _open_pulls(api, repo, token):
        other_n = int(other.get("number") or 0)
        if other_n == number:
            continue
        other_rows = _pr_file_rows(api, repo, other_n, token)
        other_files = [str(row.get("filename") or "") for row in other_rows if row.get("filename")]
        kind = classify_overlap(
            our_files,
            other_files,
            our_rag_only_version=our_rag_only,
            other_rag_only_version=_rag_only_version(other_rows),
        )
        if not kind["all"]:
            continue
        peers_out.append(
            {
                "number": other_n,
                "title": other.get("title") or "",
                "url": other.get("html_url") or "",
                "hard": kind["hard"],
                "soft": kind["soft"],
            }
        )

    body = format_overlap_comment(subject_pr=number, peers=peers_out, merged=merged)
    targets = [p["number"] for p in peers_out] if merged else [number]
    if merged:
        for target in targets:
            peer = next(p for p in peers_out if p["number"] == target)
            _upsert_comment(
                api,
                repo,
                target,
                token,
                format_overlap_comment(
                    subject_pr=number,
                    peers=[peer],
                    merged=True,
                ),
            )
    elif peers_out:
        _upsert_comment(api, repo, number, token, body)
    else:
        comments = _paginate(
            f"{api}/repos/{repo}/issues/{number}/comments?per_page=100", token
        )
        if any(COMMENT_MARK in str(c.get("body") or "") for c in comments):
            _upsert_comment(api, repo, number, token, body)
    return {"ok": True, "targets": targets, "peers": len(peers_out)}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parallel PR isolation: BUILD_VERSION rebase and overlap comments"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    help_p = sub.add_parser("help", aliases=["--help-text"])
    help_p.set_defaults(cmd="help")

    res = sub.add_parser("resolve-rag-server", help="Auto-resolve BUILD_VERSION-only 3-way")
    res.add_argument("--base", type=Path, required=True)
    res.add_argument("--ours", type=Path, required=True)
    res.add_argument("--theirs", type=Path, required=True)
    res.add_argument("--out", type=Path, required=True)
    res.add_argument("--slug-out", type=Path)

    cls = sub.add_parser("classify", help="Classify file overlap from JSON on stdin")
    cls.add_argument("--ours", help="Comma-separated our files")

    sub.add_parser("github-notify", help="Comment overlapping PRs from GITHUB_EVENT_PATH")

    args = parser.parse_args(argv)
    if args.cmd == "help":
        parser.print_help()
        return 0
    if args.cmd == "resolve-rag-server":
        try:
            text, slug = resolve_rag_server(
                _read(args.base), _read(args.ours), _read(args.theirs)
            )
        except UnresolvableBuildVersion as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        args.out.write_text(text, encoding="utf-8")
        if args.slug_out:
            args.slug_out.write_text(slug + "\n", encoding="utf-8")
        print(slug)
        return 0
    if args.cmd == "classify":
        payload = json.load(sys.stdin)
        result = classify_overlap(
            [p for p in (args.ours or "").split(",") if p]
            or list(payload.get("ours") or []),
            list(payload.get("theirs") or []),
            our_rag_only_version=bool(payload.get("our_rag_only_version")),
            other_rag_only_version=bool(payload.get("other_rag_only_version")),
        )
        json.dump(result, sys.stdout, ensure_ascii=False)
        print()
        return 1 if result["hard"] else 0
    if args.cmd == "github-notify":
        event_path = Path(os.environ.get("GITHUB_EVENT_PATH") or "")
        token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
        if not event_path.is_file():
            print("skip: no GITHUB_EVENT_PATH", file=sys.stderr)
            return 0
        try:
            event = json.loads(event_path.read_text(encoding="utf-8"))
            summary = notify_from_github_event(event, token)
            json.dump(summary, sys.stdout, ensure_ascii=False)
            print()
        except Exception as exc:  # noqa: BLE001 - notify must not fail CI
            print(f"overlap notify skipped: {exc}", file=sys.stderr)
            return 0
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
