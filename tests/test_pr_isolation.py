from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "scripts" / "ops"
sys.path.insert(0, str(OPS))

from pr_isolation import (  # noqa: E402
    COMMENT_MARK,
    UnresolvableBuildVersion,
    classify_overlap,
    extract_slug,
    format_overlap_comment,
    is_build_version_only_diff,
    resolve_rag_server,
    stamp_neutral,
)


def _rag(version: str, extra: str = "") -> str:
    return f'BUILD_VERSION = "{version}"\n{extra}def app():\n    return 1\n'


def test_stamp_neutral_and_slug() -> None:
    text = _rag("2026-09-05-041017Z-mo-meds-labs-dash")
    assert "mo-meds-labs-dash" == extract_slug(text)
    assert stamp_neutral(text) == _rag("__STAMP__")


def test_resolve_keeps_main_when_replay_only_bumped_version() -> None:
    base = _rag("2026-09-04-010000Z-old")
    ours = _rag("2026-09-05-041017Z-mo-meds-labs-dash", extra="FINDING = 1\n")
    theirs = _rag("2026-09-05-041200Z-consult-rag")
    resolved, slug = resolve_rag_server(base, ours, theirs)
    assert "FINDING = 1" in resolved
    assert slug == "consult-rag"


def test_resolve_keeps_feature_when_main_only_bumped_version() -> None:
    base = _rag("2026-09-04-010000Z-old")
    ours = _rag("2026-09-05-041017Z-main")
    theirs = _rag("2026-09-05-041200Z-consult-rag", extra="CONSULT = 1\n")
    resolved, slug = resolve_rag_server(base, ours, theirs)
    assert "CONSULT = 1" in resolved
    assert slug == "consult-rag"


def test_resolve_rejects_real_rag_server_conflict() -> None:
    base = _rag("2026-09-04-010000Z-old")
    ours = _rag("2026-09-05-041017Z-a", extra="A = 1\n")
    theirs = _rag("2026-09-05-041200Z-b", extra="B = 1\n")
    try:
        resolve_rag_server(base, ours, theirs)
    except UnresolvableBuildVersion:
        return
    raise AssertionError("expected UnresolvableBuildVersion")


def test_overlap_soft_for_version_only_rag_server() -> None:
    kind = classify_overlap(
        ["rag_server.py", "clinical_knowledge/consult_retrieval.py"],
        ["rag_server.py", "docs/a.md"],
        our_rag_only_version=True,
        other_rag_only_version=True,
    )
    assert kind["soft"] == ["rag_server.py"]
    assert kind["hard"] == []


def test_overlap_hard_for_shared_feature_file() -> None:
    kind = classify_overlap(
        ["clinical_knowledge/mo_backend.py"],
        ["clinical_knowledge/mo_backend.py"],
    )
    assert kind["hard"] == ["clinical_knowledge/mo_backend.py"]


def test_is_build_version_only_diff() -> None:
    diff = """\
--- a/rag_server.py
+++ b/rag_server.py
@@ -1 +1 @@
-BUILD_VERSION = "2026-09-05-041017Z-old"
+BUILD_VERSION = "2026-09-05-050000Z-new"
"""
    assert is_build_version_only_diff(diff) is True
    assert is_build_version_only_diff(diff + "+FINDING = 1\n") is False


def test_overlap_comment_has_stable_marker() -> None:
    body = format_overlap_comment(
        subject_pr=187,
        peers=[
            {
                "number": 188,
                "title": "consult rag",
                "url": "https://example.test/188",
                "hard": [],
                "soft": ["rag_server.py"],
            }
        ],
        merged=True,
    )
    assert COMMENT_MARK in body
    assert "rebase_task_onto_main.sh" in body
    assert "#188" in body


def test_ci_does_not_cancel_in_progress_runs() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "cancel-in-progress: false" in workflow
    assert "github.event.pull_request.number || github.ref" in workflow
    notify = (
        ROOT / ".github" / "workflows" / "pr-overlap-notify.yml"
    ).read_text(encoding="utf-8")
    assert "cancel-in-progress: false" in notify
    assert "continue-on-error: true" in notify


def test_isolation_scripts_have_help_and_syntax() -> None:
    for script in (
        OPS / "rebase_task_onto_main.sh",
        OPS / "check_pr_file_overlap.sh",
    ):
        parsed = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert parsed.returncode == 0, parsed.stderr
        help_run = subprocess.run(
            ["bash", str(script), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert help_run.returncode == 0, help_run.stderr
    py_help = subprocess.run(
        ["python3", str(OPS / "pr_isolation.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert py_help.returncode == 0
