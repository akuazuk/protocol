import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "scripts" / "ops"


def test_deprecated_promote_commands_fail_closed() -> None:
    for script in (
        OPS / "render_promote_main.sh",
        ROOT / "scripts" / "deploy_promote_main_after_push.sh",
    ):
        result = subprocess.run(
            ["bash", str(script)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "permanently disabled" in result.stderr


def test_release_scripts_require_exact_origin_main_commit() -> None:
    release = (OPS / "render_release_main.sh").read_text(encoding="utf-8")
    apply_deploy = (OPS / "render_apply_deploy.sh").read_text(encoding="utf-8")
    render_deploy = (OPS / "render_deploy.sh").read_text(encoding="utf-8")
    deploy_guard = (ROOT / "scripts" / "git_deploy_guard.sh").read_text(encoding="utf-8")
    gce_deploy = (ROOT / "deploy" / "gcp-app" / "deploy_to_gce.sh").read_text(
        encoding="utf-8"
    )

    assert "--commit=MERGE_SHA" in release
    assert 'if [[ "$requested_sha" != "$main_sha" ]]' in release
    assert "Deploy blocked. Merge the PR first" in release
    assert "local HEAD is never an implicit release source" in apply_deploy
    assert "verify_main_commit" in render_deploy
    assert 'ALLOWED_BRANCHES_DEFAULT="main"' in deploy_guard
    assert "local $branch is not exact" in deploy_guard
    assert 'COMMIT_SHA="$(git rev-parse HEAD)"' not in apply_deploy
    assert 'COMMIT_SHA="$(git rev-parse HEAD)"' not in render_deploy
    assert 'MAIN_SHA="$(git rev-parse origin/main)"' in gce_deploy
    assert 'export GIT_COMMIT_SHA="$RELEASE_SHA"' in gce_deploy
    assert 'vals["GIT_COMMIT_SHA"] = os.environ["GIT_COMMIT_SHA"]' in gce_deploy
    assert "SSH_USER=\\$(whoami)" in gce_deploy
    assert "\\$SSH_USER:\\$SSH_USER" in gce_deploy
    assert "GCE_OPS_USER=" in gce_deploy
    assert "assemble_web_env_from_sm.sh" in gce_deploy
    assert "--init-shadow-state-only" in gce_deploy
    assert "--check-primary" in gce_deploy


def test_release_scripts_have_valid_bash_syntax() -> None:
    for script in (
        OPS / "render_release_main.sh",
        OPS / "render_apply_deploy.sh",
        OPS / "render_deploy.sh",
        OPS / "render_promote_main.sh",
        ROOT / "scripts" / "deploy_promote_main_after_push.sh",
        ROOT / "deploy" / "gcp-app" / "deploy_to_gce.sh",
    ):
        result = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_version_endpoint_exposes_render_git_commit() -> None:
    server = (ROOT / "rag_server.py").read_text(encoding="utf-8")
    assert '"git_commit": (' in server
    assert 'os.environ.get("RENDER_GIT_COMMIT")' in server


def test_production_workflow_serializes_exact_main_sha() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "render-production-deploy.yml"
    ).read_text(encoding="utf-8")
    assert "group: production-render" in workflow
    assert "cancel-in-progress: false" in workflow
    assert 'test "$(git rev-parse origin/main)" = "$GITHUB_SHA"' in workflow
    assert '--commit="$GITHUB_SHA"' in workflow
    assert "scripts/ops/render_release_main.sh" in workflow
