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
    assert "/var/data/medical_exams/reports" in gce_deploy
    assert "medical_exams/state /var/data/medical_exams/reports" in gce_deploy
    rollout_runner = (
        ROOT / "scripts" / "run_mo_lab_rollout_metrics.py"
    ).read_text(encoding="utf-8")
    assert "spec_from_file_location" in rollout_runner
    assert "from clinical_knowledge.mo_lab_rollout import" not in rollout_runner


def test_release_scripts_have_valid_bash_syntax() -> None:
    for script in (
        OPS / "render_release_main.sh",
        OPS / "render_apply_deploy.sh",
        OPS / "render_deploy.sh",
        OPS / "render_promote_main.sh",
        ROOT / "scripts" / "deploy_promote_main_after_push.sh",
        ROOT / "deploy" / "gcp-app" / "deploy_to_gce.sh",
        ROOT / "deploy" / "gcp-app" / "assemble_web_env_from_sm.sh",
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
    """Прод-workflow целится в GCE и только в текущий HEAD origin/main.

    Render-workflow удалён: сервис приостановлен и отдаёт 503,
    см. docs/deploy/gce-production-runbook.md.
    """
    workflows = ROOT / ".github" / "workflows"
    assert not (workflows / "render-production-deploy.yml").exists(), (
        "Render больше не прод - workflow не должен возвращаться"
    )

    workflow = (workflows / "gce-production-deploy.yml").read_text(encoding="utf-8")
    assert "group: production-gce" in workflow
    assert "cancel-in-progress: false" in workflow
    # Разворачивать можно только текущий HEAD origin/main.
    assert 'requested" != "$main_sha' in workflow
    assert "deploy/gcp-app/deploy_to_gce.sh" in workflow
    # После деплоя обязательна сверка версии и SHA на живом домене.
    assert "protocol.kravira.by" in workflow
    assert "EXPECTED_VERSION" in workflow


def test_gce_deploy_is_reproducible_and_can_roll_back() -> None:
    """Деплой берёт коммит, версионирует образ и умеет откатываться.

    Регресс (2026-09-05): раньше на прод уезжал `tar` рабочего дерева, образ
    всегда имел один тег `:staging`, а откатываться было не на что.
    """
    deploy = (ROOT / "deploy" / "gcp-app" / "deploy_to_gce.sh").read_text(
        encoding="utf-8"
    )

    # Источник - коммит, а не рабочее дерево.
    assert 'git archive --format=tar "$RELEASE_SHA"' in deploy
    assert "tar czf -" not in deploy, "рабочее дерево больше не источник деплоя"

    # Образ версионируется по SHA - это материал для откатa.
    assert 'IMAGE_SHA_TAG="${IMAGE_REPO}:${RELEASE_SHA:0:12}"' in deploy
    assert "'$IMAGE_SHA_TAG'" in deploy

    # Откат существует и вызывается при неудаче.
    assert "rollback()" in deploy
    assert deploy.count("rollback || true") >= 3

    # Сверка версии берётся из релизного коммита, а не из рабочего дерева.
    assert 'git show "$RELEASE_SHA:rag_server.py"' in deploy
    assert "ACTUAL_SHA" in deploy

    # Контейнер не публикуется наружу.
    assert "-p 127.0.0.1:8000:8000" in deploy
    assert "-p 8000:8000 \\" not in deploy


def test_legacy_render_release_is_blocked_by_default() -> None:
    result = subprocess.run(
        ["bash", str(OPS / "render_release_main.sh"), "--commit=deadbeef"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Render не является продом" in result.stderr

    deploy_result = subprocess.run(
        ["bash", str(OPS / "render_deploy.sh"), "deploy"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert deploy_result.returncode != 0
    assert "прод на GCE" in deploy_result.stderr


def test_ci_workflow_does_not_cancel_parallel_or_previous_runs() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "cancel-in-progress: false" in workflow
    assert "github.event.pull_request.number || github.ref" in workflow
