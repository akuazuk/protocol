"""Тесты гигиены PR.

Проверка блокирующая, поэтому цена ошибки в обе стороны высока: ложное
срабатывание останавливает работу, пропуск возвращает безымянные ветки и PR без
владельца. Поэтому здесь зафиксированы и реальные имена ветвей проекта, и
реальный шаблон описания.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ops"))

from check_pr_hygiene import (  # noqa: E402
    TEMPLATE,
    check_body,
    check_branch_name,
    check_size,
    evaluate,
    template_placeholders,
)

# Имена, которыми проект реально пользовался. Если конвенция сузится, эти
# ветки не должны внезапно стать невалидными.
REAL_BRANCHES = [
    "cursor/production-readiness-agent1-pc1",
    "cursor/mo-lab-from-mis-tests-pc1",
    "cursor/mo-night-speed-alerts-plan-v1-pc1",
    "cursor/rz-quality-article-layout-agent1-pc1",
    "codex/mo-report-reconcile-agent1-pc1",
    "hotfix/mo-source-runtime-release-pc1",
    "release/gce-cutover-pc2",
]


@pytest.mark.parametrize("branch", REAL_BRANCHES)
def test_real_branch_names_pass(branch: str) -> None:
    assert check_branch_name(branch) == []


@pytest.mark.parametrize(
    "branch",
    [
        "main",
        "master",
        "codex/main-sync",
        "cursor/main-sync",
    ],
)
def test_shared_mutable_branches_rejected(branch: str) -> None:
    problems = check_branch_name(branch)
    assert problems, f"{branch} должна отклоняться"
    assert "общая" in problems[0]


@pytest.mark.parametrize(
    "branch",
    [
        "",
        "feature/no-owner",
        "cursor/no-computer",
        "cursor/Uppercase-pc1",
        "cursor/-pc1",
        "cursor/trailing-pc",
        "random-branch",
        "cursor/spaces are bad-pc1",
    ],
)
def test_bad_branch_names_rejected(branch: str) -> None:
    assert check_branch_name(branch), f"{branch} должна отклоняться"


def test_branch_must_name_the_computer() -> None:
    """`-pc<N>` обязателен: по имени ветки видно, чей checkout её держит."""
    assert check_branch_name("cursor/mo-lab-import") != []
    assert check_branch_name("cursor/mo-lab-import-pc1") == []


def test_empty_body_rejected() -> None:
    problems = check_body("", set())
    assert problems
    assert "пустое" in problems[0]


def test_unfilled_template_rejected() -> None:
    """Открытый PR с нетронутым шаблоном не проходит."""
    text = TEMPLATE.read_text(encoding="utf-8")
    placeholders = template_placeholders(text)
    assert placeholders, "в шаблоне должны быть плейсхолдеры"

    problems = check_body(text, placeholders)
    assert problems
    assert any("шаблоны" in p for p in problems)


def test_placeholders_come_from_the_template_itself() -> None:
    """Список плейсхолдеров не дублируется в коде проверки.

    Иначе правка шаблона тихо отключила бы проверку заполненности.
    """
    placeholders = template_placeholders("текст <первый плейсхолдер> и <второй>")
    assert placeholders == {"<первый плейсхолдер>", "<второй>"}


def test_filled_body_passes() -> None:
    body = """## Что и зачем
Публикация корпуса: новые протоколы доходят до поиска.

## Владение
- Владелец: agent1 / pc1
- Зона изменений: scripts/, deploy/gcp-app/
- Зависит от PR: нет

## Чек-лист
- [x] Ветка создана от свежего origin/main
- [x] Пересечений нет
"""
    assert check_body(body, template_placeholders(TEMPLATE.read_text("utf-8"))) == []


def test_unticked_checklist_rejected() -> None:
    body = "## Владение\nВладелец: agent1 / pc1\n\n- [x] Первое\n- [ ] Второе\n"
    problems = check_body(body, set())
    assert problems
    assert "Второе" in problems[0]


def test_code_block_angle_brackets_do_not_trip_placeholders() -> None:
    """Стрелки в теле PR не должны выглядеть как плейсхолдеры шаблона."""
    body = "Владелец: agent1 / pc1\n\n```\nif a <= b and c >= d: pass\n```\n"
    assert check_body(body, template_placeholders(TEMPLATE.read_text("utf-8"))) == []


def test_size_warnings_are_not_errors() -> None:
    """Крупный PR предупреждает, но не блокирует: бывает обоснованно."""
    report = evaluate(
        branch="cursor/big-change-pc1",
        body="Владелец: agent1 / pc1",
        placeholders=set(),
        changed_files=144,
        additions=10409,
        deletions=487,
    )
    assert report.ok
    assert len(report.warnings) == 2


def test_small_pr_has_no_size_warning() -> None:
    assert check_size(3, 40, 5) == []


def test_ack_label_downgrades_errors() -> None:
    """Метка `hygiene-ack` разрешает обход, но обход остаётся видимым."""
    kwargs = {
        "branch": "some-bad-branch",
        "body": "",
        "placeholders": set(),
    }
    blocked = evaluate(**kwargs)
    assert not blocked.ok

    allowed = evaluate(**kwargs, acknowledged=True)
    assert allowed.ok
    assert any("hygiene-ack" in w for w in allowed.warnings)


def test_template_has_the_fields_coordination_depends_on() -> None:
    """Шаблон обязан спрашивать владельца, зону и зависимости.

    Без них соседняя вкладка не может решить, ждать ей merge или можно править.
    """
    text = TEMPLATE.read_text(encoding="utf-8")
    for field in ["Владелец", "Зона изменений", "Зависит от PR", "BUILD_VERSION"]:
        assert field in text, f"шаблон PR потерял поле {field}"
