#!/usr/bin/env bash
# Подключает отслеживаемые git-хуки из .githooks/.
#
# Зачем: правило .cursor/rules/no-ai-vendor-attribution.mdc ссылается на
# локальный .git/hooks/prepare-commit-msg, но файлы внутри .git не
# версионируются - на новой машине и в новом worktree их нет. В ветке
# cursor/production-readiness-agent1-pc1 из-за этого девять коммитов получили
# трейлер `Co-authored-by: Cursor`.
#
# core.hooksPath указывает на каталог в репозитории, поэтому одной настройки
# хватает на все worktree этого клона.
#
# Использование:
#   scripts/ops/install_git_hooks.sh          # подключить
#   scripts/ops/install_git_hooks.sh --check  # только проверить (для CI)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

HOOKS_DIR=".githooks"
want_check=0
[[ "${1:-}" == "--check" ]] && want_check=1

if [[ ! -d "$HOOKS_DIR" ]]; then
  echo "ОШИБКА: нет каталога $HOOKS_DIR" >&2
  exit 1
fi

current="$(git config --get core.hooksPath || true)"

if [[ $want_check -eq 1 ]]; then
  if [[ "$current" != "$HOOKS_DIR" ]]; then
    echo "core.hooksPath = '${current:-не задан}', ожидается '$HOOKS_DIR'" >&2
    echo "Подключить: scripts/ops/install_git_hooks.sh" >&2
    exit 1
  fi
  echo "core.hooksPath = $HOOKS_DIR"
  exit 0
fi

chmod +x "$HOOKS_DIR"/* 2>/dev/null || true
git config core.hooksPath "$HOOKS_DIR"

echo "core.hooksPath -> $HOOKS_DIR"
echo "Подключены хуки:"
for h in "$HOOKS_DIR"/*; do
  [[ -f "$h" ]] && echo "  - $(basename "$h")"
done
echo
echo "Проверка: git commit сохранит сообщение без vendor/AI-приписок."
