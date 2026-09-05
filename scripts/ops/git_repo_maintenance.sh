#!/usr/bin/env bash
# Обслуживание локального .git: чистка агентских чекпоинтов и упаковка.
#
# Зачем: 2026-09-05 репозиторий занимал 4.2 ГБ при 690 МБ отслеживаемых файлов.
# Разбор показал две причины, обе не связанные с историей проекта:
#
#   1. 1.3 ГБ - три брошенных tmp_pack_* от прерванной упаковки (27 июля);
#   2. 980 МБ - один блоб corpus_vector_index/vectors.npy, который держал живым
#      ref из refs/codex/turn-diffs/checkpoints/. Файл закрыт .gitignore
#      (.gitignore:120), но чекпоинты агента снимают рабочее дерево вместе с
#      игнорируемыми файлами, поэтому .gitignore от такого попадания не спасает.
#
# После чистки: 4.2 ГБ -> 224 МБ. История, ветки и PR не тронуты: перезапись
# истории (filter-repo) и force-push для этого не потребовались.
#
# refs/codex/* локальные, на origin их нет, для истории проекта они не нужны -
# поэтому их удаление безопасно и повторяемо.
#
# Запуск: bash scripts/ops/git_repo_maintenance.sh [--dry-run]

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

human() { du -sh .git 2>/dev/null | awk '{print $1}'; }

echo "== .git до обслуживания: $(human)"

# 1. Брошенные временные пакеты. Валидный пакет всегда имеет .pack/.idx;
# tmp_pack_* без них - остаток прерванного gc/clone, git их не использует.
# read -a вместо mapfile: на macOS системный bash 3.2, mapfile там нет.
tmp_packs=()
while IFS= read -r line; do
  [[ -n "$line" ]] && tmp_packs+=("$line")
done < <(find .git/objects/pack -name 'tmp_pack_*' 2>/dev/null || true)
if ((${#tmp_packs[@]})); then
  size=$(du -ch "${tmp_packs[@]}" 2>/dev/null | tail -1 | awk '{print $1}')
  echo "-- брошенных tmp_pack: ${#tmp_packs[@]} ($size)"
  if ((DRY_RUN)); then
    printf '   [dry-run] удалить %s\n' "${tmp_packs[@]}"
  else
    # Страховка от гонки: не удаляем, пока идёт другая git-операция.
    if pgrep -x git >/dev/null 2>&1; then
      echo "   ПРОПУСК: работает другой процесс git, повтори позже" >&2
    else
      rm -f "${tmp_packs[@]}"
      echo "   удалено"
    fi
  fi
else
  echo "-- брошенных tmp_pack нет"
fi

# 2. Чекпоинты агентов. Это единственное, что держало 980 МБ.
codex_refs=()
while IFS= read -r line; do
  [[ -n "$line" ]] && codex_refs+=("$line")
done < <(git for-each-ref --format='%(refname)' | grep '^refs/codex/' || true)
if ((${#codex_refs[@]})); then
  echo "-- агентских чекпоинт-refs: ${#codex_refs[@]}"
  if ((DRY_RUN)); then
    echo "   [dry-run] удалить их и выполнить gc"
  else
    for r in "${codex_refs[@]}"; do git update-ref -d "$r"; done
    echo "   удалено"
  fi
else
  echo "-- агентских чекпоинт-refs нет"
fi

if ((DRY_RUN)); then
  echo "== dry-run: изменений не внесено"
  exit 0
fi

# 3. Упаковка. --prune=now допустим: недостижимые объекты после удаления
# чекпоинтов - именно тот мусор, который мы убираем.
echo "-- git gc --prune=now"
git reflog expire --expire-unreachable=now --all
git gc --prune=now --quiet

echo "== .git после обслуживания: $(human)"

# 4. Целостность: gc не должен ломать связность объектов.
if git fsck --connectivity-only --no-progress 2>&1 | grep -v '^dangling' | grep -q .; then
  echo "ВНИМАНИЕ: git fsck сообщил об ошибках - разберись до дальнейшей работы" >&2
  exit 1
fi
echo "== git fsck: связность в порядке"
