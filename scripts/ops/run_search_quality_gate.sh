#!/usr/bin/env bash
# Порог качества поиска на полном корпусе: eval + gate по pass_rate.
#
# ПОЧЕМУ НЕ В PR-CI
# Полному прогону нужен корпус протоколов, а в репозитории его нет:
# corpus_chunks_parts/ и data/catalog/corpus_path_manifest.jsonl в .gitignore,
# minzdrav_protocols/ в CI не выкачивается. В CI доступен только фикстурный
# мини-корпус из двух чанков - на нём golden-кейсы дают pass_rate 0.2
# (замер 2026-09-05), потому что кейсы написаны под полный корпус. Порог на
# таком прогоне ничего не охраняет.
#
# Поэтому в CI проверяется сам гейт (tests/test_quality_gate.py) и отбор на
# мини-корпусе (tests/test_search_golden.py), а этот скрипт запускается там,
# где есть корпус: на GCE или на машине разработчика.
#
# ПОЧЕМУ GEMINI ТОЛЬКО НА GCE
# --embed-on обращается к Gemini, а тот режет по IP клиента: из Беларуси
# приходит 400 "User location is not supported". См. .cursor/rules/gemini-via-gce.mdc.
# С Mac запускать только с --embed-off, иначе результат будет ложно низким.
#
# ИСПОЛЬЗОВАНИЕ
#   scripts/ops/run_search_quality_gate.sh                  # embed-off, порог 0.9
#   QUALITY_MIN_PASS_RATE=0.95 scripts/ops/run_search_quality_gate.sh
#   scripts/ops/run_search_quality_gate.sh --embed-on       # на GCE, с ключом API
#
# Коды возврата: 0 - порог пройден, 1 - ниже порога, 2 - ошибка прогона.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="python3"
if [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
fi

GOLDEN="${QUALITY_GOLDEN:-eval/golden_queries.jsonl}"
MIN_RATE="${QUALITY_MIN_PASS_RATE:-0.9}"
REPORT="${QUALITY_REPORT:-$(mktemp -t search_quality_XXXXXX).json}"

# По умолчанию embed-off: прогон без ключа API должен работать всегда.
# --embed-on передаётся аргументом осознанно и только там, где geo проходит.
EMBED_FLAG="--embed-off"
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --embed-on)
      EMBED_FLAG="--embed-on"
      if [[ -z "${GOOGLE_API_KEY:-}${GEMINI_API_KEY:-}" ]]; then
        echo "ОШИБКА: --embed-on без GOOGLE_API_KEY/GEMINI_API_KEY." >&2
        echo "Без ключа embed-rerank выключится молча, и pass_rate будет несопоставим." >&2
        exit 2
      fi
      ;;
    --embed-off) EMBED_FLAG="--embed-off" ;;
    *) EXTRA+=("$arg") ;;
  esac
done

if [[ ! -f "$GOLDEN" ]]; then
  echo "ОШИБКА: нет набора кейсов: $GOLDEN" >&2
  exit 2
fi

echo "== eval поиска =="
echo "   кейсы:  $GOLDEN"
echo "   режим:  $EMBED_FLAG"
echo "   порог:  $MIN_RATE"
echo "   отчёт:  $REPORT"

# Сам eval может вернуть ненулевой код из-за отдельных пограничных кейсов -
# решение о
# релизе принимает гейт по агрегату, поэтому код возврата eval здесь не фатален.
set +e
"$PY" eval/search_quality_eval.py \
  --golden "$GOLDEN" \
  "$EMBED_FLAG" \
  --report-json "$REPORT" \
  ${EXTRA[@]+"${EXTRA[@]}"}
eval_status=$?
set -e

if [[ ! -s "$REPORT" ]]; then
  echo "ОШИБКА: eval не создал отчёт (код $eval_status) - гейт проверять нечего." >&2
  exit 2
fi

echo
echo "== порог =="
"$PY" eval/quality_gate.py --report "$REPORT" --min-pass-rate "$MIN_RATE"
