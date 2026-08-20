#!/usr/bin/env bash
# Ежедневная самопроверка качества МО: версия прода, golden КП, без PHI.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PROD_URL="${PROD_URL:-https://protocol.kravira.by}"
EXPECTED="$(python3 -c "import ast,pathlib; t=pathlib.Path('rag_server.py').read_text();
print(ast.literal_eval(t.split('BUILD_VERSION = ',1)[1].splitlines()[0]))")"

echo "== 1. prod version =="
live="$(curl -fsS "$PROD_URL/api/version" | python3 -c "import sys,json; print(json.load(sys.stdin).get('version',''))")"
echo "prod:     $live"
echo "repo:     $EXPECTED"
if [[ "$live" != "$EXPECTED" ]]; then
  echo "WARN: protocol.kravira.by ещё не на текущем BUILD_VERSION (нужен deploy_to_gce.sh)"
fi

echo "== 2. live health =="
curl -fsS "$PROD_URL/health/live" >/dev/null
echo "health/live ok"

echo "== 3. golden KP suggest =="
python3 -m pytest tests/test_kp_validity.py tests/test_mo_kp_suggest_golden.py --noconftest -q

echo "== 4. GCE month eval (напоминание) =="
cat <<'EOF'
Полный CSV только на GCE, one-off контейнер, не protocol-web:

  git -C /tmp/protocol-eval pull --ff-only
  sudo docker run --rm --name kp-eval-daily --volumes-from protocol-web \
    -v /tmp/protocol-eval/clinical_knowledge:/app/clinical_knowledge:ro \
    -v /tmp/protocol-eval/scripts/eval_mo_kp_suggest_month.py:/app/scripts/eval_mo_kp_suggest_month.py:ro \
    -e PYTHONPATH=/app -e CASE_PROTOCOL_SUGGEST=1 \
    -e PROTOCOL_CARDS_PATH=/app/output/registry/protocol_cards.jsonl \
    -e MO_DATA_ROOT=/var/data/medical_exams \
    protocol-gcp-app:staging \
    python3 /app/scripts/eval_mo_kp_suggest_month.py \
      --from 2026-07-26 --to 2026-08-13 \
      --out /var/data/medical_exams/reports/kp_suggest_eval_daily.json

Цели: omnibus_top1 / n_available ≤ 5%; adult_with_child_kp = 0;
available_pct расти только за счёт нозологических КП, не омнибуса.
EOF
