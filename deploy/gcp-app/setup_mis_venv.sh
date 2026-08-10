#!/usr/bin/env bash
# Create /opt/protocol/venv-mis with PyMySQL stack for night extract.
set -euo pipefail

ROOT="${PROTOCOL_ROOT:-/opt/protocol}"
VENV="${MIS_VENV:-/opt/protocol/venv-mis}"
REQ="${ROOT}/requirements-mis-bridge.txt"

if [[ ! -f "$REQ" ]]; then
  echo "missing $REQ" >&2
  exit 2
fi

if [[ ! -w "$(dirname "$VENV")" ]] || [[ -d "$VENV" && ! -w "$VENV" ]]; then
  sudo mkdir -p "$(dirname "$VENV")"
  sudo python3 -m venv "$VENV"
  sudo "${VENV}/bin/pip" -q install --upgrade pip
  sudo "${VENV}/bin/pip" -q install -r "$REQ"
  sudo chown -R "$(whoami):$(whoami)" "$VENV"
else
  python3 -m venv "$VENV"
  "${VENV}/bin/pip" -q install --upgrade pip
  "${VENV}/bin/pip" -q install -r "$REQ"
fi
"${VENV}/bin/python" -c "import pymysql, sqlalchemy, pandas, pyarrow; print('MIS_VENV_OK', '$VENV')"
