#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec "$ROOT/scripts/deploy_after_push.sh" "$@"
