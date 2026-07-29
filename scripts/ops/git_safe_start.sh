#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec "$ROOT/scripts/git_safe_start.sh" "$@"
