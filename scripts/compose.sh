#!/usr/bin/env bash
# Docker Compose wrapper that exports host UID/GID so bind-mounted files stay writable.
set -euo pipefail
cd "$(dirname "$0")/.."

export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"

exec docker compose "$@"
