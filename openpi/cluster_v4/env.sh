#!/usr/bin/env bash

# v4 runtime environment (V4_PLAN.md §1). Reuses the portable v3.5 env contract verbatim:
# every mutable cache path derives from THIS checkout's root, so sourcing it from the v4
# worktree yields worktree-local caches (fresh JAX cache -- never shared with the live v3
# tree's cache, which corrupts under concurrent writers). The `v35/cache` namespace inside
# the worktree is scratch required by project_paths' runtime-path contract; v4 run artifacts
# live under `v4/`.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source cluster_v4/env.sh instead of executing it" >&2
  exit 2
fi

v4_env_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=../cluster_v35/env.sh
source "${v4_env_dir}/../cluster_v35/env.sh"

mkdir -p \
  "${MEMORY_PROJECT_ROOT}/v4/assets" \
  "${MEMORY_PROJECT_ROOT}/v4/checkpoints" \
  "${MEMORY_PROJECT_ROOT}/v4/diagnostics"

unset v4_env_dir
