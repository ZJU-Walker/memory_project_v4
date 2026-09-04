#!/usr/bin/env bash
# Re-sync the CODE of the v4 project into the node-local copy on iris-hgx-2 after a commit
# (data, caches, checkpoints untouched). Run from a fast-NFS host (iris-ws-18).
#   bash openpi/cluster_v4/refresh_local_code_hgx2.sh
set -u
export HOME=/iris/u/kewalk
export KRB5CCNAME=FILE:/tmp/krb5cc_24706_claude
src=/iris/u/kewalk/memory_project_v4
dst=/scr/kewalk_v4/memory_project_v4
S() { ssh -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR iris-hgx-2 "$@"; }
for sub in openpi/src openpi/scripts openpi/cluster_v4 openpi/cluster_v35 openpi/examples openpi/packages; do
  tar -C "$src/$sub" -cf - . | S "mkdir -p $dst/$sub && tar -C $dst/$sub -xf -" && echo "refreshed $sub"
done
S "cd $dst && git -C $dst rev-parse --short HEAD 2>/dev/null; cd $dst/openpi && MEMORY_PROJECT_ROOT=$dst .venv/bin/python -c 'import openpi.training.config as c; print(\"local config ok:\", [n for n in (\"pi05_yam_mem_v4_stage4e\",\"pi05_yam_mem_v4_stage4f\") if c.get_config(n)])'"
