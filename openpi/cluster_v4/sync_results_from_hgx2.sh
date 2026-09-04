#!/usr/bin/env bash
# Copy a run's checkpoints (params + assets + metadata + run manifests; optimizer state skipped)
# from the node-local project copy on iris-hgx-2 back to NFS so batteries / serving / HF upload
# can use them from any host. Run from a fast-NFS host (iris-ws-18).
#   bash openpi/cluster_v4/sync_results_from_hgx2.sh <config> <exp> [--with-train-state]
set -u
config="${1:?config}"; exp="${2:?exp}"; with_state="${3:-}"
export HOME=/iris/u/kewalk
export KRB5CCNAME=FILE:/tmp/krb5cc_24706_claude
src=/scr/kewalk_v4/memory_project_v4/v4/checkpoints/$config/$exp
dst=/iris/u/kewalk/memory_project_v4/v4/checkpoints/$config/$exp
mkdir -p "$dst"
excl=(--exclude='*/train_state' --exclude='*/train_state/*')
[ "$with_state" = "--with-train-state" ] && excl=()
ssh -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR iris-hgx-2 "tar -C $src ${excl[*]} -cf - ." | tar -C "$dst" -xf -
echo "synced -> $dst: $(ls "$dst" | tr '\n' ' ')"
