#!/usr/bin/env bash
# Stage a COMPLETE copy of the v4 project on iris-hgx-2's local disk (/scr/kewalk_v4/memory_project_v4)
# so training runs from it: the node reads /iris at ~1-2 MB/s under load. Reuses what the v5
# staging already put on the node (local CPython, LeRobot parquet, Arrow cache, pi05_base weights:
# node-local copies / hardlinks, no NFS traffic) and streams only the v4-specific parts from a
# fast-NFS host (this script must run on iris-ws-18): code, venv, data JSONs + label files,
# v4 assets, the Stage-1 graft checkpoint params. Re-runnable. Marker: <root>/.staged
#   bash openpi/cluster_v4/stage_local_project_hgx2.sh
set -u
export HOME=/iris/u/kewalk
export KRB5CCNAME=FILE:/tmp/krb5cc_24706_claude
src=/iris/u/kewalk/memory_project_v4
v5local=/scr/kewalk_v5/memory_project_v5
pylocal=/scr/kewalk_v5/python
dst=/scr/kewalk_v4/memory_project_v4
log=$src/v4/diagnostics/stage_local_hgx2.log
errlog=$src/v4/diagnostics/stage_local_hgx2.err
S() { ssh -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR iris-hgx-2 "$@" 2>> "$errlog"; }
stamp() { echo "$(date +%H:%M:%S) $*" >> "$log"; }
stream() {  # label src-dir dst-subdir [tar-excludes...]
  local label="$1" from="$2" to="$3"; shift 3
  local t0=$(date +%s)
  tar -C "$from" "$@" -cf - . | S "mkdir -p $dst/$to && tar -C $dst/$to -xf -"
  stamp "$label: exit=${PIPESTATUS[1]} in $(( $(date +%s) - t0 ))s"
}
rm -f "$errlog"; stamp "stage start (v4 -> $dst)"
S "test -x $pylocal/bin/python3.11 && test -d $v5local/data/lerobot && test -d $v5local/v35/cache/openpi" || { stamp "v5 local copies missing on the node; abort"; exit 2; }
S "mkdir -p $dst && rm -f $dst/.staged && mkdir -p $dst/v35/{assets,checkpoints,diagnostics,tmp,wandb} $dst/v35/cache/{jax,uv,huggingface,openpi} $dst/v4/{assets,checkpoints,diagnostics} $dst/data"
# --- node-local reuse (same filesystem: hardlinks for the big read-only caches, seconds not hours)
t0=$(date +%s)
S "cp -al $v5local/data/lerobot $dst/data/lerobot 2>/dev/null || cp -a $v5local/data/lerobot $dst/data/lerobot; \
   cp -al $v5local/v35/cache/openpi/. $dst/v35/cache/openpi/ 2>/dev/null || cp -a $v5local/v35/cache/openpi/. $dst/v35/cache/openpi/; \
   mkdir -p $dst/v35/cache/huggingface/datasets && cp -al $v5local/v35/cache/huggingface/datasets/. $dst/v35/cache/huggingface/datasets/ 2>/dev/null || cp -a $v5local/v35/cache/huggingface/datasets/. $dst/v35/cache/huggingface/datasets/; \
   cp -a $v5local/v35/cache/huggingface/hub $dst/v35/cache/huggingface/ 2>/dev/null; cp -a $v5local/v35/cache/huggingface/modules $dst/v35/cache/huggingface/ 2>/dev/null; true"
stamp "node-local reuse (lerobot, weights, arrow cache, hub): in $(( $(date +%s) - t0 ))s"
# --- NFS streams from this host (code + venv small files in parallel groups)
(
  stream code "$src" . --exclude='./openpi/.venv' --exclude='./v35' --exclude='./v4' --exclude='./data' --exclude='./.claude' --exclude='./.remember'
  stream venv-skeleton "$src/openpi/.venv" openpi/.venv --exclude='./lib/python3.11/site-packages'
  t0=$(date +%s); sp=$src/openpi/.venv/lib/python3.11/site-packages; export sp dst errlog
  S "mkdir -p $dst/openpi/.venv/lib/python3.11/site-packages"
  ( cd "$sp" && ls -A ) | xargs -d '\n' -P 6 -n 40 bash -c 'tar -C "$sp" -cf - "$@" | ssh -o BatchMode=yes -o LogLevel=ERROR iris-hgx-2 "tar -C $dst/openpi/.venv/lib/python3.11/site-packages -xf -" 2>> "$errlog"' _
  stamp "venv-site-packages: exit=$? in $(( $(date +%s) - t0 ))s"
) &
( stream data-json "$src/data" data --exclude='./lerobot' ) &
( stream v4-assets "$src/v4/assets" v4/assets ) &
( stream stage1-graft "$src/v4/checkpoints/pi05_yam_mem_v4_stage1/v4_stage1_20260901_r3_h100/1000" v4/checkpoints/pi05_yam_mem_v4_stage1/v4_stage1_20260901_r3_h100/1000 --exclude='./train_state' ) &
wait
# --- relocate the venv to the local python and the local source tree
S "cd $dst/openpi/.venv && sed -i 's#^home = .*#home = $pylocal/bin#' pyvenv.cfg && for b in python python3 python3.11; do ln -sfn $pylocal/bin/python3.11 bin/\$b; done && cd lib/python3.11/site-packages && sed -i 's#/iris/u/kewalk/memory_project_v4/#$dst/#g' _editable_impl_openpi.pth _editable_impl_openpi_client.pth && cat _editable_impl_openpi.pth _editable_impl_openpi_client.pth" >> "$log" 2>&1
S "cd $dst/openpi && MEMORY_PROJECT_ROOT=$dst .venv/bin/python -c 'import sys, jax, openpi.shared.project_paths as pp; pp.configure_v35_runtime_environment(); pp.validate_executing_openpi_checkout(); print(\"local project ok\", sys.executable, jax.__version__, pp.memory_project_root())' && touch $dst/.staged && du -sh $dst/* $dst/data/* $dst/v35/cache/* 2>/dev/null && df -h /scr | tail -1" >> "$log" 2>&1
stamp "stage done staged=$(S "test -e $dst/.staged && echo yes || echo NO")"
