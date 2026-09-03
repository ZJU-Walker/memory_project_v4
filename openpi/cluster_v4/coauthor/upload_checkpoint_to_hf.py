"""Upload a trained checkpoint to the Hugging Face Hub so it can be downloaded and served elsewhere.

What goes up (project-relative paths are kept, so `download_checkpoint.sh` lands it where the
server and the evaluation scripts expect it):
  v4/checkpoints/<config>/<exp>/<step>/{params,assets,_CHECKPOINT_METADATA}   the policy (~10 GB)
  v4/checkpoints/<config>/<exp>/{v4_run_manifest.json,initialization_graft_manifest.json,wandb_id.txt}
  v4/diagnostics/train_<exp>.log                                              the training log
The optimizer state (train_state/, ~18 GB) is skipped unless --with-train-state.

Usage:
  python openpi/cluster_v4/coauthor/upload_checkpoint_to_hf.py                # newest step of the newest stage4e run
  python openpi/cluster_v4/coauthor/upload_checkpoint_to_hf.py --exp <exp>    # newest step of that run
  python openpi/cluster_v4/coauthor/upload_checkpoint_to_hf.py --checkpoint v4/checkpoints/<config>/<exp>/<step>
      [--repo kewalk123/openpi-v4-memory-artifacts] [--private] [--with-train-state] [--dry-run]
  python openpi/cluster_v4/coauthor/upload_checkpoint_to_hf.py --check-login  # token present? (run_all.sh, before training)

Token: `openpi/.venv/bin/huggingface-cli login` once, or HF_TOKEN=<token> (https://huggingface.co/settings/tokens,
"write" access). If the token cannot write to --repo (you are not its owner), the upload goes to
<your-user>/openpi-v4-memory-checkpoints instead (created on the fly, public unless --private) and the
exact download command is printed at the end -- send that line to the maintainer.
"""

import argparse
import pathlib
import re
import subprocess
import sys
import time

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

ROOT = pathlib.Path(__file__).resolve().parents[3]  # memory_project/
DEFAULT_REPO = "kewalk123/openpi-v4-memory-artifacts"
FALLBACK_REPO_NAME = "openpi-v4-memory-checkpoints"
DEFAULT_CONFIG = "pi05_yam_mem_v4_stage4e"
RUN_FILES = ("v4_run_manifest.json", "initialization_graft_manifest.json", "wandb_id.txt")
STEP_PATTERNS = ("params/**", "assets/**", "_CHECKPOINT_METADATA")


def _step_dirs(run_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        (d for d in run_dir.iterdir() if d.is_dir() and re.fullmatch(r"\d+", d.name) and (d / "params").is_dir()),
        key=lambda d: int(d.name),
    )


def resolve_checkpoint(args: argparse.Namespace) -> pathlib.Path:
    """Return the absolute step directory to upload."""
    if args.checkpoint:
        step_dir = (ROOT / args.checkpoint).resolve() if not pathlib.Path(args.checkpoint).is_absolute() else pathlib.Path(args.checkpoint)
        if not (step_dir / "params").is_dir():
            sys.exit(f"{step_dir} has no params/ directory (pass the <step> directory, e.g. .../<exp>/5999)")
        return step_dir
    ckpt_root = ROOT / "v4/checkpoints"
    if args.exp:
        runs = [d for d in ckpt_root.glob(f"*/{args.exp}") if d.is_dir()]
        if not runs:
            sys.exit(f"no run named {args.exp!r} under {ckpt_root}")
    else:
        cfg_dir = ckpt_root / DEFAULT_CONFIG
        runs = [d for d in cfg_dir.iterdir() if d.is_dir()] if cfg_dir.is_dir() else []
        runs = [d for d in runs if _step_dirs(d)]
        if not runs:
            sys.exit(f"no {DEFAULT_CONFIG} run with a saved step under {ckpt_root}; pass --exp or --checkpoint")
        runs.sort(key=lambda d: d.stat().st_mtime)
    run_dir = runs[-1]
    steps = _step_dirs(run_dir)
    if not steps:
        sys.exit(f"{run_dir} has no saved checkpoint step yet (training saves every 500 updates)")
    return steps[-1]


def dir_size_gb(path: pathlib.Path, skip: tuple[str, ...] = ()) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and not any(part in skip for part in p.relative_to(path).parts):
            total += p.stat().st_size
    return total / 1e9


def writable_namespace(api: HfApi, repo: str) -> tuple[bool, dict]:
    who = api.whoami()
    namespace = repo.split("/")[0]
    orgs = {o.get("name") for o in who.get("orgs", [])}
    role = (who.get("auth") or {}).get("accessToken", {}).get("role")
    can_write = (namespace == who["name"] or namespace in orgs) and role in (None, "write", "fineGrained")
    return can_write, who


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="step directory (project-relative or absolute)")
    parser.add_argument("--exp", help="experiment name; uploads its newest saved step")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--private", action="store_true", help="fallback repo (if created) is private")
    parser.add_argument("--with-train-state", action="store_true", help="also upload train_state/ (optimizer, ~18 GB)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-login", action="store_true", help="only verify that a Hub token is available")
    args = parser.parse_args()

    api = HfApi()
    try:
        who = api.whoami()
    except Exception as exc:  # noqa: BLE001 - any auth failure reads the same to the user
        sys.exit(
            "[upload] no Hugging Face token: run `openpi/.venv/bin/huggingface-cli login` (write token from "
            f"https://huggingface.co/settings/tokens) or export HF_TOKEN=<token>. ({type(exc).__name__}: {exc})"
        )
    role = (who.get("auth") or {}).get("accessToken", {}).get("role")
    print(f"[upload] logged in as {who['name']} (token role: {role or 'unknown'})", flush=True)
    if args.check_login:
        if role == "read":
            sys.exit("[upload] the stored token is read-only; uploads need a token with write access")
        return

    step_dir = resolve_checkpoint(args)
    run_dir = step_dir.parent
    exp = run_dir.name
    config = run_dir.parent.name
    rel_step = step_dir.relative_to(ROOT).as_posix()
    log_path = ROOT / "v4/diagnostics" / f"train_{exp}.log"
    patterns = list(STEP_PATTERNS) + (["train_state/**"] if args.with_train_state else [])
    size_gb = dir_size_gb(step_dir, skip=() if args.with_train_state else ("train_state",))

    repo = args.repo
    can_write, _ = writable_namespace(api, repo)
    if not can_write:
        repo = f"{who['name']}/{FALLBACK_REPO_NAME}"
        print(f"[upload] token cannot write to {args.repo}; using {repo} instead", flush=True)

    print(f"[upload] checkpoint {rel_step}  (config {config}, run {exp}, {size_gb:.1f} GB) -> {repo}", flush=True)
    run_files = [run_dir / n for n in RUN_FILES if (run_dir / n).is_file()]
    for p in run_files + ([log_path] if log_path.is_file() else []):
        print(f"[upload]   + {p.relative_to(ROOT).as_posix()}", flush=True)
    if args.dry_run:
        print("[upload] dry run; nothing uploaded")
        return

    api.create_repo(repo, repo_type="model", private=args.private, exist_ok=True)
    started = time.time()
    try:
        api.upload_folder(
            folder_path=str(step_dir),
            path_in_repo=rel_step,
            repo_id=repo,
            repo_type="model",
            allow_patterns=patterns,
            commit_message=f"upload {rel_step}",
        )
        for p in run_files:
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=p.relative_to(ROOT).as_posix(),
                repo_id=repo,
                repo_type="model",
                commit_message=f"upload {p.relative_to(ROOT).as_posix()}",
            )
        if log_path.is_file():
            api.upload_file(
                path_or_fileobj=str(log_path),
                path_in_repo=log_path.relative_to(ROOT).as_posix(),
                repo_id=repo,
                repo_type="model",
                commit_message=f"upload training log for {exp}",
            )
    except HfHubHTTPError as exc:
        sys.exit(f"[upload] Hub refused the upload to {repo}: {exc}\n         (403 = the token has no write access to that repo; re-run with --repo <your-user>/<name>)")
    print(f"[upload] done in {time.time() - started:.0f}s: https://huggingface.co/{repo}/tree/main/{rel_step}", flush=True)
    print(
        "[upload] to fetch it on another machine (inside a clone of the code repo):\n"
        f"    bash openpi/cluster_v4/coauthor/download_checkpoint.sh {repo} {rel_step}",
        flush=True,
    )
    git_head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if git_head:
        print(f"[upload] code revision of this checkout: {git_head}", flush=True)


if __name__ == "__main__":
    main()
