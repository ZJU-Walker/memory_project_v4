# V4 Final Plan — Dual-Bank Visual + Semantic TTT Memory

**Status:** Finalized against the real repo (standalone repository at `memory_project_v4`, branch `v4`)
**Supersedes:** the v1.0 architecture draft (`V4_Dual_Bank_Visual_Semantic_TTT_Memory_Plan_EN.md`). All seven design invariants of the draft are preserved; §2 records the places where the implementation deliberately differs from the draft and why.
**Non-interference contract:** the old `memory_project` tree stays the v3 line (the v36 pilot and every older run remain testable there). This repository never reads from or writes into it.

---

## 1. Repo isolation (done; standalone since 2026-09-01)

| Item | Value |
|---|---|
| Repository | standalone git repo, branch `v4` (full history preserved back through the v3 lineage; formerly a worktree of `memory_project`) |
| Data | `data/` is a REAL in-repo directory (gitignored, ~40G): the converted v36 LeRobot dataset (39G, byte-verified copy), every `0830_0831*` manifest JSON incl. approval ledger + inventory, the v4 fact sidecar, AND the three raw collection folders the frozen manifest references (`0830_bin_part1`, `0830_bin_part2`, `0831_bin`, ~1G) — the v3.5 frozen-record validator hashes each episode's raw `subtask_labels.json` at loader init. Only 0830+0831 data exists here; 0816 was excluded at the v36 freeze (Gate-B leak) and is not present in any form. |
| v4 artifacts | `v4/{diagnostics,assets,checkpoints}` (gitignored) |
| Python env | own `openpi/.venv` via `GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen`; recreate the same way on a new cluster |
| Caches | repo-local under `v35/cache/` (the path name is the portable runtime contract of `project_paths`): JAX cache, HF caches, and the pi05_base checkpoint cache (12G, pre-seeded). A new cluster re-downloads pi05_base automatically if absent. |
| GPUs | Stage 1 fits a single 24G GPU (A5000-verified); later stages size like v3.5 (4x H100). |

Legacy note: `openpi/cluster_v34/` and some pre-v4 scripts reference the old absolute paths and raw-data workflows; they are kept as history and are inert for v4. `project_paths.SHARED_DATA_LINKS` (the symlink affordance used during the worktree phase) remains supported but unused now that `data/` is real.

---

## 2. Finalization decisions (deltas from the draft)

The draft was architecture-level; these are the binding implementation decisions:

1. **Stage 0 is already done.** The shared TTT core is the v3.5 `TitansMemory` in `delta_output`/`pooled_frame` mode (`openpi/src/openpi/models/memory.py`) — commit→retain→read→use was validated by the v3.5/v36 Gate-C battery (13/13 core checks, exact rank-one commit residuals, analytic decay). Both banks reuse it unchanged. No new memory core is built for v4.
2. **The visual bank IS the v36 pathway, unchanged.** Layer-8 dual-query compressors, pooled-frame delta write, tanh_rms injection, blind memory tokens with zero-init slot embeddings, reseed-CE, the v3.5 write/decision masks — all kept, same module names (`self.memory`, `read_query_compressor`, …) so a v36 checkpoint grafts directly through `AuditedPartialCheckpointWeightLoader`.
3. **The semantic bank does not use the hidden-token projection path at all.** It is a second `TitansMemory` driven purely through the *explicit key-space API* (`delta_write_kv` / `read_key`) that the v3.4 refactor created for exactly this purpose. Keys come from a learned fact-slot table, values from a memory-blind fact head (§4). This makes "no teacher forcing" and "no memory echo" structural properties, not training-time disciplines.
4. **Structured facts are a typed head first, text later.** The draft's `<phase> + <memory_fact>` *text* output is deferred to v4.1. v4-Base supervises facts with a per-slot classification head (the direct generalization of v3.5's `memory_write_side_head`). Reasons: (a) the subtask string is the *only* label channel and its vocabulary is string-matched in three independent places (`label_subtasks._phase_of/_side_of`, `MemoryV34Labels`, converter preflight) plus the frozen manifest `task_vocabulary` — a grammar change forces a full relabel + re-freeze; (b) `causal_token_len=128` leaves little CE budget; (c) the draft itself forbids teacher-forced fact tokens from feeding the write, so text emission is presentation, not mechanism. The typed head satisfies draft invariant 5 (persistent facts ≠ transient phases) by construction.
5. **No new dataset, no new labels, no re-freeze for v4-Base.** Fact labels derive *deterministically* from the frozen v36 manifest (`data/0830_0831_episode_manifest_v36_frozen.json`, sha `9085fe50…`, sides 3×-verified): `target_side=left` ⇒ facts {banana→left_bin, grey_pepper_box→right_bin}. Same 70 episodes, same 54/8/8 seed-36 split, same pinned norm stats. Gate A reuses the identical frozen inputs; the derivation gets its own hashed sidecar + audit script. 0816 stays excluded (Gate-B leak, 0.82 p=.001).
6. **Fusion stays minimal for v4-Base.** Both banks inject as separate token groups with separate calibrated gates and bank type embeddings; the upper blocks select via attention. No learned cross-bank mixer, no bank dropout unless Stage-4 ablations show a permanently ignored bank (draft's contingency, kept as contingency).
7. **Capacity:** V4-Base = both banks at the current geometry (draft's ~9.4M total; in delta mode the per-episode fast state is just `w3` [1024×2048] per bank). V4-Large (asymmetric ~20M) stays gated on *measured* interference, exactly as the draft demands.

---

## 3. Architecture mapped to code

Timestep flow (all inside `_compute_sequence_loss_v32` / `sample_with_memory`):

```
h8 = blocks 0..8 (memory-blind by construction: reads are injected AFTER block 8)
  ├── visual read:   read_query_compressor(h8_top) → project_q → read_key(M_vis)       [16 tokens]
  ├── semantic read: read_key(M_sem, L2Norm(fact_keys))                                 [F tokens]
  ├── inject: per-bank tanh_rms gates + per-bank slot/type embeddings → memory tokens
  ├── blocks 9..17 over [images | context | mem_vis | mem_sem | causal] → subtask CE + flow
  ├── visual write:  write_query_compressor(h8_top) → project_kv → pool → delta_write_kv(M_vis)   (E-steps)
  └── semantic write: fact head(h8) → per-slot (key from table, value = soft target embedding)
                      → delta_write_kv(M_sem)                                            (gated E-steps)
```

Read-before-write ordering per step is already the v3.5 contract (`delta_write_kv` is documented as a transition API: current-step read precedes it). It now covers both banks.

**New modules on `Pi0` (all under `memory_v4_dual_bank=True`, default-off, bit-identity with v36 config when off — repo convention):**

- `self.memory_semantic = TitansMemory(config.memory_semantic, rngs)` — independent `MemoryState`, so one bank can never overwrite the other (draft invariant 1).
- `self.fact_keys` — `nnx.Param[F_max, d_key]`, L2-normalized at use (same pattern as `memory_aux_queries`). One row per (entity, relation) slot. Bin task uses F=2 of a static `F_max=8` (static shapes for jit; unused slots masked).
- `self.fact_head` — memory-blind fact predictor on h8: instruction-row cross-attention (reuse `MemoryQueryConditioner` pattern with `instruction_only` context — the v3.4 lesson: state rows leak phase) → per-slot logits over the fact-target vocab `{left_bin, right_bin, unknown}`.
- `self.fact_value_embed` — `nnx.Param[num_targets, d_value]`; the written value is `L2Norm(softmax(logits_slot) @ fact_value_embed)` — the draft §6.1 "soft embedding of the predicted distribution". Never the ground-truth token, never a post-read representation.
- `self.memory_sem_inject_w`, `self.memory_sem_slot_embedding` (zero-init — v36 lesson: any nonzero init breaks the 1.02× step-0 transparency bound), `self.memory_sem_type_embedding` / visual type embedding — bank identity visible at fusion (draft invariant 2).
- `self.memory_fact_read_head` — read-side supervision on D-valid steps from raw semantic retrieval (generalizes `memory_read_side_head`).

**Config (`Pi0Config`):** `memory_v4_dual_bank: bool = False`, `memory_semantic: MemoryConfig` (v4-Base: identical geometry to `memory`), `memory_fact_slots: int = 8`, `memory_fact_targets: int = 3` (left/right/unknown), `memory_fact_write_conf: float = 0.9`, `memory_fact_write_loss_weight`, `memory_fact_read_loss_weight`, plus a `__post_init__` block enforcing the v4 invariants the way `memory_v35_enabled` does (delta_output both banks, blank_initial_output, tanh_rms, frozen gates at init, zero-init semantic slot embeddings).

**Semantic write policy (draft §7), implemented with existing masks:**
- Eligible only on `seq_write_mask` E-steps (evidence phase, lids open — answer-related writes stop at close, as required for the occlusion task).
- Event-driven: a slot writes only when `max softmax > memory_fact_write_conf` and argmax ≠ `unknown`; otherwise that step is decay-only for the bank (`analytic_decay` — exact under the sparse clock).
- Sparse by construction: F=2 associations per episode vs the visual bank's per-E-step stream.
- Both banks share the v3.5 sparse clock (`alpha_step=0.01` per 15-frame step, analytic gap collapse). A separate, smaller semantic alpha is a recorded knob, not a v4-Base change.

**Inherited landmines (all remain binding):** FP32 matmul pin (TF32 breaks the exact-arithmetic contract — memory.py already pins, keep it for every new einsum on the semantic path), fp32 comparison for `alpha_step`, orbax structural-None merge on load, RMSNorm zero-stream (semantic tokens get the same zero-init slot-embedding treatment under blinding), `state_cotangent_clip`/`kv_cotangent_clip` configured **per bank**, per-bank telemetry in `v35_runtime_identity`/`v35_cumulative_telemetry` (the scan carry now holds two `MemoryState`s — checkpoint & iterator-state schema versions bump).

---

## 4. Write integrity (draft §6) — how each rule is enforced structurally

| Draft rule | Enforcement in code |
|---|---|
| No teacher-forcing leakage | The write value is built only from `fact_head(h8)` logits. Ground-truth fact labels touch only the CE loss on those logits — there is no code path from a label embedding to `delta_write_kv`. |
| No memory echo | `fact_head` consumes h8, which is computed before any injection (block-8 boundary). Retrieved tokens enter only blocks 9..17. A unit test asserts the fact-head output is invariant to arbitrary `M_vis`/`M_sem` contents. |
| Memory-blind writer vs memory-conditioned policy | Already the architecture: CE/flow read the fused stream; both writers read h8. |
| Phase tokens never overwrite facts | Semantic writes exist only for fact slots; phases live in the causal CE and never reach `M_sem`. |
| Unknown ⇒ no write | The `unknown` target and the confidence gate map to decay-only steps. |

---

## 5. Stage plan → concrete runs

**Stage 0 — shared core.** Done (v3.5/v36). One addition: run the synthetic key-space battery (`v35_rung_collect` stage-0 pattern) on the fact-key geometry — F up to `F_max=8` near-orthogonal keys through `delta_write_kv`/`read_key`, measuring cross-slot interference of the rank-one updates. Pure CPU/1-GPU, no training.

**Stage 1 — semantic understanding without memory** (first v4 training run, cheap, single-frame — no sequence scan):
- Config `pi05_yam_v4_stage1`: pi05 + `predict_subtask` path + fact head on h8; no memory modules active.
- Train on the v36 train-54; supervise fact CE only on frames where the fact is *observable* (inspect phase ∩ E-visibility sidecar); label `unknown` elsewhere.
- Pass gates (all machinery exists): fact accuracy on fresh dev episodes; **cross-session transfer** (train 0830→eval 0831 and reverse — the writer-signal-is-session-specific failure of v3.4 is the #1 risk here); prompt swap; state masking (`state_null_embedding`); arm-region patch masking; `unknown` before lids open; class balance. Plus a Gate-B-style probe: episode-OOF side decode from fact-head *features* on pre-evidence frames must be at chance.
- **This is the go/no-go for the whole semantic branch**: if facts aren't visually grounded on fresh data, nothing downstream is authorizable.

**Stage 2 — semantic memory only** (visual injection frozen at zero):
- **2a oracle writes:** write `L2Norm(fact_value_embed[gt_target])` at eligible E-steps (pattern: `_v35_oracle_injected_content`; see also the old `oracle_memory` branch). Trains the read/fuse/use path in isolation. Gates: exact commit residuals (delta rule is analytic), retention over the occlusion horizon, D-step read-head accuracy, reset→chance and donor-swap→donor's answer flips (episode-level, `memory_swap_read_step` extended per bank).
- **2b predicted writes:** swap oracle for the Stage-1 fact head. The 2a↔2b gap *is* the remaining perception error, cleanly measured.

**Stage 3 — visual memory only.** This *is* v36, running now on `main`. Its result slots in unchanged; v4 never reruns it. If v36's pilot fails its gates, the visual-bank fix happens on the v3 line and merges forward — the semantic thread (Stages 1–2) is not blocked.

**Stage 4 — dual-bank fusion.** Both banks on, `pi05_yam_mem_v4` full config, through the cloned gated pipeline (§6). Eval battery per bank and combined: {both, visual-only, semantic-only} × {reset, donor-swap, both-swapped}. Bank dropout only if one bank is provably ignored.

**Stage 5 — generalization/capacity.** Multi-collection training, larger fact vocab, and — required before any "visual bank is necessary" claim (draft §4.2) — a benchmark needing non-semantic visual detail (appearance/pose matching). Task design starts only after Stage 4 passes. V4-Large only on measured interference.

---

## 6. Gates & authorization (clone, don't fork-in-place)

Clone `cluster_v35/` → `cluster_v4/` and the `v35_prepare_pilot.py` orchestrator with:
- New `CONFIG_NAME = "pi05_yam_mem_v4"` — a real new `TrainConfig` (v36 overloaded the v35 name via `project_paths`; don't repeat that), new `V4_*` constants in `openpi/src/openpi/shared/project_paths.py`, artifacts under `v4/diagnostics/runs/<exp>/`, checkpoints under `v4/checkpoints/`.
- Frozen hashes: manifest/dataset/norm hashes stay the v36 values (§2.5); add the fact-label derivation sidecar hash to Gate A.
- Gate B: existing two preregistered probes + a third preregistered probe on fact-head features (pre-evidence frames, episode-OOF, permutation null).
- Stage-06 calibration: per-bank c/tau (the semantic bank's retrieval RMS differs from the visual bank's; one shared calibration would mis-scale one of them).
- Gate C: the 13 core checks + delta-identity checks run per bank; step-0 task health keeps the 1.02× transparency bound with *both* injections armed.
- Two-stage sha-sealed authorization flow unchanged (`v35_training_authorization.py` pattern, new criteria-version strings `openpi.v4.*`). Note: pass `--pilot-authorization` explicitly — the `v35_train.py` default points at a directory real runs never populate.
- Every new script gets its paired `*_test.py`, matching the existing scaffolding.

---

## 7. Success criteria & claim boundary

Unchanged from the draft §13: grounded facts on fresh data (Stage 1), per-bank commit/retain/read (2a, v36), causal reset/swap flips (2, 4), semantic control of fact decisions (2b), visual value beyond facts (Stage 5 benchmark), correct source selection when fused (4). The draft's non-claims also stand — in particular v4-Base does **not** claim unsupervised fact emergence (labels are derived), fully learned write timing (the confidence gate + phase mask are scaffolding), or that the bin task needs the visual bank.

---

## 8. Status (2026-09-01)

Done (commits `6a1b979`, `a897550`):
1. Worktree venv + `cluster_v4/{env.sh,stage1_train.sh}` (worktree-local caches, fresh JAX cache).
2. Chunk A: dual-bank model core -- `delta_write_kv_multi`, independent semantic bank via the
   key-space API, memory-blind fact head on h8, `_memory_token_total` layout property, full
   `_compute_sequence_loss_v32` integration, `v4_fact_probe_step`. Bit-identity when off proven
   by the complete v3.2/v3.3/v3.4/v3.5 regression suites.
3. Chunk B: fact sidecar (`data/v4_fact_labels_0830_0831.json`, triple-authenticated in the
   loader), `MemoryV4FactLabels` transform, configs `pi05_yam_mem_v4` + `pi05_yam_mem_v4_stage1`.
4. train.py: class-balanced macro fact CE + read-side CE in both loss paths; Stage-1
   calibration-lock carve-out (`_is_v4_stage1_config`; the lock is vacuous, not bypassed --
   nothing trains through any injection; any shape drift reinstates it); host-side v3.5
   machinery keyed off for Stage-1 while the device-side accounting guard stays on; per-bank
   gate validation; two latent bugs fixed (None-leaf BF16 cast, pre-cast graft target spec).
5. Stage-1 battery `scripts/v4_stage1_eval.py`: evidence accuracy + write eligibility,
   pre/post abstention, prompt-swap + state-neutral agreement (dev split), Gate-B-style
   episode-OOF leak probes with stratified permutation nulls (train split -- an 8-episode
   split's ~36-arrangement permutation space cannot support the 0.05 gate).
6. Smoke green on the A5000 (`v4/diagnostics/stage1_smoke_r5.log`): semantic injection RMS
   exactly 0, raw retrieval 0.65, 36 commits / 0 degenerate, guard clean, checkpoint saved.
7. **Stage-1 training launched**: `v4_stage1_20260901_r1`, 4000 steps, batch 2, A5000
   (`v4/diagnostics/stage1_20260901_r1.log`; checkpoints every 500 under
   `v4/checkpoints/pi05_yam_mem_v4_stage1/v4_stage1_20260901_r1`).

**Stage-1 verdict (2026-09-01 14:37): PASS.** r1 (A5000, batch 2, LR 1e-4, abstention only on
decision steps) failed the two capability gates at ckpt-1000 (evidence accuracy 0.50 on dev AND
train; pre-abstention 0.59) while all purity gates passed; the h8 linear probe showed the fact is
present in frozen layer-8 features (0.98 OOF with 4x4 spatial pooling). After the fixes
(`unknown` supervised on every non-observable step; LR 3e-4; batch 4 with 12 workers on an H100),
r3 `v4_stage1_20260901_r3_h100` ckpt-500 passes all 7 gates: evidence accuracy 0.990
(banana 1.000 / pepper 0.979; 0830 0.979 / 0831 1.000), pre-abstention 1.000, post-abstention
1.000, prompt-swap 0.979, state-neutral 0.990, leak probes p=0.57/0.87 (train split, OOF);
write-eligible rate 0.90. Report: `v4/diagnostics/stage1_eval_r3_500/stage1_eval.json`.
H100 note: batch >= 6 on one H100 dies with CUDA_ERROR_ILLEGAL_ADDRESS at step 1
(`v4/diagnostics/h100_isolation*/summary.txt`); batch 4 / 12 workers is the proven recipe.

**Stage-1 confirmed at ckpt-1000** (all gates 1.000; leak p 0.70/0.68). Stage-1 artifact:
`v4/checkpoints/pi05_yam_mem_v4_stage1/v4_stage1_20260901_r3_h100/1000/params`.

**Protocol decision (user, 2026-09-01 15:12): v4 is a clean break from the v3.5 seal.** Every
v4 config sets `v4_protocol=True` (train.py): plain train step (no checkified guard), no
calibration lock / pilot authorization / bootstrap-0 resume / telemetry ledger, a small
`v4_run_manifest.json` per run (config identity, git commit, graft sources, init-tree hash),
and `v4_graft_sources` overlays (Stage 2a takes the trained fact head from the Stage-1
checkpoint; the backbone still comes from pi05_base). Adversarial testing happens through
the v4 batteries once each stage's pieces exist.

**Stage 2a** (`pi05_yam_mem_v4_stage2a`): oracle semantic writes on observable E steps,
visual injection off, fact head frozen, semantic gate 0.5 with the v3.4 constant c=12.4
(pinned, not calibrated), backbone / action expert / semantic core / slot embeddings / read
head training at the v3.5 schedule, 1000 updates. Battery: `scripts/v4_stage2_eval.py
--params <ckpt>/params --output-dir ...` (normal / reset / donor read-side interventions on
decision steps; decision-step subtask CE, use-pressure flow, read accuracy; provisional
causal gates). H100 recipe: SigLIP + embedder frozen (FP32 masters + Adam OOM otherwise),
batch 2 per device, 12 workers (batch >= 6 on one device dies with
CUDA_ERROR_ILLEGAL_ADDRESS at step 1).

**Stage 2a r2 postmortem (2026-09-01, GPU 3, stopped at step 644): the memory path never
trained.** `v4_fact_read_loss` sat at ln(3) (1.075 -> 1.065) while CE/flow fell normally;
the ckpt-250 leaf diff showed `memory_fact_read_head`, `memory_sem_slot_embedding` and the
semantic core bitwise unchanged with `action_out_proj` moving 2.3%. Root cause, confirmed
by `v4/diagnostics/head_grad_probe.py` on the real ckpt-250 + dev batch: the read head's
own gradient is healthy (norm 0.12-0.15) but `memory_sem_slot_embedding`'s is **inf**, so
train.py's memory-group pre-clip (`memory_grad_clip=5.0`, `scale = 5/(inf) = 0`) multiplied
EVERY memory-path gradient by exactly zero from step 0 (`memory_grad_norm=inf` in the log at
every logging step). The inf is the v3.4 exactly-zero-token-stream RMSNorm singularity: in
2a all 16 visual tokens (injection off, zero-init slot embeddings) and the 8 semantic tokens
before the first oracle commit are exactly zero, every late block's RMSNorm at zero
multiplies the K/V cotangent other rows send them by rsqrt(eps)=1e3, and the chain overflows
on its way back to the slot embeddings. v36 got away with it only because its visual
retrieval stops being exactly zero after the first write.
Fix (this commit): `Pi0Config.memory_mask_zero_tokens` (required for every v4 config) --
the interface reports per-slot validity (`memory_valid = any(token != 0)`) on the cast
tokens, and `_v32_prepare_memory_prefix` / `_v32_causal_mask` / `_v32_step_mask` /
`_v32_suffix_mask` drop invalid slots from every row's key set (a memory row always keeps
its own key so no softmax row is all-masked; blind rows see valid memory keys + self). An
exactly-zero token is therefore invisible -- a blank bank cannot influence the policy and no
cotangent enters its stream -- while written slots stay live. Unit contract:
`pi0_v4_test.py::test_mask_zero_tokens_hides_blank_slots_and_cuts_their_gradient` (mask
off: nonzero slot-embedding gradient from zero K/V tokens; mask on: exactly zero for blank
slots, nonzero for live ones; Stage-2a sequence objective finite with identical read-term
counts). All v3.2/v3.4/v3.5/v4 model suites pass; the v3.x geometry is bit-identical with
the flag off (default). Verification on the real model: `v4/diagnostics/memory_group_grad_probe.py`
(per-leaf memory-group gradient norms, mask off vs on, ckpt-250 + dev batch) must report a
finite group norm with the mask on before r3 launches. Stage-1's artifact is unaffected
(the fact head reads h8 before the memory tokens; its gates never touched the late blocks).
Verified on the real ckpt-250 + dev batch (`v4/diagnostics/memory_group_grad_probe_2a_r2_250.log`,
train mode, 19 trainable memory-group leaves): mask off -> group norm **inf** (both slot
embeddings inf; the semantic core reads 0 because its state-cotangent clip saw the inf);
mask on -> group norm **1.32**: semantic core m0 w0/w1/w2 0.45-0.64, read head 0.15/0.12,
semantic slot embedding 0.05 (from the post-commit live slots), visual slot embedding
exactly 0 (never attended). CE moves 4.112 -> 4.132 (the zero-K/V softmax sink is gone).
**r3 launched 2026-09-01 18:44 PDT** on GPUs 2+3 (`CUDA_VISIBLE_DEVICES=2,3 --fsdp-devices 2
--batch-size 4`, batch 2 per device, pid 2926227), exp `v4_stage2a_20260901_r3`, log
`v4/diagnostics/stage2a_20260901_r3.log`; health signal = `memory_grad_norm` finite and
`v4_fact_read_loss` falling below ln(3) by step ~200. Battery on GPU 3 after the run.

**r3 result (2026-09-01 21:42, 1000 updates, checkpoints 250/500/750/999):** the memory path
trained -- `memory_grad_norm` finite at every interval (3.1 -> 0.48), `v4_fact_read_loss`
1.075 -> 0.327, CE 15.8 -> 2.05, flow 0.148 -> 0.0096.
**Stage-2 battery, ckpt-999 (`v4/diagnostics/stage2_eval_2a_r3_999/stage2_eval.json`, dev
split, 12x4 sequences): PASS=False, 3/5 gates.** Read side is perfect and causal: read
accuracy 1.000 normal, 0.500 under reset, 0.504 under donor swap (donor mismatched-pair
fraction 0.54 -> the head reports the DONOR's fact). Use side is weak: decision-step subtask
CE 2.155 normal / 2.465 reset (ratio 1.144 < 1.2) / 2.310 donor (ratio 1.072 < 1.2);
use-pressure flow 0.0105 / 0.0116 reset (ratio 1.103, passes 1.1). So the backbone reads
"a memory is present" more than "which fact": wrong content hurts less than no content.
Note raw decision-step retrieval RMS 0.0116 < tau 0.02, i.e. the non-amplifying floor binds
and the semantic tokens enter at ~0.5*12.4*0.58 = 3.6 RMS (~29% of residual scale), not
the nominal 6.2. Losses were still falling at step 999; the 2a use gates are provisional.
**ckpt-500 battery (`stage2_eval_2a_r3_500/`): PASS=False, 4/5** -- read 1.000 / reset
0.500 / donor 0.504 (same as 999); decision CE 2.166 normal / **2.916 reset (ratio 1.346,
passes)** / 2.370 donor (ratio 1.094, fails); use-flow ratio 1.115 (passes).
**Trend 500 -> 999: memory USE erodes with more training** while the normal decision CE
stays flat (2.166 -> 2.155): reset ratio 1.346 -> 1.144, donor ratio 1.094 -> 1.072. The
backbone found a route to the same CE that does not need the memory (phase/visual cues in
the dev decision steps, cf. the v3.4 leak history), and it never learned content
dependence (wrong memory costs 7-9%, no memory costs 14-35%). Consequences: (1) do NOT
simply resume r3 for more updates -- that goes the wrong way; (2) ckpt-500 is the best 2a
checkpoint so far; (3) the missing piece is CONTENT use, which needs explicit pressure:
e.g. weight decision-step CE, or train with the donor-swap as an auxiliary negative (the
subtask target under a swapped bank must follow the bank), or re-check the dev decision
steps for side leakage (the 0.92-AUC wrist-camera leak was trimmed in v3.4, not proven
absent on 0830/0831). Decide before Stage 2b.

**Side-flip battery (2026-09-02, `scripts/v4_side_flip_eval.py`, commits `55b2117`..`71cc142`).**
User framing: memory matters only at the static waiting/decision frame, so the "use" question
is whether the SUBTASK DECISION there follows the memory content. The battery scores each
decision step with the true string and the single-token side-swapped string
(`▁left`=2731 / `▁right`=1833) under normal / reset / donor banks:
D = log p(true) - log p(swapped); headline = donor FLIP RATE on mismatched-side pairs.
Facts established while building it (dev split, 48 windows, `v4/diagnostics` decode):
every window's first decision step carries `wait; target bin is <side>` (27 left / 21
right); windows hold 2-3 decision steps (the last waiting frames plus the first
`open <side> bin` label); the shifted CE label names the side one step BEFORE the waiting
observation begins (44/48). The first battery version scored only windows with exactly one
decision step and so reported 9/48 -- those 9 (ckpt-500, ws-18): normal side accuracy 1.000
(+13.6 nats), reset 0.333 (-1.0 nats), donor mismatched flip rate 0.75 (3/4, -4.3 nats),
donor matched 0.6. The per-step version (all decision steps + first-step view) is what
runs from here; results are appended below when they land.

**Launch decision rule (user instruction 2026-09-02 00:23: after the tests, resume/launch
training on the 1-H100 job whatever the result; modifications allowed if documented; a
fresh run is acceptable).** Single-H100 recipe = `v4/diagnostics/run_train_1gpu.sh <config>
<exp>`: global batch 4 as micro-batch 2 x 2 accumulation (byte-identical objective to r3's
2x2 FSDP), `--resume` automatically when the experiment directory already holds a
checkpoint (the `iris` partition preempts; every relaunch resumes from the last 250-step
save). Rule:
* If the per-step side-flip shows the decision follows the content at full sample
  (mismatched-pair flip rate >= 0.5 at ckpt-500 or ckpt-999), Stage 2a has done its job
  (commit -> retain -> read -> use, with oracle writes) and the next run is **Stage 2b**:
  `pi05_yam_mem_v4_stage2b` (commit `c849c3c`; differs from 2a in exactly
  `memory_fact_oracle_writes=False`, verified by config diff), fresh from pi05_base + the
  Stage-1 fact head, exp `v4_stage2b_20260902_r1`. The 2a/2b gap on both batteries is the
  perception error of the predicted write path.
* Otherwise the next run is a **resume of 2a r3 from ckpt-999** with `--num-train-steps
  2000` (same recipe), because the small-sample flip result says the content path exists
  and just needs more training, and the CE-ratio erosion is then a dilution artifact of
  side-free later decision steps rather than a loss of content use.
Either way the run is launched by the same script and documented here with its exp name,
commit, and the exact command.

**Side-flip v2 result (2026-09-02 01:26, H100 job 17192955, commit `5c19e4c`,
`v4/diagnostics/side_flip_2a_r3_{500,999}_v2/side_flip_eval.json`).** The v1 donor pairing
was wrong for this task: the bank stores OBJECT facts (slot 0 banana, slot 1 grey pepper
box; always on opposite sides in 0830/0831), while the decision names the TARGET side of the
PROMPTED object. Under a donor bank the content-consistent answer is therefore the donor's
fact for the OWN prompted object, which is often the opposite of the donor's target side;
v1's "matched" pairs were mostly content-mismatched, which is why they scored at chance.
With the corrected pairing (all 230 decision steps / first decision step per window):
* normal: names the true side 1.000 / 1.000 (margins +13 nats, both checkpoints);
* reset: 0.60 / 0.44 (ckpt-500), 0.57 / 0.27 (ckpt-999) -- without memory the side is unknown;
* donor implies the other side: flips to it 0.85 / **1.00** (500), 0.84 / **1.00** (999);
* donor implies the same side: stays correct 1.00 / 1.00;
* follows-content rate 0.93 / **1.00** (500), 0.92 / **1.00** (999).
**Verdict: Stage 2a PASSES its purpose.** With oracle writes, the policy's decision at the
static waiting frame is a function of the memory content combined with the prompt, at
near-certainty, and collapses without the memory. The CE-ratio gates (`stage2_eval.json`)
and the v1 pairing under-measured use: the CE ratios average every causal token of every
decision step (FAST action tokens included) and the v1 pairing scored correct
memory-following as errors. The 500 -> 999 "erosion" seen in the CE ratios does not appear
in the content-following rate (0.93 -> 0.92, first-step 1.00 -> 1.00). Gate proposal for
the later stages: replace `reset_decision_ce_ratio` / `donor_decision_ce_ratio` with
`first-step follows-content >= 0.9` and `first-step reset side accuracy <= 0.6`.

**Stage 2b launched 2026-09-02 01:28 PDT** (rule branch 1): job 17192955 (1 H100,
`iris` partition, preemptible), exp `v4_stage2b_20260902_r1`, config
`pi05_yam_mem_v4_stage2b` (commit `c849c3c`: differs from 2a only in
`memory_fact_oracle_writes=False`; the frozen Stage-1 fact head writes its confidence-gated
prediction), fresh from pi05_base + Stage-1 head, code at `5c19e4c`. Exact command:
`v4/diagnostics/run_train_1gpu.sh pi05_yam_mem_v4_stage2b v4_stage2b_20260902_r1` =
`cluster_v4/train.sh pi05_yam_mem_v4_stage2b --exp-name v4_stage2b_20260902_r1 --batch-size 4
--gradient-accumulation-steps 2 --fsdp-devices 1 --no-wandb-enabled` (global batch 4 as 2x2
accumulation; the launcher adds `--resume` automatically after a preemption). Logs:
`v4/diagnostics/train_v4_stage2b_20260902_r1{.log,_status.log}`. Expected ~20 s/update,
1000 updates ~5.5 h. Battery plan for 2b: `v4_stage2_eval.py` + `v4_side_flip_eval.py` on
ckpt 500/999; the 2a/2b gap = perception error of the predicted write path (watch
`v4_sem_commit_count` / `v4_sem_write_eligible_count` in the log for the write rate).

Next after 2a's battery: Stage 2b (predicted writes replace the oracle: flip
`memory_fact_oracle_writes=False`, unfreeze nothing else) -- the 2a/2b gap is the perception
error; then dual-bank inference (`sample_with_memory`) for sampled-action interventions, then
Stage 4 (both banks).

Historical (superseded) next-steps:
1. (done) Confirm at ckpt-1000, freeze the Stage-1 artifact.
2. (superseded) Run the battery on the trained checkpoints:
   `.venv/bin/python scripts/v4_stage1_eval.py --params <ckpt>/params --output-dir v4/diagnostics/stage1_eval_<step>`
   (single GPU; wait for training to finish or use another machine).
2. Decision point on the battery gates. Only on full PASS: per-bank injection calibration,
   then Stage 2a (oracle semantic writes) per §5/§6.
3. Still open: dual-bank `sample_with_memory` (raises NotImplementedError until Stage 2),
   cluster_v4 gate-pipeline clone (§6), per-bank run-identity record for calibrated pilots.
