# Design of experiments

Keep this file current. Update it whenever factors, levels, responses or the analysis plan change, and log every executed run in §7. Formal definitions: `design/spec.md`.

## 1. Question

At an equal budget of expensive monitor checks per episode, does a cheap, policy-derived trigger detect failures of a frozen action-chunked policy earlier and more often than boundary-only, event-based (CARE, ContactGuard), fixed-rate and random schedules? Do alarm thresholds keep their nominal false-alarm rate when a trigger chooses the frames?

## 2. Hypotheses

- **H1 (efficiency):** at B = 20–30% of always-on checks, ours reaches recall within 5 points of always-on (target, not result), with lower median delay than fixed-k and random at the same B.
  - Supported if: the paired difference in delay (ours − fixed-k) has a 95% CI below 0.
  - Refuted if: the CI includes 0, or recall falls more than 10 points below always-on.
- **H2 (mid-subtask):** on `mid_subtask` failures (knock, slip), ours has lower median delay and higher recall than CARE and boundary† at their realized budgets. On nudge (`near_event`), the difference is small.
  - Supported if: the paired CIs exclude 0 in ours' favor on `mid_subtask`.
  - Refuted if: CARE or boundary† match or beat ours there.
- **H3 (calibration):** `naive_check` thresholds give measured per-check FAR above α on trigger-selected frames, and per-episode FAR rises with checks per episode. `stream_check` and `stream_episode` give measured FAR within the Wilson CI of α.
  - Refuted if: naive thresholds are already at α (still a reportable finding).

## 3. Key definitions

- **Check:** one expensive-monitor evaluation at a grid step.
- **Budget B:** mean checks per episode as a fraction of always-on.
- **Onset:** the sim-labeled failure step.
- **Delay:** check-that-alarms minus onset, with raw and effective (+ latency) versions.
- **mid_subtask:** onset more than 25 steps from any subtask boundary or commanded gripper event.
- **†:** privileged (uses sim labels).

## 4. Factors and levels

| Factor | Levels | Notes |
|---|---|---|
| Task | transfer_cube (primary), insertion (secondary) | |
| Perturbation | none, knock, slip, nudge; insertion: none, peg_misalign, knock | spec §5 |
| Monitor | VLM judge (Qwen2.5-VL-7B-AWQ), world-model residual | VLM may be dropped at the M3 gate |
| Schedule | always_on, fixed_k, random, boundary†, boundary†+timeout, care, contactguard, ours | |
| Budget B | 5, 10, 20, 30, 50% of always-on | parametric schedules only; fixed-count schedules at realized B |
| Calibration | naive_check, episode_max, stream_check, stream_episode | |
| α | 0.05, 0.10 | |
| Trigger ablation | full; disagreement only; gripper only; no timeout; + progress lag; weighted vs OR | |

Schedules, budgets, calibrations and ablations are offline factors: they cost no simulation.

Held fixed, not factors:
- **Hardware:** every reported episode, score and latency comes from the 8x RTX 3070 server (D-22). Its environments come from `server/setup.sh`, and `run_3070_server.sh --check` must pass first (`docs/server.md`).
- **Frame size:** the pilot stores 448 px frames. The VLM sees them at that size; the world model sees them area-resized to 224 px (D-26). The main sweep's size is decided after the M3 gate.

## 5. Episode counts (to be resized after the pilot)

| Task | Split / perturbation | Episodes | Use |
|---|---|---|---|
| transfer_cube | none / train | 500 | world-model training |
| transfer_cube | none / calib | 300 | τ, η, σ, F̂ (successful ones only) |
| transfer_cube | none / test | 300 | FAR (successful), natural-failure recall |
| transfer_cube | knock, slip, nudge / test | 200 each | recall and delay |
| insertion | none / calib, test | ≈750 / 300 | Q4 in decisions.md |
| insertion | peg_misalign, knock / test | 200 each | |

Perturbation yield on the 10 dev review episodes each (M2, `runs/review_20261125-172434`): slip 9 failures, all drops 3-6 steps after the slip; knock (transfer_cube) 8 failures, 7 of them drops 7-9 steps after the push, 2 recovered (replays put the drop rate at 42%, D-18); nudge 7 failures of 8 fired, `grasp_miss` 15-21 steps after the nudge, 2 not fired; insertion knock 7 drops, 1 not fired, 1 natural drop before the knock; peg_misalign 9 failures but 6 unlabeled (Q7). If knock drops stay near 40-70%, 300-400 knock episodes are needed for about 200 labeled drops; the pilot measures the yields on test seeds before the main sweep is resized. Stored episodes take about 4.7 MB each (frames and chunks), about 16 GB for the main sweep.

The rows sum to 1,700 transfer-cube and 1,450 insertion episodes. Measured in M1 with three processes on one laptop GPU, shadow queries on and without writing frames: about 5.8 s per transfer_cube episode and 9.5 s per insertion episode of wall time, so about 2.7 h and 3.8 h for these counts. Perturbed episodes run the full 400 steps more often, and frame storage adds time; the pilot re-measures both.

## 6. Responses and analysis plan

- Responses: spec §11.
- Where it runs: rollouts, world-model training and both scoring stages run on the server; `evaluate` and `plot` replay the packed server results on the laptop (D-27).
- Primary comparisons:
  - H1: ours vs fixed-k, ours vs random, at each B.
  - H2: ours vs care, ours vs boundary†, at their realized B, on `mid_subtask`.
  - H3: measured vs nominal FAR for each calibration.
- Paired bootstrap (10k) over episodes for differences; Wilson intervals for proportions. No multiple-comparison correction for the pre-specified primary comparisons; all other comparisons are labeled exploratory.

## 7. Executed runs

| Date | Run dir | Version | Spec | Hardware | Notes |
|---|---|---|---|---|---|
| Nov 25, 2026 | `runs/baseline_20261125-103116` | 0.2.00 (uncommitted tree on dc666ed) | `sweeps/baseline.yaml`, seeds 1000-1199, shadow queries on | RTX 4070 Laptop; rendering D3D12 AMD 780M (WSL2) | M1. transfer_cube 161/200 = 80.5% [74.5, 85.4]; published on the same seeds 167/200 = 83.5% (discordant 2 vs 8, exact McNemar p = 0.109); card 83.0%. insertion 40/200 = 20.0% [15.0, 26.1]; card 20.6%. 0 failed episodes. |
| Nov 25, 2026 | `runs/perturb_calibration_20261125-165042`, `runs/knock_release_20261125-170035` | 0.2.0 (uncommitted) | `scripts/calibrate_perturbations.py`, 40 replayed baseline successes per setting | laptop | Strength sweeps by exact replay (D-18, D-19). Not reported. |
| Nov 25, 2026 | `runs/review_20261125-170939`, `runs/nudge_dev_*_20261125-1718*` | 0.2.0 (uncommitted) | `sweeps/review.yaml`; nudge variants by `overrides` | laptop | First review set and live nudge check (D-19). Superseded. |
| Nov 25, 2026 | `runs/review_20261125-172434` | 0.3.00 (ed0b7ca, clean) | `sweeps/review.yaml`, dev seeds 2000-2009, video | laptop | M2 review set: 50 episodes, reels in `figures/`. Outcomes identical to the first set. Awaiting the user's visual check. |
| Nov 25, 2026 | `runs/smoke_20261125-172434` | 0.3.00 (ed0b7ca, clean) | `sweeps/smoke.yaml`, stage rollout | laptop | M2 acceptance: 18 episodes written end to end, none failed to run. |
| Nov 26, 2026 | `runs/review_nominal_20261126-125053` | 0.3.0 (dirty: trigger code only, rollout code unchanged) | `sweeps/review_nominal.yaml`, dev seeds 2000-2009, unperturbed | laptop | Counterparts for the trigger check; action sequences identical to the perturbed episodes up to the perturbation in 46/46. Not reported. |
| Nov 26, 2026 | `runs/trigger_check_20261126-130051` | 0.4.00 (ecf9587, clean) | `scripts/trigger_signal_check.py`, review + review_nominal | laptop | Trigger signal check (D-21); same numbers as the dirty-tree run `trigger_check_20261126-125411`, which it supersedes. Dev only, not reported. |
| Nov 26, 2026 | `runs/vlm_sanity_20261126-130910` | 1.0.0 (dirty: M3 code) | `scripts/vlm_sanity.py`, 24 pairs from review + review_nominal at 224 px | laptop; vLLM 0.34.0 | M3 gate (D-24), sim-proposed labels: 7B AWQ AUROC 0.516 (0.644 upscaled to 448), 3B AWQ 0.632. Latency not representative (7B needed CPU offload). Awaiting the user's labels. |
| Nov 26, 2026 | `runs/vlm_gate_448_20261126-133351`, `runs/vlm_sanity_20261126-134125` | 1.0.0 (dirty: M3 code) | `sweeps/vlm_gate_448.yaml`, dev seeds 2000-2009, 448 px frames; 23 pairs | laptop; vLLM 0.34.0 | M3 gate at native 448 px: 7B AWQ 0.709, 3B AWQ 0.352 (sim-proposed labels). Not reported. |
| Nov 26, 2026 | `runs/wm_dev_20261126-135440`, `runs/wm_check_20261126-140702` | 1.1.0 (dirty: M4 code) | `sweeps/wm_dev.yaml` (dev seeds 2000-2109, unperturbed) and `scripts/wm_dev_check.py` | laptop | M4 acceptance (D-25): val loss 0.0077 -> 0.0053; mean score in the 20 steps after onset, perturbed vs same seed unperturbed: knock 0.0090 vs 0.0061 (peak higher 7/7), slip 0.0095 vs 0.0071 (8/9), nudge 0.0050 vs 0.0044 (5/6; the rise comes at the nudge, before the onset). 9.1 ms per check on the laptop GPU. Not reported. |
| Nov 26, 2026 | `runs/server_check_20261126-201412` | 1.2.0 installed (dirty: 1.3 code) | `server/check.sh` on the laptop (`ALLOW_OTHER_GPU=1`, `EGL_PER_GPU=0`, `--gpu-memory-utilization 0.80`) | laptop; vLLM 0.34.0 | Tests of the server scripts, not results. Determinism PASS (same GPU twice). VLM fallback worked: 7B failed twice (no offload on the laptop), 3B loaded, 150-154 ms per request at 448 px. Tests FAIL because of a test-order bug: cuBLAS was initialized before `CUBLAS_WORKSPACE_CONFIG`. Fixed with `tests/conftest.py`. |
| Nov 26, 2026 | `runs/server_check_20261126-202549`, `runs/smoke_20261126-202958` | 1.3.0 (dirty: uncommitted 1.3 code) | `server/check.sh` with `SKIP_VLM=1`, then `run_3070_server.sh` on `sweeps/smoke.yaml` with a 3B override | laptop; vLLM 0.34.0 | Check PASS. Smoke ran all four stages over 30 episodes, every status 0: 12 WM and 22 VLM score files. WM latency 29 ms with two shards sharing the GPU, so `score_wm` now runs one shard per GPU. Packed with `server/pack.sh`: 305 MB. Not reported. |
