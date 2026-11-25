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

## 5. Episode counts (to be resized after the pilot)

| Task | Split / perturbation | Episodes | Use |
|---|---|---|---|
| transfer_cube | none / train | 500 | world-model training |
| transfer_cube | none / calib | 300 | τ, η, σ, F̂ (successful ones only) |
| transfer_cube | none / test | 300 | FAR (successful), natural-failure recall |
| transfer_cube | knock, slip, nudge / test | 200 each | recall and delay |
| insertion | none / calib, test | ≈750 / 300 | Q4 in decisions.md |
| insertion | peg_misalign, knock / test | 200 each | |

The rows sum to 1,700 transfer-cube and 1,450 insertion episodes. Measured in M1 with three processes on one laptop GPU, shadow queries on and without writing frames: about 5.8 s per transfer_cube episode and 9.5 s per insertion episode of wall time, so about 2.7 h and 3.8 h for these counts. Perturbed episodes run the full 400 steps more often, and frame storage adds time; the pilot re-measures both.

## 6. Responses and analysis plan

- Responses: spec §11.
- Primary comparisons:
  - H1: ours vs fixed-k, ours vs random, at each B.
  - H2: ours vs care, ours vs boundary†, at their realized B, on `mid_subtask`.
  - H3: measured vs nominal FAR for each calibration.
- Paired bootstrap (10k) over episodes for differences; Wilson intervals for proportions. No multiple-comparison correction for the pre-specified primary comparisons; all other comparisons are labeled exploratory.

## 7. Executed runs

| Date | Run dir | Version | Spec | Hardware | Notes |
|---|---|---|---|---|---|
| Nov 25, 2026 | `runs/baseline_20261125-103116` | 0.2.00 (uncommitted tree on dc666ed) | `sweeps/baseline.yaml`, seeds 1000-1199, shadow queries on | RTX 4070 Laptop; rendering D3D12 AMD 780M (WSL2) | M1. transfer_cube 161/200 = 80.5% [74.5, 85.4]; published on the same seeds 167/200 = 83.5% (discordant 2 vs 8, exact McNemar p = 0.109); card 83.0%. insertion 40/200 = 20.0% [15.0, 26.1]; card 20.6%. 0 failed episodes. |
