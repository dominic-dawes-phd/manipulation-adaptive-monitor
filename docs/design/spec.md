# Technical specification

This is the source of truth for data formats, schedules and metrics. Symbols: T = episode length in control steps (50 Hz); H = ACT chunk size; t indexes control steps; Γ = a monitor's check grid.

## 1. Tasks

| Task | Env id (verify) | Success | Phase stages (verify in source) |
|---|---|---|---|
| transfer_cube (primary) | `gym_aloha/AlohaTransferCube-v0` | final reward stage = max | 1 right gripper touches cube; 2 cube lifted by right; 3 left gripper touches cube; 4 transferred (held by left, off table) |
| insertion (secondary) | `gym_aloha/AlohaInsertion-v0` | final reward stage = max | 1 both grippers touch peg/socket; 2 both lifted; 3 peg touches socket; 4 peg inserted |

Phase p_t is the highest reward stage reached so far, with regressions tracked separately. A *subtask boundary* is a step where p_t increases.

## 2. Episode format

One directory per episode: `episodes/<task>/<perturbation>/<split>/ep_<seed:06d>/`.

`data.npz` (compressed):

| Key | Shape / dtype | Content |
|---|---|---|
| `frames` | [T, 224, 224, 3] uint8 | top camera, resized from native (resize method in run.json) |
| `agent_pos` | [T, 14] float32 | joint state |
| `action_exec` | [T, 14] float32 | executed action |
| `chunks` | [T, H, 14] float16 | shadow-query chunk at each step; `chunks[t, j]` predicts the action for step t+j |
| `act_latent` | [T, D] float16 | optional ACT encoder feature (probe / RND) |
| `reward` | [T] int8 | env reward stage |
| `phase` | [T] int8 | p_t = max(reward[:t]) |
| `contact_left`, `contact_right` | [T] bool | gripper–object contact, any object |
| `contact`, `held` | [T, K, 2] bool | per object and arm (left, right); held as in §4 |
| `obj_pos` | [T, K, 3] float32 | positions of task objects (cube, or peg and socket) |

Alignment (D-17): every array except `reward` describes frame t, before action t is applied; `reward[t]` is the outcome of action t.

`meta.json`: episode id, task, perturbation name and sampled parameters, `perturb_step` (−1 if none), `success`, `onset_step` (−1 if none or unlabeled), `failure_type` (`none`, `drop`, `grasp_miss`, `handover_miss`, `insert_miss`, `unlabeled`), `boundary_steps` (list), seed, split, `sim_version`. Native-resolution video is optional (`video.mp4`, for viz only).

Scores live in `scores/<monitor>/<episode_id>.npz`: `grid_steps` [N] int, `score` [N] float32, plus `meta.json` with model hash, prompt hash, and measured `latency_ms` (mean and p95).

## 3. Shadow queries

At every step t, the policy is also queried for a full chunk A_t ∈ ℝ^{H×14} from the current observation. This query is logged only. Execution follows the checkpoint's own eval configuration (chunk queue or temporal ensembling) through a separate code path. Executed actions must be bit-identical with shadow queries on or off (test required).

## 4. Failure onset labels

A failure event is detected from sim state. An object is *held* by an arm when either of its fingers touches it and its center is more than h_hold above its resting height (D-16).
- **drop:** the object is held by some arm at t−1 and by no arm for the next d steps (`drop_debounce_steps`). Onset = t.
- **grasp_miss:** a gripper close command (gripper channel crosses its close threshold) with no contact within W_g steps. Onset = the close-command step.
- **handover_miss** (transfer_cube): stage 3 or higher is not reached within W_h steps of the first stage 2 (a good handover skips stage 3). Onset = the first late step. A drop in the handover window is labeled `drop`.
- **insert_miss** (insertion): both lifted, but stage 4 is not reached within W_i steps of stage 3. Onset = the stage-3 step (weak label; reported separately).

For a failed episode, `onset_step` = the earliest event after which success is never reached. Failed episodes with no detected event get `failure_type = unlabeled`: they count toward recall but have no delay.

Perturbed episodes that still succeed go in the **recovered** bucket. They are excluded from both recall and false-alarm rate, and reported on their own.

All thresholds (h_hold, d, W_g, W_h, W_i) are in `configs/base.yaml`. Labels must be checked against videos (HANDOFF M2).

## 5. Perturbations

Each perturbation fires once per episode at a step sampled within a phase window: the first step the window condition holds, plus U{0, D} steps.

| Name | Window condition | Mechanism | Defaults |
|---|---|---|---|
| knock | object held by right gripper, phase 2 | horizontal force on the object body via `xfrc_applied`, random direction, stopped at the first physics substep pair without finger contact or after n_k steps (D-18) | 5 N, n_k = 10 |
| slip | phase 2, object held and rising | right finger geom friction scaled to ε and geom priority raised for n_s steps, then restored (D-19) | ε = 0.01, n_s = 20 |
| nudge | phase 0, no contact, right finger midpoint within r_n of the object | teleport the object Δ horizontally, resampling the direction if it creates a contact | Δ = 5 cm, r_n = 6 cm |
| peg_misalign (insertion) | peg held by right gripper, phase 2 | rotate the peg about the finger closing axis, around the pinch point | 10–20°, random sign |
| none | — | — | — |

Delays D (`delay_max`): knock 30, slip 20, nudge 0, peg_misalign 20. Parameters were set by `scripts/calibrate_perturbations.py` on baseline seeds and checked on `dev` review videos. A perturbation whose window never opens leaves `perturb_step = −1`; such episodes are reported as "not fired" and excluded from perturbation metrics.

## 6. Expensive monitors

Each monitor m has a grid Γ_m (steps where it can run), a score s_m(t) for t ∈ Γ_m, and a measured latency ℓ_m (ms), converted to steps as λ_m = ceil(ℓ_m / 20 ms).

### 6.1 VLM judge
- Model: `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` via vLLM (fallback 3B-AWQ).
- Input: the frame at t (capped pixel count, set in base.yaml) and the task prompt from `configs/prompts/`.
- Score: s = exp(lp_No) / (exp(lp_Yes) + exp(lp_No)), from first-token logprobs. Temperature 0.
- Default grid: every 5 steps (10 Hz).

### 6.2 World-model monitor
- Latent z_t = pooled encoder features of frame t.
- Predictor f(z_{t−L+1..t}, a_{t..t+k−1}) → ẑ_{t+k}, trained on `train`.
- Score at t: s(t) = 1 − cos(ẑ_t, z_t), where ẑ_t is predicted from data up to t−k. It is therefore a residual on the *arriving* frame, and it is causal at t.
- Default L = 4, k = 8, grid = every step.

## 7. Splits

Unperturbed episodes are split by seed range into `train` (world-model training), `calib` (trigger τ, alarm thresholds, normalization statistics) and `test`. Perturbed episodes are `test` only. Ranges and counts are in `configs/base.yaml` and `docs/DOE.md` §5.

## 8. Trigger features (computed at every step)

**Normalization σ.** Per-dimension standard deviation of `action_exec` over `calib` successful episodes.

**Disagreement.** With query stride s (default 1) and near-term window M (default 20), for groups g ∈ {L_arm (0:6), L_grip (6), R_arm (7:13), R_grip (13)} (verify the index order):

  e_g(t) = (1/M') Σ_{j=0}^{M'−1} ‖ (A_t[j, g] − A_{t−s}[j+s, g]) / σ_g ‖₂ ,   M' = min(M, H − s)

  d(t) = max_g e_g(t). Per-group values are kept for ablations.

**Gripper flip.** g(t) = 1 if, for either arm, the predicted gripper channel in A_t[0..h_g] crosses the open/close midpoint relative to the current executed gripper state, *and* this predicted flip event was not already flagged (edge-triggered; the event is identified by arm and direction, with refractory period r_g).

**Progress lag (optional).** A linear probe on `act_latent` predicts normalized progress π̂(t). Lag ρ(t) = π_nom(t) − π̂(t), where π_nom is the mean progress at step t on calib.

**Percentile normalization.** d̃(t) = F̂_calib(d(t)), and likewise ρ̃. Each is the empirical CDF value of the raw feature over all calib successful steps, in [0, 1].

## 9. Schedules

Each schedule maps an episode's causal stream to a set of requested steps R. Each request r is served at the next grid point γ(r) = min{γ ∈ Γ_m : γ ≥ r}. Duplicate grid points collapse into one check. C = the set of served checks.

| Schedule | Requested steps |
|---|---|
| always_on | all of Γ_m |
| fixed_k | every k-th grid point, with k chosen to meet budget B |
| random | B grid points sampled uniformly without replacement, seeded; results averaged over 20 draws |
| boundary (privileged, D-06) | each subtask boundary from sim labels, plus the final step; variant `boundary+timeout` adds checks when there has been none in the last G steps |
| care | each commanded gripper close and release event (either arm) in `action_exec`, plus every 50 steps (verify against the CARE paper) |
| contactguard | each commanded open→close event (either arm) |
| ours | fire at t if [ d̃(t) ≥ τ OR g(t) = 1 OR ρ̃(t) ≥ τ_ρ OR t − t_last ≥ G ] AND t − t_last ≥ r_min |

Here t_last is the last fire. Default G = 100 steps (2 s), r_min = 5. Variant `ours_weighted`: fire when w_d d̃ + w_g g + w_ρ ρ̃ ≥ τ (ablation).

**Budget matching.** τ is found by bisection on calib episodes so that mean |C| per episode = B. B is expressed as a fraction of |Γ_m| (always-on). The realized budget on test is always reported.

## 10. Alarm thresholds (calibration)

A check at grid step γ alarms if s_m(γ) ≥ η. Thresholds come from calib successful unperturbed episodes, at level α (default 0.05 and 0.10), using the finite-sample split-conformal quantile: q = the ⌈(n+1)(1−α)⌉-th smallest of n values.

| Name | Values the quantile is taken over | Intended control |
|---|---|---|
| `naive_check` | all always-on grid scores | per-check false alarms on random frames |
| `episode_max` | per-episode max over all grid scores | per-episode false alarms under always-on |
| `stream_check` (ours) | scores at the frames schedule S selects | per-check false alarms for S |
| `stream_episode` (ours) | per-episode max over the frames S selects | per-episode false alarms for S |

The trigger τ depends only on action features, never on monitor scores, so selection is computed first and η second (the fixed point). Exchangeability caveat: see 2610.19438. Hence rule 5 in CLAUDE.md.

## 11. Metrics

Let C be the checks, η the threshold, and o the onset.

- **Detected:** some c ∈ C with c ≥ o and s(c) ≥ η, at or before T−1. Detection step t* = the smallest such c.
- **Recall:** detected failure episodes / failure episodes, by perturbation and failure type.
- **Delay (raw):** t* − o, in steps. **Delay (effective):** t* − o + λ_m. Report median, IQR and CDF.
- **Early alarm rate:** failure episodes with an alarm at c < o. Reported separately; these are not counted as detections.
- **Per-episode FAR:** fraction of successful unperturbed test episodes with ≥ 1 alarm.
- **Per-check FAR:** alarms / checks on successful unperturbed test episodes.
- **Budget:** mean |C| per episode, as a fraction of |Γ_m|; and GPU-ms per episode = |C| · ℓ_m.
- **Onset location:** each failure is tagged `mid_subtask` if more than W_b steps (default 25) from every boundary and every commanded gripper event; otherwise `near_event`. H2 is tested on the `mid_subtask` subset.

**Statistics.** All schedules are evaluated on the same episodes, so comparisons are paired. Report 95% CIs from a paired bootstrap over episodes (10,000 resamples) of the difference to the comparison arm, and Wilson intervals for single proportions.

## 12. Printed summary

One block per (task, monitor, budget): one row per schedule with recall, median raw/effective delay, per-episode FAR and per-check FAR (measured vs α), checks per episode, and recall on `mid_subtask` only.

## 13. Figures (PNG + PDF + CSV twin)

1. **Hero:** recall and median delay vs budget, one curve per schedule (fixed-count schedules as points), split into `mid_subtask` and `near_event`.
2. Delay CDF at matched budget, by perturbation (knock, slip, nudge).
3. Measured vs nominal false-alarm rate (per check and per episode) for the four calibrations of §10.
4. Trigger ablation bars (component alone, no timeout, weighted vs OR).
5. Episode timeline: d̃(t), g(t), each schedule's checks, alarms and onset, synced to video frames (also exported as video by `experiments/viz.py`).
