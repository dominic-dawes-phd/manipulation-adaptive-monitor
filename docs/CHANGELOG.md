# Changelog

What changed in the code, data formats and experiment tooling, newest first. Rationale goes in `design/decisions.md`; the experimental design goes in `DOE.md`.

**Versions are `MAJOR.MINOR.PATCH`** with a two-digit patch (`0.1.00`). `pyproject.toml` holds the same number in PEP 440 form (`0.1.0`), and every `run.json` records it as `amon_version`, so any result traces back to an entry here.

- **MAJOR:** breaks comparability with earlier results (episode format, label or metric definitions, the condition set). Earlier runs must be re-run before comparing.
- **MINOR:** a new capability, schedule, perturbation, monitor or sweep factor. Earlier results stay comparable.
- **PATCH:** fixes, refactors, docs and tooling.

Each entry is 3–6 bullets and names its date.

## [2.0.00] Rendering without multisampling, for bit-identical frames (Nov 26, 2026)
- **Breaks comparability:** `render.offsamples: 0` (new key) turns off MuJoCo's 4× multisample anti-aliasing. On NVIDIA EGL it made repeated renders of one state differ by a grey level, so seeds did not replay (D-29). Every earlier episode was rendered with multisampling.
- `AlohaSim(task, render)` now takes the render settings and applies `offsamples` before the first render. All callers are updated.
- New test `test_rendering_the_same_state_is_deterministic` renders one state 9 times.
- `server/chip_job.sbatch` runs on RTX 6000 nodes only (the L40S nodes cannot render), accepts any sweep name as a step, and sets `PACK_HOST` so `pack.sh` prints the login host. `run_local.sh` runs `baseline_report` once. `docs/chip.md` is the chip runbook.

## [1.3.01] UMBC chip batch job (Nov 26, 2026)
- `server/chip_job.sbatch`: one Slurm job on chip-gpu (4 GPUs, one node) that runs setup, check, smoke and pilot in order, with `STEPS` to choose them. It keeps caches in the group space (home quota is 500 MB) and refuses mixed GPU models.
- Chip's GPUs are not RTX 3070s, so results from this job are a separate hardware group and cannot be pooled with the 3070 server (D-09). The decision is recorded in `design/decisions.md` once the hardware is confirmed.
- Tooling only: no change to metrics, episode format or labels, so earlier results stay comparable.

## [1.3.00] Monitor stages and the 3070 server runbook (Nov 26, 2026)
- `experiments.run` stages `train_wm`, `score_wm` and `score_vlm` (`monitors/stages.py`). Scores go to `scores/<monitor>/<episode_id>.npz`, plus a per-shard meta with model hashes and the latency measured on that GPU. The stages resume, and `--endpoint` is wired through.
- World-model input is area-resized to `monitors.wm.input_size` (224 px), so 448 px frames can be stored for the VLM (D-26). New keys: `monitors.vlm.latency_sample`, `monitors.wm.latency_repeats`, `server.*`.
- `server/setup.sh`: no-sudo install of both environments and all weights, chosen by driver version. `requirements/vlm_cu12.txt` adds vLLM 0.11.0 on CUDA 12.8 for drivers below 580 (D-28). `scripts/fetch_models.py` replaces `fetch_models.sh`.
- `server/check.sh`: test suite, cross-GPU bit-identity (`scripts/compare_runs.py`, `sweeps/server_check.yaml`) and VLM fit with fallbacks. Sweeps refuse to run until it passes for the current commit.
- `cluster/run_local.sh`: vLLM via `scripts/serve_vlm.sh` and `scripts/vlm_lib.sh`, with a health timeout and cleanup on Ctrl+C. `RUN_DIR` continues an interrupted run. `server/pack.sh` adds checksums. `docs/server.md` is the runbook.
- Smoke and pilot specs: 448 px frames and the four implemented stages; `evaluate` and `plot` follow M5 (D-27).

## [1.2.00] M4 world-model monitor (Nov 26, 2026)
- `monitors/wm.py`: frozen DINOv2 ViT-S/14 from a pinned torch.hub commit (mean patch token), residual MLP predictor of z_t from z_{t-k-L+1..t-k} and a_{t-k..t-1}, cosine-residual training with an episode-level validation split, causal score 1 - cos(ẑ_t, z_t), save/load.
- `WmSettings`; `base.yaml` `monitors.wm` gains `encoder_repo`, `encode_batch`, `hidden_dim`, `val_fraction`, `seed`; encoder default switched to `dinov2_vits14` because DINOv3 is gated (D-25).
- `scripts/wm_dev_check.py` and `configs/sweeps/wm_dev.yaml`: dev-seed acceptance check (train on 100 unperturbed dev episodes, score the review episodes against their counterparts).
- Tests on synthetic latents: window indexing, learning, causality of scores, save/load round trip.

## [1.1.00] M3 VLM judge and sanity gate tooling (Nov 26, 2026)
- `monitors/vlm.py`: client for a vLLM OpenAI-compatible server; score = P(No) / (P(Yes) + P(No)) from first-token logprobs, case and whitespace variants pooled; cache by (model, prompt, frame) in JSON lines; identical frames sent once; failed requests logged and scored NaN, never cached.
- `scripts/serve_vlm.sh` (separate `~/amon/venv-vlm`, vLLM 0.34.0; `requirements/vlm_freeze.txt`), `scripts/vlm_sanity.py` (paired frame set from dev review episodes and their unperturbed counterparts, blind contact sheet, AUROC and latency), `eval/stats.py: auroc`.
- `base.yaml`: `monitors.vlm` serving limits for 8 GB GPUs (`max_model_len` 1024, `max_num_seqs` 4, `gpu_memory_utilization`), `top_logprobs`, `concurrency`, `request_timeout_s`; `VlmSettings`.
- `configs/sweeps/vlm_gate_448.yaml`: the review seeds re-rendered with 448 px frames for the gate.
- Preliminary gate result (sim-proposed labels): no model reaches AUROC 0.75 (D-24).

## [1.0.00] Perturbation-aware failure labels, Q7 and Q8 (Nov 26, 2026)
- MAJOR: `label_failure` takes the perturbation and `perturb_step`. An event visible before the first perturbed frame gives `failed_before_perturbation` (Q8); no event within `labels.perturbation_event_window` = 50 steps of the perturbation gives `failure_type` = the perturbation name with onset `perturb_step + 1` (Q7). Episodes stored by earlier versions carry the old labels; regenerate before comparing.
- On the M2 review episodes, the new rules relabel 11 of 50: 5 `unlabeled` and 2 late drops of peg_misalign, 1 late `handover_miss` of a knock and 2 late labels of nudges become perturbation-attributed, and insertion knock seed 2007 becomes `failed_before_perturbation`.
- Spec sections 2 and 4 updated; label test covers both rules and the unperturbed path.

## [0.4.00] Trigger features and the dev signal check (Nov 26, 2026)
- `trigger/features.py`: spec section 8 disagreement per action group (`group_disagreement`), `normalization_sigma` and edge-triggered `gripper_flips`, pure and causal; `TriggerSettings` in `settings.py`, new key `trigger.disagreement_stride`.
- `scripts/trigger_signal_check.py`: d(t) after each perturbation against the same dev seed unperturbed (paired, identical up to the perturbation step); `configs/sweeps/review_nominal.yaml` rolls out those counterparts.
- Tests on synthetic chunk streams: zero disagreement for consistent chunks, a plan change shows once in its group, sigma and window, flip edges and refractory period.
- Q7 and Q8 accepted (decisions.md); implemented in the next version.

## [0.3.00] M2: labels, perturbations, episode store, 3070 server runner (Nov 25, 2026)
- `sim/state.py`, `sim/labels.py`, `sim/perturb.py`: privileged state, failure labels (D-16) and the four perturbations (D-18, D-19); `rollout/store.py` writes the spec section 2 format with frame-aligned arrays and `held`/`contact` (D-17); `rollout/sweep.py` and the `rollout` stage expand sweep cells into seeded episodes, resumable and sharded.
- Physics substep hook in `AlohaSim.step`, used by the knock to stop pushing when the grip breaks; top camera rendered alone (bit-identical to M1, about 25% faster).
- `scripts/calibrate_perturbations.py` (strength sweeps by exact replay of baseline actions), `scripts/verify_against_baseline.py`, `experiments/viz.py` (review reels and contact sheets), `configs/sweeps/review.yaml`.
- `base.yaml`: `hold_margin_m`, `drop_debounce_steps`, `handover_window` 200, knock 5 N with release cutoff, nudge 5 cm at a 6 cm radius, `dev` split, `calib` widened to 10000-10999.
- Sweep specs accept `overrides:` (dotted base.yaml keys, unknown keys rejected).
- `run_3070_server.sh` and `requirements/server_3070.txt` (hash-locked, identical versions to `requirements/laptop_4070.txt`); `run_local.sh` can put several processes on one GPU and stops after a failing stage.

## [0.2.00] M0 and M1: environment, policy adapter, baseline (Nov 25, 2026)
- Pinned the simulation stack in `pyproject.toml` (D-11) and added the package skeleton: `settings.py`, `version.py`, `sim/env.py`, `policy/act.py`, `rollout/{runner,manifest,baseline}.py`, `eval/stats.py` (Wilson, exact McNemar).
- `configs/base.yaml`: names, units, episode length and success reward verified against gym-aloha 0.1.1 and the checkpoints; checkpoint revisions pinned; per-object rest heights and `splits.baseline` added.
- Shadow queries through `predict_action_chunk`; tests show bit-identical execution with them on and off on both tasks, and that the logged chunk equals the executed chunk at each re-query.
- `experiments/run.py` with the stages `baseline` and `baseline_report`; `configs/sweeps/baseline.yaml`. Other stages raise an error until their milestone.
- Measured nominal success (200 episodes per task): transfer_cube 80.5% [74.5, 85.4], insertion 20.0% [15.0, 26.1]; both intervals contain the model-card value. Details in `DOE.md` section 7.
- Findings recorded as D-11 to D-15 and Q2, Q5 and Q6, including the open-loop executor (re-queries only every 100 steps).

## [0.1.00] START (Nov 25, 2026)
- Scaffold handed off: `HANDOFF.md`, `CLAUDE.md`, `README.md`, `configs/base.yaml`, sweep specs (smoke, pilot, main), VLM prompts.
- Design docs: `docs/design/spec.md` (episode format, trigger, schedules, calibration, metrics), `docs/design/decisions.md` (D-01 to D-10, open questions Q1–Q4), `docs/DOE.md`.
- Paper skeleton: `docs/paper/main.tex`, `references.bib` (2026 entries marked VERIFY), `related_works.md`.
- Cluster tooling templates: `cluster/` (Slurm array via `job.sbatch`, `run_local.sh` for a multi-GPU box, collect and pack scripts).
- No code beyond the scaffold; no results.
