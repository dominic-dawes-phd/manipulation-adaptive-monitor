# Changelog

What changed in the code, data formats and experiment tooling, newest first. Rationale goes in `design/decisions.md`; the experimental design goes in `DOE.md`.

**Versions are `MAJOR.MINOR.PATCH`** with a two-digit patch (`0.1.00`). `pyproject.toml` holds the same number in PEP 440 form (`0.1.0`), and every `run.json` records it as `amon_version`, so any result traces back to an entry here.

- **MAJOR:** breaks comparability with earlier results (episode format, label or metric definitions, the condition set). Earlier runs must be re-run before comparing.
- **MINOR:** a new capability, schedule, perturbation, monitor or sweep factor. Earlier results stay comparable.
- **PATCH:** fixes, refactors, docs and tooling.

Each entry is 3–6 bullets and names its date.

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
