# Changelog

What changed in the code, data formats and experiment tooling, newest first. Rationale goes in `design/decisions.md`; the experimental design goes in `DOE.md`.

**Versions are `MAJOR.MINOR.PATCH`** with a two-digit patch (`0.1.00`). `pyproject.toml` holds the same number in PEP 440 form (`0.1.0`), and every `run.json` records it as `amon_version`, so any result traces back to an entry here.

- **MAJOR:** breaks comparability with earlier results (episode format, label or metric definitions, the condition set). Earlier runs must be re-run before comparing.
- **MINOR:** a new capability, schedule, perturbation, monitor or sweep factor. Earlier results stay comparable.
- **PATCH:** fixes, refactors, docs and tooling.

Each entry is 3–6 bullets and names its date.

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
