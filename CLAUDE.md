# CLAUDE.md: standing rules

Project: budgeted scheduling of expensive failure monitors for frozen ACT policies in gym-aloha. Start with `HANDOFF.md`. Definitions are in `docs/design/spec.md`; they are the source of truth for every metric and schedule.

## Hard rules

1. **The monitor is passive.** Nothing a monitor or schedule outputs may change the rollout. This is what makes offline replay valid.
2. **Causality.** Schedules and triggers may use only data available at step t. Sim labels (phase, onset, contacts) are for evaluation only, except the privileged boundary schedule (decision D-06), which must stay labeled as privileged everywhere it appears.
3. **Shadow queries never change execution.** A test enforces identical executed actions with logging on and off.
4. **Splits are by episode and seed range.** Calibrate and tune trigger τ only on `calib`. Train the world model only on `train`. Report only on `test`. Seed ranges are fixed in `configs/base.yaml`.
5. **False-alarm rates are measured, never assumed.** Every table that shows a nominal α also shows the measured per-check and per-episode false-alarm rate.
6. **Equal-budget comparisons.** Any claim that one schedule beats another states the budget (checks per episode) at which it holds.
7. **No mixing hardware or software versions within a comparison** (laptop vs cluster, LeRobot versions, VLM quantizations). `run.json` records them.
8. **Global parameters live in `configs/base.yaml`**, loaded by `src/amon/settings.py`. No magic numbers in modules.
9. **Never edit a finished run directory.** Re-run instead.

## After every implementation pass

- Add a `docs/CHANGELOG.md` entry: 3–6 bullets with a date. Bump the version in `pyproject.toml`: MAJOR.MINOR.PATCH, two-digit patch in the changelog (`0.2.01`), PEP 440 in pyproject (`0.2.1`).
  - MAJOR: breaks comparability with earlier results (metric definitions, episode format, labels).
  - MINOR: a new capability, schedule, perturbation or monitor; earlier results stay comparable.
  - PATCH: fixes, refactors, docs, tooling.
- Update `docs/DOE.md` whenever factors, levels, responses or the analysis plan change. Log executed runs in DOE §7.
- Add a dated entry to `docs/design/decisions.md` for any design choice, with what was rejected and why.
- Run `scripts/check.sh` (pytest + ruff). Do not report a milestone done with failing checks.

## Code style (the user's review standards)

- Google Python style; ruff clean; newline at end of file; all imports present.
- Type hints everywhere; no `Any` where a specific type is possible.
- Legibility first. No proxy variables (e.g., setting a `failure_message` and branching on it later). Handle each `except` fully in place (log and continue), then proceed knowing the result is safe. No nested try/except.
- No lambdas, no nested closures, never a nested class definition. Few helper functions, each with a plain, descriptive name.
- Don't pass around dict fields nobody reads.
- Logging: concise, no excessive debug output.

## Run directories

`runs/<name>_<YYYYMMDD-HHMMSS>/` holds `run.json` (version, git commit and dirty flag, host, GPU, package versions, model hashes, spec copy), `episodes/`, `scores/`, `eval/`, `figures/`, `logs/`. Smoke runs use name `smoke`, pilots `pilot`, main sweeps `main` or a descriptive name.
