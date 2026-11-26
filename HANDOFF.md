# HANDOFF: Adaptive Monitoring Schedule (CMSC 901)

Read this file first, then `CLAUDE.md`. It describes what to build, in what order, and how each milestone is accepted. Formal definitions live in `docs/design/spec.md`. Open choices live in `docs/design/decisions.md`; entries marked **PROPOSED** need the user's confirmation before results are produced with them.

## 1. The project in one paragraph

Action-chunked robot policies are often watched by an expensive failure monitor, such as a VLM judge or a world-model rollout. It is too expensive to run at every control step, so systems pick when to run it:
- at subtask boundaries (the industry default: Gemini Robotics-style success checks);
- at hand-picked events (CARE: gripper close/release plus every 50 actions; ContactGuard: gripper close only);
- on a fixed clock.

We propose a **cheap, policy-derived trigger** that decides when to run the expensive monitor. It is built from:
- disagreement between consecutive ACT action chunks on their overlapping timesteps;
- imminent gripper flips;
- an optional progress-lag probe;
- a max-gap timeout.

Its threshold is tuned to a fixed budget of expensive checks per episode. We compare schedules at **equal budget** on the same logged episodes in gym-aloha, and also measure whether the alarm thresholds keep their nominal false-alarm rate when the trigger picks the frames.

The headline cases are failures **in the middle of a subtask** (the cube is knocked or slips during transport). No gripper event or subtask boundary falls there, so boundary checking and CARE should detect them late or not at all.

## 2. Hypotheses (mirror of `docs/DOE.md` §2)

- **H1 (efficiency):** at 20–30% of the always-on check budget, our trigger's recall is close to always-on, and its detection delay is lower than fixed-k and random at the same budget.
- **H2 (mid-subtask failures):** at equal budget, ours detects knock and slip failures earlier than CARE and boundary checking. On nudge failures, which fall right before a gripper close, CARE and ours are about equal.
- **H3 (calibration):** per-check alarm thresholds calibrated on all nominal frames give an inflated measured false-alarm rate on trigger-selected frames, and the per-episode rate grows with the number of checks. Calibrating on the frames each schedule actually selects ("stream calibration") brings the measured rate back to nominal α.

## 3. Core design: log once, evaluate offline

The monitor is **passive**: it never stops or changes the robot. So every schedule can be evaluated on the same recorded episodes:

1. **Rollout:** run frozen ACT in gym-aloha, with and without perturbations. Log frames, every predicted chunk ("shadow queries", see §6), executed actions and ground-truth sim labels, including the failure onset step.
2. **Score:** run each expensive monitor on every frame of its check grid once. This gives the "always-on" result.
3. **Replay:** each schedule selects grid frames using only causal information. Alarms are read from the precomputed scores.
4. **Evaluate:** compute recall, detection delay, false-alarm rates (per check and per episode) and checks per episode, with paired bootstrap confidence intervals across schedules.

If the design ever makes the monitor act (stop, recover), this shortcut breaks: each schedule would need its own live rollouts. Do not add that without a decision entry.

## 4. Target repository layout

```
adaptive-monitoring/
  HANDOFF.md            this file
  CLAUDE.md             standing rules for every session
  README.md             setup, local and Slurm runs, visualization
  pyproject.toml
  configs/
    base.yaml           every default parameter (loaded by settings.py)
    sweeps/             smoke.yaml, pilot.yaml, main.yaml
    prompts/            vlm_judge_transfer_cube.txt, vlm_judge_insertion.txt
  src/amon/
    settings.py         loads configs/base.yaml; the only home for global params
    version.py          reads the version from pyproject
    sim/                env.py (gym-aloha wrapper, physics access),
                        perturb.py (knock, slip, nudge, peg_misalign),
                        labels.py (phase, onset, failure type from sim state)
    policy/act.py       LeRobot ACT adapter: executor + shadow chunk queries
    rollout/            runner.py, store.py (episode format, spec §2)
    trigger/            features.py (disagreement, gripper flip, progress lag), trigger.py
    monitors/           vlm.py (vLLM client, logprob score), wm.py (model),
                        wm_train.py, wm_score.py, rnd.py (optional cheap reference)
    schedules/          always_on, fixed_k, random_budget, boundary, care,
                        contactguard, ours (one module each, shared base)
    calibration/        conformal.py (per-check, per-episode, stream)
    eval/               replay.py, metrics.py, stats.py (paired bootstrap)
    viz/                episode_video.py, plots.py
  experiments/
    run.py              `python -m experiments.run --spec ... --stage ...`
    sweep.py            expands a spec into shards
    analyze.py          metrics tables (CSV + printed summary)
    plot.py             figures (PNG, PDF, CSV twin)
    viz.py              episode video with overlays
  cluster/              cluster.env, setup.sh, submit.sh, job.sbatch,
                        run_local.sh, collect.sh, pack_results.sh
  scripts/              check.sh, fetch_models.sh, vlm_sanity.py
  tests/
  docs/
    CHANGELOG.md, DOE.md
    design/             spec.md, decisions.md
    paper/              main.tex, references.bib, related_works.md, figures/
  runs/                 <name>_<YYYYMMDD-HHMMSS>/ (git-ignored except README)
```

## 5. Milestones and acceptance criteria

Do them in order. Each ends with: tests pass, a CHANGELOG entry, a version bump, and a DOE update if the design changed.

**M0: Environment and scaffold (day 1)**
- Create the package layout above, `settings.py` loading `configs/base.yaml`, and `scripts/check.sh` (pytest + ruff).
- Pin versions in `pyproject.toml` after checking compatibility: `lerobot`, `gym-aloha`, `mujoco`, `torch`.
- Keep vLLM in a **separate environment** (`.venv-vlm`), because its torch pin may conflict with LeRobot's (decision D-08).
- Accept: `import amon` works. `gym.make("gym_aloha/AlohaTransferCube-v0")` renders a frame under `MUJOCO_GL=egl`.
- Accept: the model XML has been inspected, and the body/geom/joint names of the cube, peg, socket and gripper fingers are written into `configs/base.yaml`.

**M1: Policy adapter and baseline success (days 1–2)**
- `policy/act.py` loads `lerobot/act_aloha_sim_transfer_cube_human` and `lerobot/act_aloha_sim_insertion_human`.
- The executor reproduces each checkpoint's own eval config (`n_action_steps`, temporal ensembling) read from its `config.json`.
- **Shadow queries** (spec §3): at every step, also call the model for a full chunk and log it. The log must not change which action is executed.
- Find the method that returns a full chunk in the pinned LeRobot version (for example `predict_action_chunk`, or the model forward pass). Do not assume.
- Accept: 200 nominal episodes per task, success rate reported with a Wilson 95% CI next to the model-card numbers (83.0% transfer-cube, 20.6% insertion).
- LeRobot issue #2605 reports lower rates on newer versions. Record whatever you measure; do not tune toward the card.
- Accept: a test confirms that the executed-action sequence is identical with shadow queries on and off (same seed).

**M2: Labels, perturbations, episode store (days 2–4)**
- `sim/labels.py`: per-step phase from the env's reward stages (transfer-cube and insertion each have 4 stages; verify against the gym-aloha source); contact flags; object heights; failure events and **onset step** (spec §4).
- `sim/perturb.py`: knock, slip, nudge, peg_misalign, with parameters from `configs/base.yaml`, timed by phase windows (spec §5).
- `rollout/store.py`: the episode format of spec §2.
- Accept: for each perturbation, 10 episodes are rendered to video (M7's viz tool can be a quick version here), and a human (the user) confirms visually that the event happens as intended.
- Accept: onset labels land within ±2 steps of the visible event on those videos.
- Accept: a smoke spec writes `runs/smoke_<stamp>/` end to end for rollout.

**M3: VLM monitor and sanity gate (days 3–5)**
- `monitors/vlm.py`: an OpenAI-compatible client to a local vLLM server running `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`. Prompts come from `configs/prompts/`.
- Score = P("No" | "Is the task on track?") from first-token logprobs over the Yes/No tokens (spec §6.1).
- Batch offline scoring across frames. Cache responses by (model hash, prompt hash, frame hash).
- `scripts/vlm_sanity.py`: 50 hand-labeled frames (25 on-track, 25 failed) drawn from M2 videos. Report AUROC and ms per call on a 3070.
- **Gate:** if AUROC < 0.75, or the 7B model won't fit in 8 GB, try `Qwen2.5-VL-3B-Instruct-AWQ`. If that also fails, make the world model the primary expensive monitor and record the decision.

**M4: World-model monitor (days 4–6)**
- Default (PROPOSED, D-04): a frozen image encoder (DINOv3 ViT-S/16; DINOv2 ViT-S/14 if DINOv3 access is gated) plus a small trained action-conditioned predictor over latent history. Score = k-step latent prediction error (spec §6.2).
- Variant: a LeWM-style end-to-end JEPA (≈15M params), if time allows and code is available. Check for official LeWM code (arXiv 2603.19312) before writing it from scratch.
- Train on the `train` split of unperturbed transfer-cube episodes (spec §7). Measure ms per check.
- Accept: on M2 knock/slip episodes, mean score rises after onset compared with nominal episodes (a plot is enough).

**M5: Trigger, schedules, calibration, replay, metrics (days 5–8)**
- Implement exactly spec §8–§11. These modules are pure functions over logged arrays, so unit-test them heavily with synthetic streams:
  - known disagreement spikes;
  - known gripper flips;
  - budget bisection hits its target;
  - conformal quantile uses the finite-sample formula;
  - delay arithmetic, including grid snapping and latency.
- Accept: `experiments.run --stage evaluate` on the pilot run gives a metrics CSV and the printed summary of spec §12.

**M6: Pilot then main sweep (days 8–11)**
- Pilot spec (20 episodes per cell). Check the full pipeline, measure throughput, then size the main sweep in `docs/DOE.md` §5.
- Main sweep, locally (`cluster/run_local.sh`) or on Slurm (`cluster/submit.sh`). Never mix hardware within one comparison (D-09).
- Optional: P²-style progress probe; FIPER-style RND cheap-monitor reference.

**M7: Analysis, figures, paper (days 11–14)**
- Figures (spec §13) with CSV twins, an episode overlay video, the results section, and an arXiv re-check of related work.

## 6. Things that are easy to get wrong

- **ACT action space is joint space**: 14-D, with 6 arm joints + 1 gripper per arm, left arm first (verify order). There is no end-effector translation/rotation, so disagreement is normalized per arm and per gripper.
- **Shadow queries must not change execution.** Keep the executor's chunk queue separate from the logging query, and check RNG streams too: ACT's eval-time VAE latent is zero, but confirm the forward pass is deterministic.
- **Causality.** A schedule at step t may use only information available at t: frames ≤ t, chunks queried ≤ t, executed actions ≤ t. Never sim labels; the boundary schedule is the one documented exception (D-06).
- **Grid snapping.** The expensive monitor runs on a grid (VLM: every 5 steps by default). A trigger firing between grid points is served at the next grid point ≥ t, and that wait counts toward delay.
- **Budget fairness.** Fixed-count schedules (CARE, boundary, ContactGuard) have whatever budget they produce. Compare ours at *their realized budget* and also on the full budget curve.
- **Calibration data.** Calibrate only on successful, unperturbed episodes from the `calib` split. Measure false alarms only on successful, unperturbed `test` episodes. Report perturbed episodes that still succeed in their own bucket.
- **Monitor latency.** Report raw delay (steps from onset to the check that alarms) and effective delay (+ monitor latency in steps at 50 Hz). An always-on 7B VLM is not physically realizable; say so in the paper.

## 7. First session instructions for Claude Code

1. Read `CLAUDE.md`, `docs/design/spec.md`, `docs/design/decisions.md`, `docs/DOE.md`.
2. Do M0. Before writing perturbation code, open the gym-aloha package source (env classes, task reward definitions, assets XML). Record the real names and reward stages in `configs/base.yaml` and `docs/design/decisions.md`.
3. Stop after M1 and report the measured success rates to the user before generating any perturbed data.
