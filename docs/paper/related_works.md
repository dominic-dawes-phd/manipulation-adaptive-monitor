# Related work tracker

One row per paper: what it does, how much it overlaps with this project, the exact difference (delta), and how closely it has been read. **Read** = full text; **Abstract** = abstract or search snippet only. Re-check arXiv for new work the week of the presentation, with queries such as "adaptive monitoring trigger robot", "when to query VLM robot failure", "event-triggered failure detection action chunk".

## Closest: when and what to check

| Paper | What it does | Overlap | Delta to ours | Read |
|---|---|---|---|---|
| CARE (2609.24118) | Runs VLM checks at gripper close/release events plus every 50 actions | Medium | Its events are fixed by hand. Ours fires on policy-derived risk (chunk disagreement), so it can fire in mid-transport, and its budget is tuned to match. **Our main "theirs" arm.** | Abstract |
| ContactGuard (2608.13438) | Runs a world-model check at the commanded gripper open→close transition; leaves "general-purpose triggers for other contact events" as future work | Medium-high | Single fixed event, no budget matching or schedule comparison. A "theirs" arm. | Abstract |
| CheckVLA (2607.26789) | World-model verification of chunks; beats periodic replanning at a matched policy-call budget | Medium-high | Its trigger is world-model exceedance and it budgets *policy* calls. Ours gates an *expensive monitor* with a cheap action-derived trigger. | Abstract |
| VLA-Corrector (2607.01804) | Event-triggered adaptive execution horizon from latent visual deviation | Medium | Changes how far to execute open-loop, not when to run a monitor | Abstract |
| RePlanMem (2610.22556), RESETTLE (2610.12185) | Chunk disagreement triggers memory writes or recovery | Low-medium | Disagreement used as a trigger for other things, not for monitor invocation | Abstract |
| Knowing When to Stop (2609.00908), 2610.21434 | Adaptive execution horizon | Low | Not monitoring | Abstract |
| LADDER (2610.18053) | Counts VLM calls per episode as a cost | Low | Supports our cost metric | Abstract |

## Runtime failure detectors (what an expensive or cheap monitor computes)

| Paper | What it does | Overlap | Delta | Read |
|---|---|---|---|---|
| FAIL-Detect (2503.08558, RSS 2025) | Learned OOD scores (logpZO) with time-varying conformal bands | Low | A detector, not a schedule. Its per-episode calibration is our `episode_max` baseline. | Abstract + code |
| FIPER (2510.09459, NeurIPS 2025) | RND in policy embedding + action-chunk entropy, both conformally calibrated | Low | Always-on cheap detector; optional cheap reference arm | Abstract |
| Sentinel / STAC (2410.04640, CoRL 2024) | Temporal action consistency + VLM progress monitor | Medium | Uses consistency as the *detector*; its VLM runs on its own schedule. Ours uses consistency to *schedule* the expensive monitor. | Abstract |
| VLA-FAIL / ACC (2606.21386) | Action-chunk consistency detector; smoothing the score delays detection | Medium | Same signal family, used as a detector. Cite for the disagreement definition. | Abstract |
| Rewind-IL / TIDE (2604.16683) | Split-conformal inter-chunk discrepancy, then state respawn | Medium | Detector plus recovery; no expensive monitor | Abstract |
| ActProbe (2606.08508) | Learned probe on chunk-consistency features | Low-medium | Detector | Abstract |
| Foresight (2606.23085), FoMo-FD (2607.27511) | World-model latent monitors with functional / global conformal thresholds | Low | Candidates for the world-model monitor design | Abstract |
| Progress Probes / P² (2610.31613) | Linear probes on VLA activations estimate progress; flags lag | Low-medium | Cheap always-on monitor. We use a P²-style probe as an optional *trigger component*. | Read |

## Orchestration practice (motivation for "industry = check at subtask boundaries")

| Paper | Relevance |
|---|---|
| Gemini Robotics 1.5 (2510.03342) | An embodied-reasoning orchestrator issues subtasks and checks success at their ends |
| What Matters in Orchestrating Robot Policies (2606.10267) | Success-detector termination beats fixed-frequency switching; studied static environments only |

## Calibration

| Paper | Relevance |
|---|---|
| Vovk et al. 2005; Angelopoulos & Bates 2021 | Split-conformal quantiles and exchangeability |
| 2610.19438 | Measured exceedance of 15.6% (canonical) and 33.3% (adaptive stream) at nominal 10%. Motivates H3 and our reporting rule. |
| Median Temporal Ensembling (2609.27167) | Consistency monitors can be blind to common-mode shifts; a known limit of our trigger's disagreement component |

## Tools

ACT (Zhao et al. 2023), LeRobot, gym-aloha, MuJoCo, Qwen2.5-VL, vLLM, DINOv2/DINOv3, DINO-WM, LeWM.
