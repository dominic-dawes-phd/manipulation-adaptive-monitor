# adaptive-monitoring

When should an expensive failure monitor (a VLM judge or a world-model rollout) look at a running robot policy? This repository compares monitoring schedules at an equal check budget for frozen ACT policies in the gym-aloha bimanual simulator. Every episode is logged once and every schedule is evaluated offline on the same episodes.

- Project brief and milestones: `HANDOFF.md`
- Definitions (formats, schedules, metrics): `docs/design/spec.md`
- Experimental design and run log: `docs/DOE.md`
- What changed: `docs/CHANGELOG.md`

## Directory map

| Path | Contents |
|---|---|
| `configs/` | `base.yaml` (all defaults), `sweeps/` (smoke, pilot, main), `prompts/` (VLM prompts) |
| `src/amon/` | the package: sim, policy, rollout, trigger, monitors, schedules, calibration, eval, viz |
| `experiments/` | command-line entry points (`run`, `sweep`, `analyze`, `plot`, `viz`) |
| `cluster/` | Slurm and multi-GPU-box runners |
| `scripts/` | checks, model downloads, VLM sanity test |
| `docs/` | DOE, changelog, design docs, paper |
| `runs/` | `<name>_<YYYYMMDD-HHMMSS>/` result folders (git-ignored) |

## 1. Setup (Linux, NVIDIA GPU)

Two Python environments are used, because vLLM pins its own torch version (decision D-08).

```bash
# Simulation + policy + world model
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
export MUJOCO_GL=egl                     # headless rendering

# Windows: use WSL2 and keep the environment on the Linux filesystem, for example
# uv venv ~/amon/venv --python 3.11 (a venv under /mnt/c is slow).

# VLM server (separate environment)
uv venv .venv-vlm --python 3.11
.venv-vlm/bin/pip install vllm           # version pinned in docs/design/decisions.md after M0

# Weights (ACT checkpoints, VLM, encoder) into $HF_HOME
bash scripts/fetch_models.sh
```

Check the install:

```bash
bash scripts/check.sh                    # pytest + ruff
python -c "import gymnasium as gym, gym_aloha; env = gym.make('gym_aloha/AlohaTransferCube-v0'); env.reset(seed=0); print('ok')"
```

## 2. First runs on one machine

```bash
# Measure baseline ACT success (M1); writes runs/baseline_<stamp>/
python -m experiments.run --spec configs/sweeps/baseline.yaml --stage all

# Full smoke test, all stages
python -m experiments.run --spec configs/sweeps/smoke.yaml --stage all
```

The VLM stages need a server. Start it in the VLM environment, one per GPU, each on its own port:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vlm/bin/vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
  --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image": 1}'   # flag syntax differs across vLLM versions: check --help
curl -s localhost:8000/health && python scripts/vlm_sanity.py --endpoint http://127.0.0.1:8000/v1
```

## 3. All GPUs on one machine (no Slurm)

`cluster/run_local.sh` runs one shard per GPU through the same per-shard script as Slurm, then collects the results:

```bash
bash cluster/run_local.sh configs/sweeps/pilot.yaml 0,1,2,3,4,5,6,7
```

Shard logs go to `runs/<name>_<stamp>/logs/shard_<i>.out` and to the terminal with a `[shard i]` prefix. Ctrl+C stops every shard. Run it inside `tmux` for long sweeps (`tmux new -s amon`; detach with Ctrl+B D; reattach with `tmux attach -t amon`).

## 4. University Slurm cluster

1. Edit **only** `cluster/cluster.env`: account, cluster flag, GPU constraint, data directory (keep everything off a small home quota), time limit.
2. One-time setup inside a GPU job (module systems often don't work on login nodes):
   ```bash
   srun $SLURM_CLUSTER_FLAG --account=$ACCOUNT --gres=gpu:1 --time=01:00:00 --pty bash cluster/setup.sh
   ```
   This creates both venvs under `$DATA_DIR`, downloads the weights, and writes `cluster_info.txt` (partitions, driver, disk).
3. Submit a sweep as a job array; a dependent job collects the results:
   ```bash
   bash cluster/submit.sh configs/sweeps/main.yaml
   squeue $SLURM_CLUSTER_FLAG -u $USER
   ```
4. Bring results back:
   ```bash
   bash cluster/pack_results.sh runs/main_<stamp>       # prints the scp line
   ```

Each stage (`rollout`, `score_vlm`, `train_wm`, `score_wm`, `evaluate`, `plot`) is its own array or job, chained with `--dependency=afterok`. Never pool cluster and workstation runs in one comparison (D-09).

## 5. Analyze and visualize

```bash
python -m experiments.analyze --run runs/pilot_<stamp>          # metrics CSV + printed summary
python -m experiments.plot    --run runs/pilot_<stamp>          # figures/ (PNG, PDF, CSV twin)
python -m experiments.viz     --run runs/pilot_<stamp> \
  --episode transfer_cube/knock/test/ep_030012                  # MP4 with overlays
```

The episode video shows the top camera with a synced timeline underneath:
- the trigger features d̃(t) and g(t);
- tick marks for each schedule's checks and alarms;
- the ground-truth onset (red line) and subtask boundaries (grey lines).

## 6. What the terminal shows

Each stage prints a header (spec, stage, shard, version, git commit, GPU) and then progress lines with episodes done, rate and estimated time left. A failed episode is logged and skipped; the stage's exit code reports it, and `collect.sh` writes an `only:` rerun line into `summary.txt`.

## Troubleshooting

- `GLFW` or `OpenGL` errors: set `MUJOCO_GL=egl` (on CPU-only nodes, `osmesa`).
- vLLM out of memory on an 8 GB card: lower `--max-model-len`, cap `max_pixels` in `base.yaml`, or switch to the 3B-AWQ fallback.
- ACT success far below the model card: expected on newer LeRobot versions (issue #2605). Record the measured rate; do not tune toward the card.
