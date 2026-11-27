# Running experiments on the 8× RTX 3070 server

Everything reported in the paper comes from this machine (D-22), and it has no Claude Code. Every step below is a command to copy and paste. Results come back to the laptop as one tar file per run.

You need:
- a shell account on the server with internet access (no sudo);
- about 50 GB free in your home directory, or set `AMON_HOME` to a larger disk;
- about 10 GB free next to the repository for `runs/`.

| Step | Command | Time | Needed |
|---|---|---|---|
| 1 | copy the code | 2 min | once, and after every update |
| 2 | `bash server/setup.sh` | 20–40 min | once |
| 3 | `bash run_3070_server.sh --check` | 10–20 min | after every update |
| 4 | smoke sweep | about 10 min | once per update |
| 5 | pilot sweep | about 1 h | once |
| 6 | pack and copy back | a few minutes | per run |

## 1. Get the code onto the server

The repository is private, so the simplest route needs no GitHub login on the server: make a git bundle on the laptop and copy it. The commit hash travels with it, and every `run.json` records it.

On the laptop (Git Bash or WSL, inside the repository):

```bash
git bundle create amon.bundle m0-m2-pipeline
scp amon.bundle <you>@<server>:~/
```

On the server:

```bash
git clone -b m0-m2-pipeline ~/amon.bundle manipulation-adaptive-monitor
cd manipulation-adaptive-monitor
```

To update later, make a new bundle on the laptop and copy it. Then, on the server:

```bash
cd ~/manipulation-adaptive-monitor
git pull ~/amon.bundle m0-m2-pipeline
```

Do not edit tracked files on the server. A change makes `run.json` mark the run dirty. The one exception is the VLM model override that step 3 may ask for.

## 2. Setup (once)

```bash
bash server/setup.sh
```

The script does the following:
- checks the GPUs, the NVIDIA driver, glibc, disk space and network access;
- installs `uv` into `~/.local/bin`;
- installs the simulation environment (`~/amon-3070/venv`, the exact versions used on the laptop);
- installs a separate VLM environment (`~/amon-3070/venv-vlm`) that matches the driver: vLLM 0.34 for driver 580 or newer, otherwise vLLM 0.11;
- uses `zig` as a C compiler if the server has none;
- downloads the ACT checkpoints, both Qwen VLMs and the DINOv2 encoder.

It prints a machine summary and saves it to `~/amon-3070/server_info.txt`. You can rerun it safely; finished steps are skipped.

It ends with `[setup] done. Next: bash run_3070_server.sh --check`. If it stops with `[amon] ERROR: ...`, the message names the problem. Common ones:

| Message | Meaning | Fix |
|---|---|---|
| `not an RTX 3070` | another GPU model in the box | `ALLOW_OTHER_GPU=1 bash server/setup.sh` only if you are sure; such results are a separate hardware group |
| `NVIDIA driver ... older than 525` | driver too old for CUDA 12 | ask the admin to update the driver |
| `cannot reach ...` | proxy or firewall | set `https_proxy`, or ask the admin |
| disk warning | under 50 GB free | `AMON_HOME=/bigdisk/amon bash server/setup.sh`; use the same `AMON_HOME` in every later command |

## 3. Server check (before any data, and after every update)

```bash
bash run_3070_server.sh --check
```

This takes 10–20 minutes. It runs three checks:
1. **The test suite**, including the determinism tests (about 3 min).
2. **Determinism across processes and GPUs.** The same 6 dev episodes are rolled out on GPU 0 and on GPU 1, and every stored array must be bit-identical (about 5 min).
3. **The VLM.** The script first tries `Qwen2.5-VL-7B-Instruct-AWQ` on GPU 0. If that fails, it retries with `--enforce-eager`, then falls back to the 3B model. It then measures the latency of single requests. The first start compiles kernels, so it can take 5–10 min.

The end of the output looks like this:

```
server check 2026-11-27T10:12:03 on <host>, commit 1a2b3c4
  tests        PASS   (runs/server_check_.../pytest.log)
  determinism  PASS   (runs/server_check_.../compare.log)
  vlm          PASS   Qwen/Qwen2.5-VL-7B-Instruct-AWQ transfer_cube 448 px: 350 ms mean, 420 ms p95; ...

Server check passed. Next: bash run_3070_server.sh configs/sweeps/smoke.yaml
```

- **If the 7B needed `--enforce-eager`:** the check records this in `~/amon-3070/vlm.env`, and the sweeps use it automatically.
- **If only the 3B loaded:** the check prints one line to add under `overrides:` in `configs/sweeps/smoke.yaml` and `configs/sweeps/pilot.yaml`. Add it before running them.
- **If tests or determinism FAIL:** do not produce results. Copy back `runs/server_check_<stamp>/` and `~/amon-3070/server_info.txt` (step 6 shows how).

Sweeps refuse to start until the check has passed for the current commit.

## 4. Smoke sweep

Run long jobs inside `tmux`, so they survive a dropped connection:
- start a session: `tmux new -s amon`
- detach: Ctrl+B, then D
- reattach: `tmux attach -t amon`

If tmux is missing, use `screen -S amon` or `nohup ... &`.

```bash
tmux new -s amon
bash run_3070_server.sh configs/sweeps/smoke.yaml
```

The smoke sweep has 30 episodes and runs four stages, each across all GPUs:
1. `rollout`: the episodes, stored with 448 px frames.
2. `train_wm`: the world model, trained on the 8 train episodes.
3. `score_wm`: world-model scores.
4. `score_vlm`: one vLLM server per GPU, then VLM scores.

Every stage prints `[run_local] stage ... started` and per-shard progress lines. The run ends with a `summary.txt` that is also printed:

```
== shard statuses ==
shard=0 stage=rollout status=0
...
== failed episodes ==
== episode counts ==
30
```

Every `status=` should be 0 and the failed-episode list empty. If a stage fails, the script stops after that stage and prints how to continue. Fix the cause, then rerun the same command with `RUN_DIR=runs/smoke_<stamp>` in front; finished episodes and scores are kept.

## 5. Pilot sweep

```bash
bash run_3070_server.sh configs/sweeps/pilot.yaml
```

The pilot has about 320 episodes over both tasks and all perturbations, with the same four stages. Expect roughly 15–30 min of rollouts on 8 GPUs with 2 processes each. World-model training and scoring take about 10 min, and VLM scoring 10–30 min.

The pilot measures:
- perturbation yields, which size the main sweep;
- throughput;
- VLM and world-model latency on the 3070;
- a first look at the scores.

`evaluate` and `plot` are not run on the server yet. They are added once M5 is built, and they run on the laptop from the packed results (D-27).

## 6. Pack and copy back

On the server:

```bash
bash server/pack.sh runs/smoke_<stamp>
bash server/pack.sh runs/pilot_<stamp>      # about 4 GB with 448 px frames
```

Each command prints the exact `scp` line for the laptop. On the laptop (Git Bash or WSL, inside the repository):

```bash
scp <you>@<server>:~/manipulation-adaptive-monitor/runs/pilot_<stamp>.tar* runs/
cd runs && sha256sum -c pilot_<stamp>.tar.sha256 && tar -xf pilot_<stamp>.tar && cd ..
```

For a failed check, copy the folder and the machine summary directly:

```bash
scp -r <you>@<server>:~/manipulation-adaptive-monitor/runs/server_check_<stamp> runs/
scp <you>@<server>:~/amon-3070/server_info.txt runs/server_check_<stamp>/
```

Then tell Claude Code on the laptop which runs arrived. Every run folder carries `run.json` (commit, GPU, renderer, package versions), `summary.txt` and `logs/`, so the analysis needs nothing else.

## What each run folder holds

| Path | Content |
|---|---|
| `run.json`, `spec.yaml` | code commit, host, GPU, renderer, packages, the sweep spec |
| `episodes/<task>/<perturbation>/<split>/ep_<seed>/` | `data.npz` (frames, actions, shadow chunks, labels) and `meta.json` |
| `models/wm_<task>.pt`, `eval/wm_train_<task>.json` | world model and its validation losses |
| `scores/wm/...`, `scores/vlm/...` | per-episode `grid_steps` and `score`; `meta_<task>_shard<i>.json` holds the model hashes and the latency measured on that GPU |
| `logs/` | one log per shard and stage, plus the vLLM server logs |
| `summary.txt` | shard statuses, failed episodes, episode count |

## Troubleshooting

- **A vLLM server does not start in `score_vlm`.** Read `runs/<run>/logs/vllm_shard<i>.log`. Out of memory means another process holds GPU memory; check with `nvidia-smi`. Rerun `--check` to re-derive the flags.
- **`EGL` or `OpenGL` errors in rollouts.** Rendering on each process's own GPU may not work on this driver. Use `EGL_PER_GPU=0` in front of both the check and the sweep, so all rendering happens on the default GPU. Rerun the check that way first, because pixels can depend on the rendering device.
- **A stray vLLM process after Ctrl+C.** `pkill -f "venv-vlm/bin/vll[m]"`.
- **Another user's jobs share the GPUs.** Pick free GPUs explicitly, e.g. `bash run_3070_server.sh configs/sweeps/pilot.yaml 4,5,6,7`. Do the same for `--check`.
- **Disk full.** Runs live in `runs/` next to the repository. Pack and copy them back, then delete them on the server.
