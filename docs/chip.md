# Running experiments on UMBC chip (Slurm)

chip is UMBC's HPC cluster. GPU jobs go to the `chip-gpu` cluster through Slurm, so nothing runs on the login node except file work and job submission. This page is the runbook, followed by what the first attempt taught about the machine.

## The machine in one table

| Item | Value | Why it matters |
|---|---|---|
| Login | `ssh ddawes2@chip.rs.umbc.edu` (lands on `chip-login1`) | no GPU on the login node |
| GPU cluster flag | `-M chip-gpu` on `squeue`, `sacct`, `scancel`, `scontrol` | without it Slurm looks at the CPU cluster |
| Account | `pi_cmat` | `--account=pi_cmat` |
| Home quota | 500 MB | environments, caches and runs live in `/umbc/rs/pi_cmat/users/ddawes2` |
| g20-01 … g20-13 | Quadro RTX 6000 (24 GB), 4–8 per node, 96 CPUs | **the only nodes that can render the simulator** |
| g24-01 … g24-12 | L40S (46 GB), 2–4 per node, 32 CPUs | no NVIDIA EGL library: MuJoCo cannot render here |
| Driver | 610.57 on both | VLM environment is the vLLM 0.34 lock (D-28) |
| Pick a GPU model | `--constraint=RTX_6000` or `--constraint=L40S` | node features; `scontrol -M chip-gpu show node g20-08` lists them |

## Runbook

On the laptop, copy the code (the repository is private, so a git bundle needs no login on chip):

```bash
git bundle create amon.bundle m0-m2-pipeline
scp amon.bundle ddawes2@chip.rs.umbc.edu:/umbc/rs/pi_cmat/users/ddawes2/
```

On chip, the first time:

```bash
mkdir -p /umbc/rs/pi_cmat/users/ddawes2/amon/logs
cd /umbc/rs/pi_cmat/users/ddawes2
git clone -b m0-m2-pipeline amon.bundle manipulation-adaptive-monitor
cd manipulation-adaptive-monitor
STEPS="setup check smoke" sbatch server/chip_job.sbatch
```

To update later: copy a new bundle, then `git pull /umbc/rs/pi_cmat/users/ddawes2/amon.bundle m0-m2-pipeline` in the repository, and rerun `check` before any sweep.

`server/chip_job.sbatch` asks for one g20 node with 4 GPUs, 32 CPUs, 128 GB and 12 hours. `STEPS` lists what it runs, in order: `setup`, `check`, or the name of any sweep in `configs/sweeps/` (`smoke`, `pilot`, `baseline`). Each sweep is packed into a tar file at the end, as on the 3070 server (`docs/server.md` section 6).

## Watching a job

```bash
squeue -M chip-gpu -u $USER                         # PD = waiting, R = running
tail -f /umbc/rs/pi_cmat/users/ddawes2/amon/logs/amon-<jobid>.out
sacct -M chip-gpu -j <jobid> --format=JobID,State,Elapsed,ExitCode
scancel -M chip-gpu <jobid>
```

A short interactive test on one GPU (good for a quick diagnostic, not for sweeps):

```bash
srun -M chip-gpu -p gpu -A pi_cmat --constraint=RTX_6000 --gres=gpu:1 -c 4 --mem=16G -t 00:20:00 --pty bash
```

`sbatch --wait` blocks until the job ends and returns its exit code, which is handy in scripts.

## What the first attempt taught (Nov 26, 2026)

1. **The determinism check failed, and the cause was anti-aliasing.** MuJoCo renders offscreen with 4× multisampling by default (`offsamples=4`). On the NVIDIA EGL driver, rendering the very same state six times gave 2–3 different images: 1–2 pixels out of 307,200 differed by one grey level. The physics state was bit-identical. The policy reads the image, so one grey level in one pixel eventually changes an action, and the episode diverges. With `offsamples=0` all renders were bit-identical. The fix is `render.offsamples: 0` in `configs/base.yaml` (D-29), and `tests/test_sim_and_policy.py::test_rendering_the_same_state_is_deterministic` guards it.
2. **How it was found.** The failing test, `test_reset_is_deterministic`, does not touch the policy: it resets the simulator twice and compares the images. That ruled out GPU kernels in the policy and pointed at rendering. A small script then rendered one state repeatedly with and without multisampling.
3. **Only g20 nodes can render.** The g24 (L40S) nodes have only Mesa's EGL (`/usr/share/glvnd/egl_vendor.d/50_mesa.json`, no `libEGL_nvidia.so`), so dm_control fails with `Cannot initialize a headless EGL display`. The g20 nodes have `10_nvidia.json`. The job therefore requests `--constraint=RTX_6000`.
4. **The `gpu` partition mixes GPU models.** Without a constraint, a job can land on either model. Results from different GPU models are different hardware groups (D-09), so always pin the model.
5. **Nothing big in home.** The 500 MB quota fills with one pip cache. `chip_job.sbatch` sets `AMON_HOME` and `XDG_CACHE_HOME` in the group space.
6. **Python and EGL import order.** In a diagnostic script, import dm_control (or anything from `amon.sim`) before `OpenGL`. PyOpenGL picks its platform on first import, and dm_control can only select EGL if it comes first.
