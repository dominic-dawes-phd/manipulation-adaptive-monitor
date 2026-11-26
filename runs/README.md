# runs/

One folder per run, named `<name>_<YYYYMMDD-HHMMSS>/`, for example `smoke_20261126-091500/` or `main_20261203-220000/`. Folder contents are git-ignored.

```
<name>_<stamp>/
  run.json        amon_version, git commit + dirty flag, host, GPU, package versions,
                  model hashes, the full resolved spec
  spec.yaml       copy of the sweep spec as submitted
  episodes/       <task>/<perturbation>/<split>/ep_<seed>/ (data.npz, meta.json)
  scores/         <monitor>/<episode_id>.npz
  models/         trained world model checkpoints
  eval/           metrics.csv, summary.txt, per-schedule selections
  figures/        PNG + PDF + CSV twin per figure
  logs/           shard_<i>.out, gpu_shard<i>.csv
  summary.txt     written by cluster/collect.sh
```

Never edit a finished run folder. Log every run you report in `docs/DOE.md` §7.
