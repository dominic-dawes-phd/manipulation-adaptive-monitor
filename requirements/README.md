# Locked environments

`pyproject.toml` holds the direct pins (D-11). These files freeze everything underneath them for Python 3.11 on Linux x86_64.

| File | Machine | Used by |
|---|---|---|
| `laptop_4070.txt` | RTX 4070 Laptop, WSL2 Ubuntu 22.04 | record of the environment that produced the M1 and M2 runs |
| `server_3070.txt` | 8x RTX 3070 server, Ubuntu 20.04 or newer (glibc 2.31) | `run_3070_server.sh`; same versions, with hashes |

Both list the same versions, so the two machines differ only in hardware, driver and renderer (D-09, D-12). torch 2.7.1 is the CUDA 12.6 build; the NVIDIA driver must support CUDA 12. To regenerate the server file after changing a pin:

```bash
uv pip compile pyproject.toml --extra dev --python-version 3.11 \
  --python-platform x86_64-manylinux_2_31 --generate-hashes --no-annotate --no-header \
  -c requirements/laptop_4070.txt -o requirements/server_3070.txt
```

The VLM server has its own environment (D-08), `~/amon/venv-vlm`, created with `uv venv --python 3.11` and `uv pip install vllm openai`. `vlm_freeze.txt` records it as first installed on the laptop (vLLM 0.34.0, torch 2.14.0 for CUDA 13.0, transformers 5.22.2). CUDA 13 needs NVIDIA driver 580 or newer, so check the 3070 server's driver before installing there. A hashed lock for the server follows once the M3 gate has picked the model.

```bash
uv venv --python 3.11 ~/amon/venv-vlm && source ~/amon/venv-vlm/bin/activate
uv pip install -r requirements/vlm_freeze.txt
scripts/serve_vlm.sh            # model, pixel cap and port from configs/base.yaml
```
