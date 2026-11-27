# Locked environments

`pyproject.toml` holds the direct pins (D-11). These files freeze everything underneath them for Python 3.11 on Linux x86_64.

| File | Machine | Used by |
|---|---|---|
| `laptop_4070.txt` | RTX 4070 Laptop, WSL2 Ubuntu 22.04 | record of the environment that produced the M1 and M2 runs |
| `server_3070.txt` | 8x RTX 3070 server, Ubuntu 20.04 or newer (glibc 2.31) | `server/setup.sh`; same versions, with hashes |

Both list the same versions, so the two machines differ only in hardware, driver and renderer (D-09, D-12). torch 2.7.1 is the CUDA 12.6 build; the NVIDIA driver must support CUDA 12. To regenerate the server file after changing a pin:

```bash
uv pip compile pyproject.toml --extra dev --python-version 3.11 \
  --python-platform x86_64-manylinux_2_31 --generate-hashes --no-annotate --no-header \
  -c requirements/laptop_4070.txt -o requirements/server_3070.txt
```

The VLM server has its own environment (D-08). Which lock it uses depends on the NVIDIA driver (D-28); `server/setup.sh` picks it automatically:

| File | Driver | Contents |
|---|---|---|
| `vlm_freeze.txt` | 580 or newer (CUDA 13) | vLLM 0.34.0, torch 2.14.0, transformers 5.22.2; frozen from the laptop's `~/amon/venv-vlm` |
| `vlm_cu12.txt` | 525 or newer (CUDA 12.8) | vLLM 0.11.0, torch 2.8.0, transformers 4.57.6, with hashes; compiled from `vlm_cu12.in` |

To regenerate the CUDA 12 lock:

```bash
uv pip compile requirements/vlm_cu12.in --python-version 3.11   --python-platform x86_64-manylinux_2_31 --generate-hashes --no-header -o requirements/vlm_cu12.txt
```

By hand, outside the server scripts:

```bash
uv venv --python 3.11 ~/amon/venv-vlm && uv pip sync -p ~/amon/venv-vlm/bin/python requirements/vlm_freeze.txt
scripts/serve_vlm.sh            # model, pixel cap and port from configs/base.yaml
```
