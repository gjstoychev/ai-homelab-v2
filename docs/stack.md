# Software stack

What is installed on each node right now, and every change to it. Experiments record the exact stack of each run themselves; this file is the current state and its history. Update both the table and the changelog on every change.

## formula — current state

Recorded 2026-11-25.

| Layer | Version | Source |
|---|---|---|
| OS | Ubuntu 24.04.5 LTS | |
| Kernel | 7.0.0-43-generic | Ubuntu |
| NVIDIA driver | 580.178.04, open kernel module (`nvidia-driver-580-server-open` 580.178.04-0ubuntu0.24.04.1) | Ubuntu; supports CUDA up to 13.0 |
| GPU VBIOS | 98.06.4E.00.8B (both cards) | |
| CUDA toolkit | 13.0 (`cuda-nvcc-13-0` 13.0.88-1, `libcublas-dev-13-0` 13.1.1.3-1), in `/usr/local/cuda-13.0` | NVIDIA apt repository |
| NCCL | 2.28.9-1+cuda13.0 (`libnccl2`, `libnccl-dev`) | NVIDIA apt repository |
| Docker | 29.8.0 (`docker-ce` 5:29.8.0-1~ubuntu.24.04~noble) | Docker apt repository |
| NVIDIA Container Toolkit | 1.20.2-1 | NVIDIA apt repository |
| llama.cpp | v1.0.0 (build 12359, commit `b1b770d9`), CUDA, `sm_120`, in `~/llama.cpp/v1.0.0`; `~/llama.cpp/current` points to it | built from source with `scripts/build-llamacpp.sh` |
| Ollama | 0.32.14, installed, disabled at boot; started by hand when needed | v1 (Lab 0b; disabled by its role `ollama` on 2026-11-24) |
| Python | 3.12.3 | Ubuntu |

Models are in `~/models/<name>/`.

## CUDA toolkit on the host

The toolkit, not the driver, comes from NVIDIA's apt repository. The repository, its key and the apt pins were put in place by v1's Ansible role `llama_cpp` on 2026-11-24 and are left as they are:

- `/etc/apt/sources.list.d/nvidia-cuda.sources`, key `/etc/apt/keyrings/nvidia-cuda.asc` (fingerprint `EB693B3035CD5710E231E123A4B469963BF863CC`).
- `/etc/apt/preferences.d/nvidia-cuda`:
  - everything from `developer.download.nvidia.com` at priority 100 (Ubuntu: 500), so a package in both repositories, such as the driver's `libnvidia-*` libraries, keeps coming from Ubuntu and `apt upgrade` cannot swap them under the running kernel module;
  - `libnccl2 libnccl-dev` pinned to `*+cuda13.0` at 600, because the repository also carries NCCL builds for newer CUDA versions under the same package names.

Installed by hand with (dry run first; no line may start with `Inst nvidia-` or `Inst libnvidia-`):

```bash
sudo apt-get -s --no-install-recommends install build-essential cmake ninja-build git libssl-dev cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-driver-dev-13-0 cuda-cccl-13-0 libcublas-dev-13-0 cuda-cuobjdump-13-0 libnccl2 libnccl-dev | grep -E '^Inst'
```

Side effects of the install: `/usr/local/cuda` and `/usr/local/cuda-13` point at `cuda-13.0` (update-alternatives), and `/etc/ld.so.conf.d/000_cuda.conf` and `987_cuda-13.conf` put CUDA 13.0's `libcudart` and `libcublas` on every program's library path.

A newer CUDA (13.1+) starts with a driver upgrade; until then binaries built with it would run only through NVIDIA's minor version compatibility.

## hero

Not built yet.

## Changelog

| Date | Node | Change |
|---|---|---|
| 2026-11-25 | formula | Claude Code sandbox turned off (`/etc/claude-code/managed-settings.json`; previous version kept as `managed-settings.json.v1-sandbox`). This repo started. |
| 2026-11-24 | formula | CUDA 13.0 toolkit and NCCL 2.28.9+cuda13.0 installed by hand; llama.cpp v1.0.0 built; Ollama disabled at boot (experiment `2026-11-llamacpp-flags`). |
| 2026-11 | formula | Board swap to ASUS ROG Maximus Z690 Formula, both GPUs on CPU lanes (experiment `2026-11-boardswap`). Docker and the NVIDIA Container Toolkit installed by hand for it. |
