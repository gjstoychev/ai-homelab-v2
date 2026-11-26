# Software stack

What is installed on each node right now, and every change to it. Experiments record the exact stack of each run themselves; this file is the current state and its history. Update both the table and the changelog on every change.

## formula — current state

Recorded 2026-11-26.

| Layer | Version | Source |
|---|---|---|
| OS | Ubuntu 24.04.5 LTS | |
| Kernel | 7.0.0-44-generic (`linux-image-7.0.0-44-generic` 7.0.0-44.44~24.04.1, HWE) | Ubuntu, installed by unattended-upgrades |
| NVIDIA driver | 595.91.07, open kernel module (`nvidia-driver-595-server-open`, on apt hold); `nvidia-smi` reports CUDA 13.2; `nvidia-persistenced` active | Ubuntu; DKMS built for 7.0.0-44 only |
| GPU VBIOS | 98.06.4E.00.8B (both cards) | |
| CUDA toolkit | 13.0 (`cuda-nvcc-13-0` 13.0.88-1, `libcublas-dev-13-0` 13.1.1.3-1), in `/usr/local/cuda-13.0` | NVIDIA apt repository |
| NCCL | 2.28.9-1+cuda13.0 (`libnccl2`, `libnccl-dev`) | NVIDIA apt repository |
| Docker | 29.8.0 (`docker-ce` 5:29.8.0-1~ubuntu.24.04~noble) | Docker apt repository |
| NVIDIA Container Toolkit | 1.20.2-1 | NVIDIA apt repository |
| llama.cpp | v1.0.0 (build 12359, commit `b1b770d9`), CUDA, `sm_120`, in `~/llama.cpp/v1.0.0`; `~/llama.cpp/current` points to it | built from source with `scripts/build-llamacpp.sh` |
| Basalt (27B build) | v0.5.7 (same commit as above), built with `BASALT_MODELS='qwen4exp;qwen35'` (Flash-Next and the 27B), in `~/basalt-qwen35/v0.5.7`; `~/basalt-qwen35/current` points to it. The Flash-Next-only build in `~/basalt/v0.5.7` stays the pinned one | built from source with `BASALT_ROOT=~/basalt-qwen35 BASALT_MODELS='qwen4exp;qwen35' scripts/build-basalt.sh` |
| nccl-tests | v2.22.0 (commit `826b3e6`), CUDA 13.0, `sm_120`, no MPI, in `~/nccl-tests/v2.22.0`; `~/nccl-tests/current` points to it | built from source with `scripts/build-nccl-tests.sh` |
| Strata | v0.2.21 (commit `d46f5f2`, engine 0.1.67), CUDA, `sm_120`/`sm_120a`, in `~/strata/v0.2.21`; `~/strata/current` points to it; server venv `src/terminal/.venv` (pinned `requirements.txt`); MTP draft layer in `data/mtp/rt` | built from source with `scripts/build-strata.sh` (setup.py's CUDA flags; `setup.sh` not run) |
| Basalt | v0.5.7 (commit `400b411`, engine 0.5.7), CUDA, `sm_120`/`sm_120a`, Qwen3.8-Flash-Next only (`BASALT_MODELS=qwen4exp`), in `~/basalt/v0.5.7/src/bin`; `~/basalt/current` points to it; server venv `src/.venv` (pinned `requirements.txt`) | built from source with `scripts/build-basalt.sh` |
| Hugging Face CLI | `huggingface_hub` 2.4.0, `hf-xet` 1.8.0, in `~/venvs/hf` | pip in a venv |
| Ollama | 0.32.14, installed, disabled at boot; started by hand when needed | v1 (Lab 0b; disabled by its role `ollama` on 2026-11-24) |
| vLLM | Docker images `local/vllm-openai:v0.30.0` (`vllm/vllm-openai@sha256:8a69ffad…200a4b90`) and `local/vllm-openai:v0.32.0` (`@sha256:befccf0e…7957c5b34`), both CUDA 13.0.2 | pulled by digest (experiment `2026-11-vllm-27b-tp2`) |
| Python | 3.12.3 | Ubuntu |

## Model storage

| Path | Disk | Use |
|---|---|---|
| `~/models/<name>/` | NVMe (PM9A1) | Models in use. Engines read from here while they run (mmap, expert tiers on SSD), so active models never run from SATA. |
| `/mnt/models/<name>/` | SATA (Kingston A400 240 GB), ext4, label `models` | Archive: models not in use but kept for later, the large ones first. Moved back to `~/models` before a run. |

`/mnt/models` is mounted by UUID with `defaults,noatime,nofail` (`/etc/fstab`), so the host still boots without the disk. Models are moved with `scripts/move-model.sh <model-dir> <target-parent>`: it copies the directory with its `REVISION` and `SHA256SUMS`, verifies the checksums on the target with O_DIRECT reads (so it reads the disk, not the page cache the copy just filled) and deletes the source only if every file passes.

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
| 2026-11-26 | formula | Basalt v0.5.7 built a second time with the 27B included (`~/basalt-qwen35`, step 10 of `2026-11-vllm-27b-tp2`); pack `jesdga/Qwen3.8-27B-Basalt` revision `10c9a90e` (21.1 GB) downloaded to `~/models/qwen3.8-27b-basalt-quasar-nvfp4`. `alpine` pulled again for a root-owned cleanup (it had been removed). |
| 2026-11-26 | formula | nccl-tests v2.22.0 built from source against the host's NCCL 2.28.9 and CUDA 13.0 (`scripts/build-nccl-tests.sh`; experiment `2026-11-vllm-27b-tp2`, step 9f). |
| 2026-11-26 | formula | NVIDIA driver 580.178.04 → 595.91.07 (`nvidia-driver-595-server-open`, installed by hand with `apt-get install`, package on `apt-mark hold`), rebooted on kernel 7.0.0-44. Verified: both RTX 5060 Ti at 595.91.07, power limit 180 W, both cards visible inside `local/vllm-openai:v0.32.0`, `nvidia-persistenced` active. Way back: `nvidia-driver-580-server-open`. |
| 2026-11-26 | formula | Docker: removed `local/llama.cpp:b12464-nccl`, `local/nvbandwidth:cuda13.0.0`, `ghcr.io/ggml-org/llama.cpp:full-cuda`, `ollama/ollama:0.32.14`, `nvidia/cuda:13.0.0-base-ubuntu24.04`, hello-world, and the build cache (15.8 GB); images 97.4 → 65.9 GB. vLLM v0.30.0 and v0.32.0 images pulled by digest (experiment `2026-11-vllm-27b-tp2`). |
| 2026-11-26 | formula | Kernel 7.0.0-43 → 7.0.0-44 (HWE), installed by unattended-upgrades at 08:28 and running since the reboot at 12:50; the NVIDIA 580.178.04 open module loads on it. |
| 2026-11-26 | formula | Basalt v0.5.7 built from source (experiment `2026-11-basalt`); Basalt packs GSQ-RCO IQ3_XXS (79.9 GB) and UD-IQ3_XXS (87.6 GB) downloaded to `~/models`. |
| 2026-11-25 | formula | Strata v0.2.21 built from source (experiment `2026-11-strata`); Hugging Face CLI in `~/venvs/hf`; models: GSQ-RCO IQ3_XXS downloaded to `~/models`, batiai IQ3_XXS archived to `/mnt/models`. |
| 2026-11-25 | formula | DeepSeek-V4-Flash-0731 UD-IQ1_M (81 GB) moved to `/mnt/models` with `scripts/move-model.sh`: copy 351 s (about 250 MB/s), direct-read verification 269 s, all three checksums OK. |
| 2026-11-25 | formula | Model archive: Samsung 850 120 GB (old Linux install, unused) replaced by a Kingston A400 240 GB SATA SSD, ext4, at `/mnt/models`; Samsung 860 (Windows) moved from the ASMedia controller to an Intel SATA port. Unused Ollama models removed. |
| 2026-11-25 | formula | Claude Code sandbox turned off (`/etc/claude-code/managed-settings.json`; previous version kept as `managed-settings.json.v1-sandbox`). This repo started. |
| 2026-11-24 | formula | CUDA 13.0 toolkit and NCCL 2.28.9+cuda13.0 installed by hand; llama.cpp v1.0.0 built; Ollama disabled at boot (experiment `2026-11-llamacpp-flags`). |
| 2026-11 | formula | Board swap to ASUS ROG Maximus Z690 Formula, both GPUs on CPU lanes (experiment `2026-11-boardswap`). Docker and the NVIDIA Container Toolkit installed by hand for it. |
