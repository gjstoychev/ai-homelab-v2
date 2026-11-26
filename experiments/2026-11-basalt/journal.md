# Journal: 2026-11-basalt

Every step in order: why, the exact command, expected vs actual. Mistakes and dead ends stay in. Plan, pinned versions and method: [`README.md`](README.md).

## 0. Plan (2026-11-25)

Plan written from the project's documentation, the Hugging Face file histories and two videos (sources in the README). No commands run yet.

## Decisions (2026-11-26)

- **The same weights as `2026-11-strata`:** `Qwen3.8-FN-GSQ-RCO-IQ3_XXS.basalt` at Hugging Face revision `aa76f46`, the last one before the GSQ-RCO packs were replaced by Unsloth's (`451e426`, 2026-11-07). The Hub API still lists it at that revision with the expected size and SHA-256 (79,865,545,428 bytes, `c20595c2…b365165`).
- **The same client, runner, prompts and measurement rules** as `2026-11-strata` (`scripts/bench-run.sh`, `oai-bench.py`, `quality-check.py`, `long-bench.py`), so every number sits next to Strata's and llama.cpp's.
- **Two GPUs are the main question**, with the matrix of `2026-11-strata` step 13 (README, step 11).
- **Source build at the tag, Flash-Next only** (`-DBASALT_MODELS=qwen4exp`), `scripts/build-basalt.sh`.
- **Configs read from the docs at the tag** (`docs/configuration/index.md`, `gpus.md`, `models/qwen38-flash-next.md`), not from the website:
  - one model entry per config: `"max-context": 32768`, `"kv": "int8"` (Strata's default; Basalt's is fp16, and `parallel` above 1 needs int8), `"prompt-cache": 0` (the conversation cache is off by default in Basalt);
  - "no drafting" = `"spec": 0, "suffix-draft": 0` (`spec`: "`0` or `1` turns it off"); with drafting, Basalt's default `spec` 8 (Strata ran 4, its setup's value);
  - two cards: `"gpu": "0,1"` with `"gpu-mode": "peer"` or `"split"` (required since 0.5.7); no `gpu-share` at first (the defaults: free VRAM for split, the peers' free VRAM for peer experts, memory bandwidth for peer prompt layers).
- The server's `/health` answers `"loaded": false` while it loads and the response carries llama.cpp-style `timings` (`prompt_per_second`, `predicted_per_second`, `draft_n`, `draft_n_accepted`), as Strata's: Basalt is Strata's fork, and the shared scripts work unchanged.

## 1. Disk (2026-11-26)

NVMe 279 GB free with the Strata model (71 GB) in `~/models`; both Basalt packs (79.9 + 87.6 GB) fit next to it. Nothing archived.

## 2. Build (2026-11-26)

**Reading first** (`docs/building.md`, `CMakeLists.txt`, `README.md` at v0.5.7 = `400b411c58031d064812ea21a259b2381acdaa3e`): CMake 3.24+, CUDA 13.0 or newer (`VERSION_LESS 13.0` stops the configure), `CMAKE_CUDA_ARCHITECTURES` defaults to 120 and any other value stops it ("basalt builds for compute capability 12.0 only"); the FP4/FP8 prompt kernels build as `120a-real`. No NCCL: the engine links `CUDA::cudart` and cuBLAS only. The programs land in the source tree's `bin/`, and `bin/basalt serve` runs the Python server from the folder above it. The configure step looks for a Python with numpy, `.venv/bin/python` among its candidates, so the venv is made first.

**Command:** `scripts/build-basalt.sh v0.5.7` (`-G Ninja -DCMAKE_BUILD_TYPE=Release -DBASALT_MODELS=qwen4exp -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-13.0`, half the cores).

**Expected:** a CUDA build for sm_120 linked against CUDA 13.0's cudart and cuBLAS; `bin/basalt`, `basalt-bench`, `basalt-probe`, `basalt-vision`.

**Actual** (`~/basalt/v0.5.7/build-info.txt`): 284 s with `-j 14`, no compiler warnings in the log. `bin/basalt` (162 MB): 88 kernels for `sm_120` and 1 for `sm_120a`; links `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13` from `/usr/local/cuda-13.0/lib64`; no NCCL. `BASALT_MODELS=qwen4exp`, `PEER_ADA=OFF`. The binary has no `--version` flag (the first build-info run printed its usage); its strings carry `engine=0.5.7`, and the script now records that. Venv `src/.venv`: Python 3.12.3 with `requirements.txt`'s pins.

**Verdict:** as expected.
