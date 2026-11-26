# Journal: 2026-11-vllm-27b-tp2

Every step in order: why, the exact command, expected vs actual. Mistakes and dead ends stay in. Plan, pinned versions and method: [`README.md`](README.md).

## 0. Plan (2026-11-26)

Written from the source repository (README with its compose file, `kv-quality-test.md`, `kvq/run-server.sh`), the checkpoint's Hugging Face file list and history, and the two vLLM images' configs (read with `docker buildx imagetools inspect`, nothing pulled):

- `vllm/vllm-openai:v0.30.0`: `sha256:8a69ffad…200a4b90`, created 2026-09-21, `CUDA_VERSION=13.0.2`, `NV_CUDA_CUDART_VERSION=13.0.96-1`, `TORCH_CUDA_ARCH_LIST=7.5 8.0 8.6 8.9 9.0 10.0 12.0`.
- `vllm/vllm-openai:v0.32.0`: `sha256:befccf0e…7957c5b34`, created 2026-10-18, the same CUDA.
- `apt-cache policy`: `nvidia-driver-595-server-open` 595.91.07-0ubuntu0.24.04.1 (the source's driver), `nvidia-driver-595-open` 595.99.02, `cuda-toolkit-13-2` 13.2.2-1. Nothing installed.

The images need CUDA 13.0, which the installed driver (580.178.04) runs, so the reproduction starts on it (phase A) and the source's driver is a later step of its own (phase B).
