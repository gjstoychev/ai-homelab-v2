# Journal: 2026-11-vllm-27b-tp2

Every step in order: why, the exact command, expected vs actual. Mistakes and dead ends stay in. Plan, pinned versions and method: [`README.md`](README.md).

## 0. Plan (2026-11-26)

Written from the source repository (README with its compose file, `kv-quality-test.md`, `kvq/run-server.sh`), the checkpoint's Hugging Face file list and history, and the two vLLM images' configs (read with `docker buildx imagetools inspect`, nothing pulled):

- `vllm/vllm-openai:v0.30.0`: `sha256:8a69ffad…200a4b90`, created 2026-09-21, `CUDA_VERSION=13.0.2`, `NV_CUDA_CUDART_VERSION=13.0.96-1`, `TORCH_CUDA_ARCH_LIST=7.5 8.0 8.6 8.9 9.0 10.0 12.0`.
- `vllm/vllm-openai:v0.32.0`: `sha256:befccf0e…7957c5b34`, created 2026-10-18, the same CUDA.
- `apt-cache policy`: `nvidia-driver-595-server-open` 595.91.07-0ubuntu0.24.04.1 (the source's driver), `nvidia-driver-595-open` 595.99.02, `cuda-toolkit-13-2` 13.2.2-1. Nothing installed.

The images need CUDA 13.0, which the installed driver (580.178.04) runs, so the reproduction starts on it (phase A) and the source's driver is a later step of its own (phase B).

## 0b. Disk (2026-11-26)

At the user's request: deleted from `/mnt/models` the two models of the finished `2026-11-llamacpp-flags` experiment (batiai IQ3_XXS `1d0ec19c`, DeepSeek-V4-Flash-0731 UD-IQ1_M `fbbb5b93`; both downloadable again at those revisions, `SHA256SUMS` were in their folders: batiai `4e461bef…`, `86755e0c…`; DeepSeek `b99fa246…`, `c8536187…`, `80a4eb4e…`). Then `scripts/move-model.sh` moved the GSQ-RCO GGUF (71 GB, checksums OK) and started on the Basalt GSQ-RCO pack (75 GB) to `/mnt/models`. The A400 SATA disk wrote at about 60 MB/s on these long writes (its cache full), not 250 MB/s as on the first move. Planned next: Docker cleanup (board-swap images and build cache; keep searxng, n8n and the vLLM images).

## 1. Images (2026-11-26)

`docker pull vllm/vllm-openai@sha256:…` for both digests, tagged `local/vllm-openai:v0.30.0` (30.7 GB on disk, 809 s) and `local/vllm-openai:v0.32.0` (33.1 GB, 312 s).

## 2. Checkpoint (2026-11-26)

`scripts/hf-fetch.sh gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 0cc27958cefbbe231782ec8511de8c4eb5233348 ~/models/qwen3.8-27b-nvfp4-0cc2795 <the 15 files of the source's wget loop>`: 652 s; 4 LFS files OK by SHA-256 (the three safetensors, `tokenizer.json`), 11 small git files OK by their git blob SHA-1. `hf-fetch.sh` learned the git-blob check for this (it handled LFS files only); `SHA256SUMS` has the SHA-256 of all 15.

## 3. Smoke test, first try (2026-11-26): not finished

New: `scripts/vllm-serve.sh configs/<name>.args` runs the container in the foreground and stops it (`docker stop`) on SIGTERM, so `scripts/bench-run.sh` can start and stop it like the other servers. `configs/fp8-mtp4.args`: the source's compose flags with fp8 KV and util 0.95 (the FP8 baseline), `--served-model-name bench`, no API key.

The container started and was loading (log in `runs/3-smoke/server.out`: fp8 KV, "Using KV cache scaling factor 1.0 for fp8_e4m3", LBNHC KV layout, no errors so far), but the waiting loop watched the PID of `setsid` started from an interactive shell, which forks and exits at once, so it reported "SERVER EXITED after 0 s" while the container kept running. The container was stopped by hand (`docker stop vllm-bench`) to end the session cleanly. **Next:** start it through `scripts/bench-run.sh` (a script, where `setsid` does not fork), e.g. `scripts/bench-run.sh runs/3-smoke fp8-mtp4 0 -- experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh experiments/2026-11-vllm-27b-tp2/configs/fp8-mtp4.args` (paths relative to where it runs), check `--default-chat-template-kwargs` is accepted by 0.30.0 (it is from the 0.32.0 compose), read the KV pool (source: 278,927 tokens), then the first requests. The client sends `cache_prompt` and `chat_template_kwargs`; vLLM sends no llama.cpp-style `timings`, so the prompt t/s and draft columns of `bench-summary.py` stay empty (TTFT and decode come from the client).
