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

## 0c. Disk, finished (2026-11-26)

The Basalt GSQ-RCO pack move finished: the source is gone from `~/models` (the script deletes it only after the checksums pass), and `sha256sum -c SHA256SUMS` in `/mnt/models/qwen3.8-flash-next-basalt-gsq-rco-iq3xxs` gives OK again (75 GB). `/mnt/models` 146 of 220 GB used.

Docker cleanup: `docker rm` of the two hello-world containers; `docker rmi local/llama.cpp:b12464-nccl local/nvbandwidth:cuda13.0.0 ghcr.io/ggml-org/llama.cpp:full-cuda ollama/ollama:0.32.14 nvidia/cuda:13.0.0-base-ubuntu24.04 hello-world:latest`; `docker builder prune -a -f` (15.8 GB). Images went from 97.4 to 65.9 GB on disk, and the build cache to zero. Kept: searxng (running), n8n with its volume, alpine, the two vLLM images.

## 1. Images (2026-11-26)

`docker pull vllm/vllm-openai@sha256:…` for both digests, tagged `local/vllm-openai:v0.30.0` (30.7 GB on disk, 809 s) and `local/vllm-openai:v0.32.0` (33.1 GB, 312 s).

## 2. Checkpoint (2026-11-26)

`scripts/hf-fetch.sh gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 0cc27958cefbbe231782ec8511de8c4eb5233348 ~/models/qwen3.8-27b-nvfp4-0cc2795 <the 15 files of the source's wget loop>`: 652 s; 4 LFS files OK by SHA-256 (the three safetensors, `tokenizer.json`), 11 small git files OK by their git blob SHA-1. `hf-fetch.sh` learned the git-blob check for this (it handled LFS files only); `SHA256SUMS` has the SHA-256 of all 15.

## 3. Smoke test, first try (2026-11-26): not finished

New: `scripts/vllm-serve.sh configs/<name>.args` runs the container in the foreground and stops it (`docker stop`) on SIGTERM, so `scripts/bench-run.sh` can start and stop it like the other servers. `configs/fp8-mtp4.args`: the source's compose flags with fp8 KV and util 0.95 (the FP8 baseline), `--served-model-name bench`, no API key.

The container started and was loading (log in `runs/3-smoke/server.out`: fp8 KV, "Using KV cache scaling factor 1.0 for fp8_e4m3", LBNHC KV layout, no errors so far), but the waiting loop watched the PID of `setsid` started from an interactive shell, which forks and exits at once, so it reported "SERVER EXITED after 0 s" while the container kept running. The container was stopped by hand (`docker stop vllm-bench`) to end the session cleanly. **Next:** start it through `scripts/bench-run.sh` (a script, where `setsid` does not fork), e.g. `scripts/bench-run.sh runs/3-smoke fp8-mtp4 0 -- experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh experiments/2026-11-vllm-27b-tp2/configs/fp8-mtp4.args` (paths relative to where it runs), check `--default-chat-template-kwargs` is accepted by 0.30.0 (it is from the 0.32.0 compose), read the KV pool (source: 278,927 tokens), then the first requests. The client sends `cache_prompt` and `chat_template_kwargs`; vLLM sends no llama.cpp-style `timings`, so the prompt t/s and draft columns of `bench-summary.py` stay empty (TTFT and decode come from the client).

## 3. Smoke test (2026-11-26)

Kernel 7.0.0-44 since the reboot at 12:50 (installed by unattended-upgrades at 08:28; `docs/stack.md`); driver 580.178.04, both cards at 180 W.

```
scripts/bench-run.sh experiments/2026-11-vllm-27b-tp2/runs/3-smoke fp8-mtp4 1 -- \
  experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh experiments/2026-11-vllm-27b-tp2/configs/fp8-mtp4.args
```

Run from a script this time, so `setsid` does not fork and the waiter watches the right PID. Files: `runs/3-smoke/fp8-mtp4.*` (`.console.txt` is the script's output), requests in `results.jsonl`.

| | Expected (source) | Actual |
|---|---|---|
| Starts with the source's flags on 0.30.0 | yes | yes; `--default-chat-template-kwargs` (from the 0.32.0 compose) accepted |
| KV pool, fp8, MTP k=4, util 0.95 | 278,927 tokens | **273,333** (−2.0 %); "Available KV cache memory: 4.7 GiB" per card; vLLM notes util 0.95 is effectively 0.9245 once CUDA graph memory is profiled |
| Weights | — | 8.91 GiB per card, 7.1 s (the IPC weight cache is absent, so it falls back to disk) |
| Attention backend | — | FlashInfer |
| CUDA graphs | PIECEWISE | 7 sizes in 21 s, then 7 for the drafter in 1 s; engine init 72.9 s, of which compile 22.6 s |
| Ready | — | 119.1 s after start (page cache cold after the reboot) |
| Peak VRAM | — | 15,790 MiB on each card |
| MTP | acceptance > 0 | yes: prose window mean acceptance length 2.27 (per position 0.68 / 0.36 / 0.16 / 0.07), JSON window 4.02 (0.91 / 0.80 / 0.68 / 0.64) |

First numbers (shared client: thinking off, 256 tokens, about 4.3K-token prompts, one rep after a warm-up; not the reproduction mode yet):

| Prompt | TTFT | Decode |
|---|---|---|
| prose | 0.302 s | 62.95 t/s |
| structured (JSON) | 0.340 s | 114.41 t/s |

In range of the source's k=4 numbers (78.3 prose, 124.9 structured), which were measured with thinking on and 800 tokens; step 4 does both modes.

**A stop problem:** after the run the container was still up. `bench-run.sh` sent TERM to the process group at 12:57:04; vLLM began its shutdown ("send sigterm to process EngineCore") and hung there, still logging stats. `vllm-serve.sh` ran `docker stop -t 60`, but `bench-run.sh` sends KILL to the group after 30 s, which killed `docker stop` too. Stopped by hand (`docker stop -t 30 vllm-bench`, 19 s). Fix: `vllm-serve.sh` now uses `docker stop -t 15`, so Docker SIGKILLs a hung vLLM well inside the 30 s.

Check of the fix: `scripts/bench-run.sh experiments/2026-11-vllm-27b-tp2/runs/3-smoke stop-check 0 -- …` (start, stop, no requests). The shutdown hung at the same line again ("send sigterm to process EngineCore"); `docker stop -t 15` ended it, `bench-run.sh` returned after 97 s in total, no container left, both GPUs back at 2 MiB. Peak VRAM 15,480 MiB per card without requests. The hang is vLLM's own (both times); with the wrapper it now costs 15 s per stop.

**Next:** step 4, the phase A baseline (MTP off and k=4; shared clients and the reproduction mode; TTFT at about 0.5K, 4K and 32K).
