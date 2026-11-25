# Journal: 2026-11-strata

Every step in order: why, the exact command, expected vs actual. Mistakes and dead ends stay in. Plan, pinned versions and method: [`README.md`](README.md).

## 0. Plan (2026-11-25)

Plan written from the project's documentation, the Hugging Face file histories and two videos (sources in the README). No commands run yet.

## Decisions (2026-11-25)

- **Same weights as Basalt:** ISTA-DASLab GSQ-RCO IQ3_XXS (README, "Why this model file"). The UD files are out: Strata pins Unsloth's pre-3.1 upload and Basalt's packs come from the 3.1 re-upload of 2026-11-04.
- **Source build at the v0.2.21 tag; `setup.sh` and `install.sh` are read, not run:** they can install packages through apt.
- **Downloads by revision hash, verified against the Hub's LFS SHA-256** with `scripts/hf-fetch.sh`. It reads the files back with O_DIRECT, so the check sees the disk, not the page cache the download just filled.
- **Host stack unchanged** (kernel 7.0.0-43, driver 580.178.04, CUDA 13.0). Strata warns against CUDA 13.2.0/13.2.1 for sm_120, so the CUDA upgrade for the vLLM experiment waits until this and `2026-11-basalt` are done.

## 1. Hugging Face CLI (2026-11-25)

**Why:** `hf download` with a pinned revision. Nothing of it was on the host.

**Command** (formula, user shell):

```bash
python3 -m venv ~/venvs/hf
~/venvs/hf/bin/pip install huggingface_hub
~/venvs/hf/bin/hf version
```

**Expected:** a version line.

**Actual:** `version: 2.4.0`. The venv has `huggingface_hub==2.4.0` and `hf-xet==1.8.0` on Python 3.12.3.

**Verdict:** as expected. The 2026-11-24 note in `2026-11-llamacpp-flags` used `pipx install huggingface_hub`, but `hf` was not on the host any more. A venv at a fixed path makes the version visible with `pip freeze`.

## 2. Model (2026-11-25)

### 2a. Archive the previous model

**Why:** the batiai IQ3_XXS of `2026-11-llamacpp-flags` is not used here. Rule in `CLAUDE.md`: unused models go to the archive, not the bin.

**Command:** `scripts/move-model.sh ~/models/qwen3.8-flash-next-iq3xxs /mnt/models`

**Expected:** both shards copied, checksums OK from the SATA disk, the source removed.

**Actual:** copy 1,053 s for 84,960,783,842 bytes (about 81 MB/s: the GSQ-RCO download and the Strata build ran at the same time; the first archive move, DeepSeek, ran at about 250 MB/s alone). Both shards OK, verification 262 s. Free afterwards: NVMe 317 GB (with the download in progress), archive 59 GB.

**Verdict:** as expected.

### 2b. Download at the pinned revision

**Expected sizes and SHA-256** (Hub API, `paths-info` at `ed59f92082b1e93c0e96d60a8b11aab089b52f09`; both files last changed in `fb6d866`):

| File | Bytes | SHA-256 |
|---|---|---|
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` | 47,039,860,096 | `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` | 28,800,138,432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

**Command:**

```bash
scripts/hf-fetch.sh ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF ed59f92082b1e93c0e96d60a8b11aab089b52f09 \
  ~/models/qwen3.8-flash-next-gsq-rco-iq3xxs \
  IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
```

**Expected:** both files with the sizes and SHA-256 above, read back from the disk.

**Actual:** download 2,329 s (75.8 GB, about 33 MB/s, while the archive move, the Strata build and the MTP fetch also ran). Both files OK; the O_DIRECT check took 48 s. `REVISION`: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF ed59f92082b1e93c0e96d60a8b11aab089b52f09`.

**Verdict:** as expected.

## 3. Build (2026-11-25)

**Why:** a known engine from a known commit, built for these cards, without running setup's installer.

**Reading `setup.sh` and `terminal/setup.py` first** (v0.2.21; nothing run):
- `setup.sh` only finds a Python, offers `sudo apt-get install python3 python3-venv python3-pip` when none is found, makes `terminal/.venv` and hands over to `terminal/setup.py` (12,580 lines).
- `setup.py` pins `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, the same revision as here. Its `ISTA_FILES` table has the same bytes and SHA-256 for both IQ3_XXS shards as step 2. Its catalog entry: `"IQ3_XXS": … "download_gb": 75.8, "ram_gb": 60, "arena_gb": 42.9`.
- `build_engine()` for NVIDIA with CUDA 13: `cmake -G Ninja … -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=OFF -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_Q6K_EXPERTS=ON -DCMAKE_CUDA_ARCHITECTURES=<cards> -DCMAKE_CUDA_COMPILER=<nvcc> -DCUDAToolkit_ROOT=<toolkit>`, target `strata`, `-j` half the cores. The quick command in `docs/SOURCE_INSTALL.md` has fewer flags (no Ninja, no `CUDAToolkit_ROOT`); setup's set is used here.
- Its changelog line for engine 0.1.42 confirms the video: "`STRATA_ROUTE_TAIL_SKIP=7` default on CUDA (+13-16% decode)".

**Command:** `scripts/build-strata.sh v0.2.21` (from these flags; it reuses the clone at `~/strata/v0.2.21/src` after checking it is exactly the tag and clean).

**Expected:** a CUDA build for sm_120, engine version 0.1.67, linked against CUDA 13.0's cudart and cuBLAS; `build-info.txt`.

**Actual** (`~/strata/v0.2.21/build-info.txt`):
- Source `v0.2.21` = `d46f5f268f8ed797e83a5c63a69bffbd7a581cf6`, engine `project(strata VERSION 0.1.67)`; `strata --version`: `strata 0.1.67`.
- Configure: "CUDA 13.0.88 … arch 120", "Blackwell mxf4nvf4 + mxf8f6f4 prompt & hardware FP4 decode kernels enabled (120a-real)", "no ggml in this build (own CPU pool, no MMQ, tests off)". `STRATA_BUILD_SPLIT=OFF` (the separate split-every-layer engine; the normal layer split is in `strata`), `STRATA_NATIVE_EXPERTS=ON`, `STRATA_PORTABLE=OFF`.
- Build: 153 s with `-j 14`. Warnings: only nvcc's "loop is not reachable" (#128-D), 220 times in four files.
- Binary `build/strata` (73 MB): 56 kernels for `sm_120` and 4 for `sm_120a`; links `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13` from `/usr/local/cuda-13.0/lib64`.
- No `libstrata_kq.so`: that library is the K-quant prompt path for Unsloth's UD-Q4/5/6 files, and setup copies it only when it exists. IQ3_XXS does not use it.
- Venv `terminal/.venv`: Python 3.12.3 and the pins of `terminal/requirements.txt`.

**Verdict:** as expected.

## 4. Model files for the engine (2026-11-25)

What setup does for this model (step 6 of `setup.py`, not the Q2_0, NVFP4 or low-RAM branches):

1. `iq_pack.py --gguf <shard 1> --out <pack>`: a native pack, "every tensor as the GGUF stores it; the experts are read from the GGUF at start". No `--compat-bf16` for the `qwen` family.
2. The MTP draft layer from Qwen's BF16 checkpoint (`Qwen/Qwen3.8-Flash-Next` pinned at `de4b8e4d43b917e7706784d8bb445c9af86a3540`), only its MTP tensors by HTTP range requests, each checked against SHA-256: `mtp_fetch.py fetch`, `mtp_pack.py --experts q2_0`, `mtp_rt.py`.
3. The draft vocabulary: `terminal/data/draft_vocab.bin` (setup's default subset `cjk`) copied to `mtp/rt/draft_vocab.bin`.
4. Engine arguments: `--pack <pack> --native <shard 1> --ple-gguf <shard 2>` (two shards), plus what `strata_control/setup_config.py` adds.

### 4a. MTP draft layer

**Command** (in `~/strata/v0.2.21/src/terminal`, data in `~/strata/v0.2.21/data/mtp`):

```bash
.venv/bin/python tools/mtp_fetch.py fetch --out ~/strata/v0.2.21/data/mtp
.venv/bin/python tools/mtp_fetch.py verify --out ~/strata/v0.2.21/data/mtp
.venv/bin/python tools/mtp_pack.py --src ~/strata/v0.2.21/data/mtp --experts q2_0 --out ~/strata/v0.2.21/data/mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf ~/strata/v0.2.21/data/mtp/mtp-q2_0.gguf --out ~/strata/v0.2.21/data/mtp/rt
cp data/draft_vocab.bin ~/strata/v0.2.21/data/mtp/rt/draft_vocab.bin
```

**Expected:** only the MTP tensors fetched, each matching the pinned checkpoint's SHA-256 (`verify` exits 0; 3 = a bad tensor); a packed draft layer in `rt/`.

**Actual:** fetch 556 s (sharing the line with the 75.8 GB model download), `verify` exit 0. `mtp_pack.py` wrote `mtp-q2_0.gguf` with 0.889 GB of tensor data (experts as Q2_0, attention projections as Q8_0, norms F32). `rt/`: `experts.bin` 708 MB, `dense.bin` 116 MB, `dense.txt`, `draft_vocab.bin` 425 KB. The whole `data/mtp` folder is 6.5 GB, most of it the raw BF16 tensors from the fetch, which the engine does not read.

**Verdict:** as expected. The raw tensors stay until the draft layer has been seen working in step 8.

### 4b. Pack

**Command** (in `~/strata/v0.2.21/src/terminal`):

```bash
.venv/bin/python tools/iq_pack.py --gguf ~/models/qwen3.8-flash-next-gsq-rco-iq3xxs/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --out ~/strata/v0.2.21/data/packs/iq3_xxs
```

**Expected:** a native pack (index, dense weights, tokenizer); the experts stay in the GGUF.

**Actual:** 2 s. "index.txt: 1079 tensors, 302 served natively, 0 in extra.bin, arena 1.43 GiB"; tokenizer vocab 248,320, merges 247,587, pre `qwen35`, EOS 248046. Files: `dense.bin` 1.54 GB, `index.txt`, `native_experts.txt`, `conversions.json`, `tokenizer/`.

**Verdict:** as expected.

## 5. Server configs (2026-11-25)

**Why:** the run configs setup would write, without running setup, and one per drafting setting.

From `strata_control/setup_config.py` and `setup.py` (v0.2.21) for this machine and model:
- Model arguments: `--pack <pack> --native <shard 1> --ple-gguf <shard 2>` (two shards: the table's shard named).
- Always: `--expert-profile terminal/data/expert-profile.bin --expert-cache auto --prefill auto`, and with the draft layer `--spec 4 --spec-min-p 0.5 --mtp <rt>`.
- `--kv int8` above an 8K context (`kv = (a.kv or "int8") if ctx > 8192`).
- No `--ple-io ram`: that needs RAM of `ram_gb` (60) + the table (28.8) + 16 GB.
- KV streaming (`--kv-resident 32768`) only from a 64K context; at 128K it needs 54.7 GB of RAM by setup's own function, which this PC has (62.5 GB as setup counts it). Used in step 12.
- `--pool-workers 13`, from setup's own function on this CPU: "hybrid CPU (8 performance + 12 efficiency cores): 13 CPU expert workers - the performance cores and half of the efficiency cores".
- `STRATA_ROUTE_TAIL_SKIP`: setup writes `0` only for Unsloth files, but `serve/server.py` sets an unset value to `0` at every engine start (since 0.2.17). Written explicitly here anyway.

Drafting has three settings (engine `--help` and `strata_control/draft_layer.py`):

| Config | Arguments | What drafts |
|---|---|---|
| `strata-1gpu-nodraft.json` | `--suffix-draft 0 --lookup-max-t 0`, no `--mtp` | nothing ("0 = off" for both in `--help`) |
| `strata-1gpu-lookup.json` | `--spec 4`, no `--mtp` | prompt lookup only: what setup writes with the draft layer off |
| `strata-1gpu-mtp.json` | `--spec 4 --spec-min-p 0.5 --mtp <rt>` | the MTP draft layer and prompt lookup: setup's default |

All three: one GPU (`"gpu": 0`), `--max-context 32768`, `--kv int8`, `--pool-workers 13`. Files in `configs/`.

## 6. Smoke test, one GPU (2026-11-25)

**Why:** the engine starts from these files, answers, and its log shows the settings that matter before any measurement.

**Command** (in `~/strata/v0.2.21/src/terminal`):

```bash
.venv/bin/python serve/server.py --engine strata --config ~/ai-homelab-v2/experiments/2026-11-strata/configs/strata-1gpu-nodraft.json --port 8080 --ready-file ~/strata/v0.2.21/logs/ready
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"What is the capital of Austria? Answer in one short sentence."}],"max_tokens":200,"temperature":0}'
```

### 6a. First start: refused

**Actual:** the engine stopped at start with exit code 2: "`…/packs/iq3_xxs is a native (IQ) pack: it needs --native SHARD1, --spec T (T >= 2) and --prefill CHUNK`" (`runs/6-smoke/engine-attempt1.log`).

**Cause:** with a native pack the engine always decodes through verify windows, so `--spec` is required even with no drafter. "No drafting" is therefore `--spec 4` with nothing that proposes tokens: no `--mtp`, `--suffix-draft 0`, `--lookup-max-t 0`. Fixed in `configs/strata-1gpu-nodraft.json`. This also answers the plan's question about `--spec 0`: not possible with this pack.

### 6b. Second start: up in 20 s

**Actual** (`runs/6-smoke/engine.log`, `server.out`):
- **sm_120 defaults:** "12 speed switches on by default … HC_UPMIX Q80_ROWS1 EXP_MODE SIDE_MIXER DEQUANT_SIDE QFUSE LFUSE PLE_COMPACT PLE_COPY_STREAM PREFILL_CPU_SHARE_NT PREFILL_NVFP4 ADAPT_FETCH". `ADAPT_FETCH` "is on because this card has 16 GiB of memory or less, it does not give bit-for-bit the same text" (`STRATA_ADAPT_FETCH=0` turns it off; for the quality check in step 11).
- **Memory guard:** limit 62.5 GiB, available 59.2 GiB. PCIe probe: 20.4 GB/s host to device.
- **Pack and weights:** native pack, 1,466 MiB of dense weights in 0.2 s; 300 native projection matrices (1,781 MiB); n-gram table read in place from shard 2 ("PLE direct read probe: 110997 rows/s (fine)"); an "experimental native Q5_K head".
- **Expert arena:** 39.97 GiB at 4.31 GiB/s, read through the file cache (the files were still partly cached from the checksum check), "cudaHostRegister PORTABLE ok": pinned completely, no "could not be page-locked". No huge pages (`vm.nr_hugepages=0`, THP not requested for a 39 GiB arena).
- **CPU:** 13 pool workers on logical CPUs 2,4,…,14 (P-cores, one per physical core) and 16–21 (E-cores), the host thread on CPU 0.
- **Expert cache on GPU 0:** 5,978 of 24,576 slots (24.3 %), 9.67 GiB of VRAM, pre-filled from the shipped profile. The prompt path borrows 2,640 of those slots (4.23 GiB) while a prompt is read. 870 MiB of VRAM free with everything loaded (15,040 MiB used).
- **Adaptive cache:** "mid-coverage expert cache (5978 of 24576 slots, 24.3%): adaptive defaults every 4 -> 2, swaps 96 -> 192, decay 0.70 -> 0.92".
- **CPU prompt share:** on for prompt chunks up to 4,864 tokens ("answers can differ slightly … mean KL ~0.004 below 1024 tokens").
- **Conversation cache:** 5,632 MiB of RAM for up to 4 parked conversations, plus an SSD tier of up to 32 GiB in `data/conversation-spill/`. **It reuses a repeated prompt, so the speed runs turn it off** (`--conversation-cache-mib 0`); otherwise the second request with the same prompt would not read it again.
- RAM after load: 47 GB used, 14 GB available.

**The request:**
- Answer: "The capital of Austria is Vienna.", `finish_reason` `stop`. The model thinks first: 34 of the 43 completion tokens are reasoning.
- Server timings: 65 prompt tokens in 403 ms (161.2 tok/s), 43 generated in 841 ms (**51.1 tok/s**); "drafts accepted 0 of 0", so drafting is really off.
- "decode expert cache hit rate: 68.5% (14002 hits / 20433 lookups); 207 more read by the GPU over PCIe".
- The response has llama.cpp-style `timings` (`prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`, `draft_n`, `draft_n_accepted`), which the client in step 7 records next to its own clock.

**Route tail skip:** the engine log says nothing about it when it is off. `serve/server.py` sets an unset value to 0 (line 4034 comment: "the route tail skip is Off unless the person turned it on"), and the config sets `0` explicitly. Whether the log names it when it is on is checked in step 9.

**Stop:** SIGTERM to the server: "[strata] stopping … stopped"; RAM and VRAM back to idle (2 GB used, 2 MiB per GPU).

**Verdict:** works after the `--spec` fix. For the measurement configs: `--conversation-cache-mib 0`.

**Configs after step 6:** all three get `--conversation-cache-mib 0` (no parked conversations) and `--prompt-cache 0` (`--help`: "keep N conversation checkpoints between requests (default 6 …); 0 = read every prompt from the start"), so every measured request reads its whole prompt. The four-turn job in step 12 runs with both at their defaults, as an agent would use the server.

## 7. Baseline: llama.cpp on the same GGUF (2026-11-25)

**Why:** the reference Strata must beat: llama.cpp v1.0.0 (build 12359) on exactly these files, with the same client and prompts.

### 7a. Client and runner

New shared scripts (repository `scripts/`, used by this experiment and `2026-11-basalt`):
- `oai-bench.py`: streamed `/v1/chat/completions`; the lighthouse document of `2026-11-llamacpp-flags` (4,315 prompt tokens with the prose instruction, 4,335 with the structured one) and an instruction that asks for far more than 256 tokens. Prose: "Continue this story … at least 1,500 words". Structured: "a JSON array of 60 objects" with six fixed keys. Temperature 0, seed 42, `max_tokens` 256, thinking off (`chat_template_kwargs.enable_thinking = false`). One warm-up and 5 measured requests per kind. TTFT and decode t/s on the client; the server's own `timings` kept as they come.
- `bench-run.sh`: starts the server in its own process group, samples `nvidia-smi` every 0.5 s and RAM, the server's RSS and NVMe reads every 1 s, waits for `/health` (Strata: also `"loaded"` not false), runs the client, stops the whole group.
- `bench-summary.py`: the table below from `results.jsonl` and the sampler files.

Base options of every llama.cpp run, as in `2026-11-llamacpp-flags`: `-np 1 --cache-ram 0 -t 8 -c 8192 -lv 4`.

### 7b. First run: invalid (prompt reuse), and sudo

**Command:** `scripts/bench-run.sh runs/7-llamacpp 1gpu-fit 5 -- env CUDA_VISIBLE_DEVICES=0 llama-server -m <shard 1> <base>`

**Actual:**
- From the second request on, TTFT was 0.13 s and the server's "prompt" speed about 31 t/s: llama-server reused the document from the slot's KV cache and read only the last tokens. `--cache-ram 0` does not stop that; the request field `cache_prompt: false` does (the client of `2026-11-llamacpp-flags` sent it to `/completion`). Decode was not affected (about 26.5 t/s), but every prompt number after the warm-up was wrong. **Fix:** `oai-bench.py` sends `"cache_prompt": false`. The files are kept as `runs/7-llamacpp/invalid-cached-*`.
- The runner's first version used sudo when `sudo -n true` succeeded, for a cold page cache and CPU power (RAPL), and it did succeed in Claude Code's shell through a cached sudo ticket of the user's. Claude Code is denied sudo by the managed settings, so this run dropped the page cache and read RAPL as root against that rule. **Fix:** sudo only with `BENCH_SUDO=1`, set by the user in their own shell. The runs below have no page cache drop and no CPU power; their load times come from a partly warm page cache and are not comparable with each other.

### 7c. The four runs

**Commands** (in `experiments/2026-11-strata`, `<base>` as above, `M` = shard 1):

```bash
scripts/bench-run.sh runs/7-llamacpp 1gpu-fit 5 -- env CUDA_VISIBLE_DEVICES=0 llama-server -m $M <base>
scripts/bench-run.sh runs/7-llamacpp 1gpu-best 5 -- env CUDA_VISIBLE_DEVICES=0 llama-server -m $M <base> -cmoe --moe-cache-mib 9472 --moe-cache-cpu-share 0 -lm dio -b 4096 -ub 4096
scripts/bench-run.sh runs/7-llamacpp 2gpu-fit 5 -- llama-server -m $M <base>
scripts/bench-run.sh runs/7-llamacpp 2gpu-best 5 -- llama-server -m $M <base> -cmoe --moe-cache-mib 23552 --moe-cache-cpu-share 0 -lm dio -b 4096 -ub 4096
```

"best" is the final configuration of `2026-11-llamacpp-flags`. Its cache (23,552 MiB) was sized for two GPUs. On one GPU it had to be smaller; two load-only starts (`runs/7-llamacpp/probe/`):
- 11,776 MiB (half of it): "projected to use 17411 MiB of device memory vs. 15710 MiB of free device memory", then `cudaMalloc failed: out of memory` for the MoE cache. One GPU also holds all the dense weights (3,665 MiB) and the larger compute buffer.
- 9,472 MiB (11,776 minus the 1,701 MiB overshoot and a 512 MiB margin): loads, "CUDA0 MoE cache size = 9468.31 MiB for 40925.00 MiB of host experts", peak 15,356 MiB of VRAM.

On two GPUs the cache was 11,918 MiB (CUDA0) and 11,624 MiB (CUDA1).

**Expected:** "best" well above `--fit` on both counts; two GPUs above one.

**Actual** (`runs/7-llamacpp/summary.md`; every request read its whole prompt, all ran to 256 tokens, the same text on every repetition):

| Run | Kind | TTFT s | Prompt t/s | Decode t/s | VRAM MiB (GPU0 / GPU1) | RSS GiB |
|---|---|---|---|---|---|---|
| 1 GPU, `--fit` | prose | 15.53 ± 0.03 | 278.0 ± 0.6 | 28.0 ± 0.7 | 14,882 / – | 44.4 |
| 1 GPU, `--fit` | structured | 15.69 ± 0.03 | 276.5 ± 0.5 | 27.3 ± 0.7 | | |
| 1 GPU, best | prose | 4.96 ± 0.01 | 871.7 ± 0.9 | 45.2 ± 0.1 | 15,538 / – | 41.5 |
| 1 GPU, best | structured | 5.13 ± 0.00 | 846.7 ± 0.4 | 42.4 ± 0.2 | | |
| 2 GPUs, `--fit` | prose | 10.20 ± 0.04 | 423.5 ± 1.5 | 36.0 ± 0.6 | 14,688 / 14,866 | 44.5 |
| 2 GPUs, `--fit` | structured | 10.30 ± 0.02 | 421.1 ± 0.8 | 35.9 ± 0.8 | | |
| 2 GPUs, best | prose | 5.16 ± 0.01 | 837.6 ± 1.6 | 53.1 ± 0.1 | 15,784 / 15,046 | 41.5 |
| 2 GPUs, best | structured | 5.16 ± 0.01 | 841.9 ± 1.1 | 52.9 ± 0.1 | | |

**Verdict:** as expected in direction:
- "best" against `--fit`: prompt 3.1× on one GPU and 2.0× on two, decode +61 % (one GPU, prose) and +48 % (two).
- The second GPU adds 17 % decode with "best" (45.2 → 53.1) and does not help prompts (the cache holds more experts, the prompt path does not use it more).
- The same configuration in `2026-11-llamacpp-flags` gave 803 / 59.5 t/s on two GPUs, with a different IQ3_XXS file (batiai), the native `/completion` endpoint without a chat template, and 10 requests. These numbers are not mixed with those.
- Structured output is not faster here: llama.cpp drafts nothing in these runs.

## 8. Strata, one GPU (2026-11-25)

**Why:** Strata against llama.cpp on the same files and prompts, first exact (no drafting), then with each drafter.

**Commands** (in `experiments/2026-11-strata`, `S=~/strata/v0.2.21`; configs from step 5 with both caches off):

```bash
scripts/bench-run.sh runs/8-strata-1gpu nodraft 5 -- env -C $S/src/terminal .venv/bin/python serve/server.py --engine strata --config configs/strata-1gpu-nodraft.json --port 8080
scripts/bench-run.sh runs/8-strata-1gpu lookup 5 -- … --config configs/strata-1gpu-lookup.json …
scripts/bench-run.sh runs/8-strata-1gpu mtp 5 -- … --config configs/strata-1gpu-mtp.json …
```

The engine log of each run is copied to `runs/8-strata-1gpu/<run>.engine.log`.

**Expected:** without drafting, near or above llama.cpp's one-GPU "best" (45.2 t/s decode); the drafters add little on prose and much on JSON.

**Actual** (`runs/8-strata-1gpu/summary.md`):

| Run | Kind | TTFT s | Prompt t/s | Decode t/s | Drafts accepted | Decode cache hits |
|---|---|---|---|---|---|---|
| no drafting | prose | 2.45 ± 0.02 | 1,778.7 ± 13.1 | 62.8 ± 0.1 | – | 90–92 % |
| no drafting | structured | 2.50 ± 0.00 | 1,751.8 ± 2.3 | 62.9 ± 0.3 | – | |
| lookup | prose | 2.45 ± 0.01 | 1,774.9 ± 10.5 | 62.7 ± 0.5 | 29 of 60 | 89–91 % |
| lookup | structured | 2.50 ± 0.00 | 1,749.0 ± 2.7 | 77.4 ± 2.5 | 418 of 786 | |
| MTP | prose | 2.51 ± 0.01 | 1,737.5 ± 4.2 | 77.8 ± 0.9 | 575 of 1,076 (53 %) | 86–89 % |
| MTP | structured | 2.54 ± 0.00 | 1,722.5 ± 3.9 | 118.2 ± 4.4 | 889 of 1,047 (85 %) | |

- Every request read its whole prompt ("0 reused + 4315 read", `timings.cache_n` 0), all ran to 256 tokens.
- Peak VRAM 15,040–15,152 MiB on GPU 0; RSS of the server's process group 42.8–43.0 GiB; lowest MemAvailable about 17 GB.
- The MTP draft layer takes 840 MiB of VRAM ("experts 675, dense 111") plus a 178 MiB draft head, so the expert cache shrinks from 5,978 to 5,349 slots and the hit rate drops by 2–3 points.
- The answers start word for word as llama.cpp's (the same first 150 characters, prose and JSON), but unlike llama.cpp's five identical repetitions, Strata's 256-token texts differ between repetitions (different SHA-256 for most requests). Strata documents why: the adaptive expert cache, `ADAPT_FETCH` (on for 16 GiB cards) and the CPU prompt share change the last bits of the routed output. Step 11 checks whether that changes the answers' content.

**Against llama.cpp on one GPU** (step 7, "best": 871.7 prompt / 45.2 and 42.4 decode):
- Prompt: **2.0×** (1,779 against 872 t/s); TTFT 2.45 against 4.96 s.
- Decode, exact (no drafting, nothing skipped): **+39 %** prose (62.8 against 45.2), **+48 %** JSON (62.9 against 42.4).
- Decode with MTP: prose 77.8 (1.72× llama.cpp), JSON 118.2 (2.79×). The drafted tokens are verified, so the text is still the model's own; only `relaxed`-type options would change that, and none is on.

**Verdict:** Strata is faster than llama.cpp's best configuration on every count, with the whole model and no drafting. Lookup drafting helps only output that repeats itself (JSON). The MTP drafter adds 24 % on prose and 88 % on JSON over no drafting; the video's point holds: structured-output headlines are far above prose.

## 9. Strata, the route tail skip on (2026-11-25)

**Why:** how much speed the skip buys on this machine, and how the log shows it. It was the default on 16 GB NVIDIA cards from engine 0.1.42 to 0.1.62.

**Configs:** `strata-1gpu-mtp-skip7.json` and `strata-1gpu-nodraft-skip7.json`, the step 8 configs with `"STRATA_ROUTE_TAIL_SKIP": "7"`. The no-drafting one is not in the plan; it shows the skip without a drafter.

**Commands:** `scripts/bench-run.sh runs/9-strata-skip {mtp-skip7,nodraft-skip7} 5 -- env -C $S/src/terminal .venv/bin/python serve/server.py --engine strata --config configs/strata-1gpu-<run>.json --port 8080`

**The engine source** (`engine/src/program/generate.cpp`, comment at line 14934): "STRATA_ROUTE_TAIL_SKIP: OFF BY DEFAULT on every backend (owner decision 2026-11-19, engine 0.1.63; 0.1.42 to 0.1.62 set …)", "STRATA_ROUTE_TAIL_SKIP=7 turns it on (KL 0.0126 against exact routing, decode x1.2…". With it on, the log says at start: "STRATA_ROUTE_TAIL_SKIP=7 is ON: a missed expert that every token of a verify window routes at rank 7 or lower is skipped (answers differ slightly)", and after each request "route tail skip: N missed experts skipped … since the start". With it off, the log says nothing, as in steps 6 and 8.

**Expected:** a decode gain of the size the source and the video give (×1.2; 80 against 66 tok/s on an RTX 5080), larger with MTP (longer verify windows give it more to skip).

**Actual** (`runs/9-strata-skip/summary.md`; against step 8 without the skip):

| Run | Kind | Decode t/s, skip off | Decode t/s, skip 7 | Gain | Experts skipped per request | Routed per request |
|---|---|---|---|---|---|---|
| no drafting | prose | 62.8 ± 0.1 | 62.7 ± 0.4 | 0 % | about 5,900 (5,300–10,100) | about 116,000 |
| no drafting | structured | 62.9 ± 0.3 | 63.8 ± 0.3 | +1 % | | |
| MTP | prose | 77.8 ± 0.9 | 81.7 ± 2.2 | +5 % | about 8,000 (6,700–11,600) | about 145,000 |
| MTP | structured | 118.2 ± 4.4 | 125.0 ± 3.7 | +6 % | | |

Prompt speed and TTFT unchanged (the skip is a decode path). The decode cache hit rate rises from 90–91 % to 94–96 % (no drafting) and from 86–89 % to 92 % (MTP), because skipped misses are no longer looked up.

**Verdict:** the skip leaves out about 5–6 % of all routed expert computations (about half of the misses) for 0–6 % more speed. On this machine a miss is cheap: 90 % of the routed experts are in the GPU cache, and the 13 CPU workers compute most misses while the GPU works, so leaving them out saves little time. The video's RTX 5080 had a 75 % hit rate, where misses cost more. The quality side (answers with and without the skip) is step 11. All further steps keep the skip off.

## 10. Strata, two GPUs (2026-11-25)

**Why:** what the second card adds with Strata's layer split, against llama.cpp's two-GPU "best" (837.6 prompt / 53.1 decode t/s).

**Configs:** `strata-2gpu-nodraft.json` and `strata-2gpu-mtp.json`: the step 8 configs with `"gpu": [0, 1]` and `"layer_split": "auto"` instead of `"gpu": 0`.

**Commands:** `scripts/bench-run.sh runs/10-strata-2gpu {nodraft,mtp} 5 -- env -C $S/src/terminal .venv/bin/python serve/server.py --engine strata --config configs/strata-2gpu-<run>.json --port 8080`

**The split** (`runs/10-strata-2gpu/nodraft.engine.log`):
- "layer split auto: K=26 - predicted 39.3 ms per decode window; the caches hold 14721 of 24576 profiled pairs (~98.5% of the routed mass), best of 46 placements".
- CUDA0 runs layers 0–25 with an expert cache of 8,187 slots (11.91 GiB); CUDA1 runs layers 26–47 and the output head with 6,311 slots (11.58 GiB). Each card keeps only its own layers' dense weights (CUDA0 "1499 MiB freed").
- PCIe probes: CUDA0 20.4 GB/s, CUDA1 19.4 GB/s ("miss cost x1.00" for both).
- The expert arena stays once in RAM: RSS 44.3 GiB, lowest MemAvailable 15.4 GB.

**Expected:** more decode than one GPU, since the caches hold 60 % of all experts instead of 24 %.

**Actual** (`runs/10-strata-2gpu/summary.md`):

| Run | Kind | Prompt t/s | Decode t/s | Against one GPU (step 8) | Decode cache hits |
|---|---|---|---|---|---|
| no drafting | prose | 1,878.3 ± 2.3 | 64.8 ± 0.2 | +3 % (62.8) | 96.8–97.4 % |
| no drafting | structured | 1,877.4 ± 0.3 | 64.8 ± 0.1 | +3 % (62.9) | |
| MTP | prose | 1,764.9 ± 1.2 | 86.2 ± 1.0 | +11 % (77.8) | 98.3–99.0 % |
| MTP | structured | 1,764.2 ± 0.4 | 132.4 ± 3.6 | +12 % (118.2) | |

Prompt +6 % without drafting (TTFT 2.32 against 2.45 s), +2 % with MTP. Peak VRAM 14,890–15,032 MiB per card.

**Verdict:**
- The hit rate went from about 90 to 97–99 %, but decode without drafting rose only 3 %. Decode is not limited by misses on this machine: a layer split runs the layers one card after the other, so a token still passes 48 layers in sequence, and the misses were already cheap (step 9). The second card mainly adds cache.
- With MTP the gain is larger (+11–12 %): verify windows route more experts per step, so more of them miss on one card.
- Against llama.cpp on two GPUs, exact against exact: decode +22 % (64.8 against 53.1), prompt 2.2× (1,878 against 838). With MTP: prose 1.62×, JSON 2.5×.
- For a single user on this machine the second card is worth 3–12 %. The 2× RTX 5060 Ti row on Basalt's site (161 t/s) used Basalt's peer mode, a different way to use the second card; `2026-11-basalt` measures it.

### 10b. Other ways to use the second card (2026-11-25)

**Why:** step 10 used the layer split. Strata's Main/Helper roles (`docs/MULTI_GPU.md`) are the names of that split's cards (Main runs the last stage, the head and the draft layer; of two identical cards the higher number is Main), so step 10 already ran Main = GPU 1, Helper = GPU 0. Two other ways are in the docs:
- **Helper caches** (`docs/SECOND_GPU.md`, "experimental"): CUDA0 runs every layer, CUDA1 only holds extra experts and computes their rows during decode; results return through pinned host buffers, no P2P needed. `--expert-cache-device1 auto --remote-expert-opt`. This is the closest to Basalt's peer mode.
- **Pipeline windows** (`--pipeline-windows 2`, opt-in on NVIDIA): on a layer split, the first card starts the next verify window while the last card still verifies this one.
- Not tried: the peer tier `--peer-device`, which needs NVLink or another peer-to-peer path; these cards have none (`docs/hardware.md`).

**Configs:**
- `strata-helper-{nodraft,mtp}.json`: the one-GPU configs without `"gpu"`, with `"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,1"` in `env` and `--expert-cache-device1 auto --remote-expert-opt`. Without `"gpu"` the server does not make a layer split: it treats two cards in `"gpu"` as a split unless `--peer-device` is set (`serve/server_parts/engines.py`, `layer_split_of`).
- `strata-2gpu-mtp-pipe2.json`: the step 10 MTP config with `--pipeline-windows 2`.

**Commands:** `scripts/bench-run.sh runs/10b-strata-2gpu-modes <run> 5 -- env -C $S/src/terminal .venv/bin/python serve/server.py --engine strata --config configs/strata-<run>.json --port 8080`

**Actual** (`runs/10b-strata-2gpu-modes/summary.md`):

| Run | Kind | Prompt t/s | Decode t/s | Against the layer split (step 10) | Against one GPU (step 8) |
|---|---|---|---|---|---|
| helper cache, no drafting | prose | 1,757.4 ± 13.4 | 66.9 ± 0.2 | +3 % (64.8) | +7 % (62.8) |
| helper cache, no drafting | structured | 1,741.6 ± 0.9 | 66.6 ± 0.4 | +3 % | +6 % |
| helper cache, MTP | prose | 1,721.1 ± 8.0 | 93.4 ± 3.5 | +8 % (86.2) | +20 % (77.8) |
| helper cache, MTP | structured | 1,715.9 ± 3.4 | 138.3 ± 1.3 | +4 % (132.4) | +17 % (118.2) |
| layer split, MTP, pipeline windows 2 | prose | 1,777.0 ± 0.7 | 90.6 ± 1.5 | +5 % (86.2) | +16 % |
| layer split, MTP, pipeline windows 2 | structured | 1,778.9 ± 0.5 | 159.2 ± 4.1 | **+20 %** (132.4) | **+35 %** |

- Helper cache: "CUDA1: 8854 additional experts, 14.44 GiB; results return through pinned host rows". Per request CUDA1 computed about 75,000–78,000 expert entries, two thirds of all routed ("81833 more read by the GPU over PCIe or from another GPU (66.6% of all 122880 routed)"), and the host waited 326–417 ms per request for it. The log: "STRATA_ADAPT_FETCH=2 (the default for this card) is not used in this run (helper GPUs)", "adaptive overlap: off: not with --remote-expert-opt". Prompts run on CUDA0 only (as the docs say), so prompt speed is the one-GPU speed.
- Pipeline windows: "272 MiB of CUDA0 kept out of the expert cache (the second verifier, the GDN snapshots)"; the split moved to layers 0–23 / 24–47. The log notes "--pipeline-windows 2 on a layer split of two stages without the draft layer on the first card (STRATA_MTP_STAGE0=1 asks for it): the draft chain then shares the last card with its window". Not tried with `STRATA_MTP_STAGE0=1`.
- All requests read their whole prompt and ran to 256 tokens.

**Verdict:**
- Exact (no drafting), the helper cache is the best way to use the second card here: 66.9 t/s, +7 % over one GPU and +3 % over the layer split, with one-GPU prompt speed.
- With MTP, the best mode depends on the output: prose is fastest with the helper cache (93.4), JSON with the pipelined layer split (159.2), where accepted windows are long and the guessed next window usually holds.
- The ranking of all two-card modes is within 3–20 % of each other; none doubles anything. The 161 t/s two-card row on Basalt's site is not comparable with any number here: Basalt's community decode figure averages chat, prose and structured output, on a different pack and version. `2026-11-basalt` measures Basalt with this client.

### 10c. Pipeline windows with the draft layer on the first card (2026-11-25)

**Why:** the 10b log said the draft layer stayed on the last card and named `STRATA_MTP_STAGE0=1` to move it; `docs/MULTI_GPU.md` says the first card is the intended place with the pipeline on ("its window has finished when the chain starts").

**Config:** `strata-2gpu-mtp-pipe2-stage0.json`: `strata-2gpu-mtp-pipe2.json` with `"STRATA_MTP_STAGE0": "1"` in `env`.

**Command:** `scripts/bench-run.sh runs/10b-strata-2gpu-modes 2gpu-mtp-pipe2-stage0 5 -- … --config configs/strata-2gpu-mtp-pipe2-stage0.json …`

**Expected:** at least as fast as 10b's pipeline run (90.6 prose / 159.2 JSON), the docs' intended layout.

**Actual:** the 10b note about the draft layer is gone; "two verifiers per stage, 2 stages overlap across windows (decode too …)"; split 0–22 / 23–47.

| Kind | Prompt t/s | Decode t/s | Draft layer on the last card (10b) |
|---|---|---|---|
| prose | 1,758.8 ± 1.3 | 77.9 ± 1.1 | 90.6 ± 1.5 |
| structured | 1,757.5 ± 0.8 | 156.5 ± 5.4 | 159.2 ± 4.1 |

**Verdict:** slower: prose −14 % (back to the one-GPU MTP speed, 77.8), JSON −2 % (within the spread). On these two equal cards the default placement (draft layer on the last card) is the better one. The ranking of 10b stands.

## 11. Quality check (2026-11-25)

**Why:** speed is only comparable if the answers are the model's. The question is not how good IQ3_XXS is, but whether an engine or a setting changes the answers.

**Client:** `scripts/quality-check.py` (new, shared): ten fixed prompts with a simple check each (17 × 23, the capital of Australia, the first ten primes, sorting five words, a train's journey time, the author of *Pride and Prejudice*, a German translation, a JSON object with fixed keys, a palindrome function, three sentences on why the sky is blue). Each prompt twice; non-streamed, temperature 0, seed 42, thinking off, no prompt reuse. `scripts/bench-run.sh` starts the server and runs it (`BENCH_CLIENT=scripts/quality-check.py`); `scripts/quality-compare.py` makes the table.

**Configurations** (one GPU each):
- `llamacpp-1gpu-best`: step 7's best one-GPU configuration, the reference.
- `strata-nodraft`: step 8's speed config (Strata's defaults, drafting off).
- `strata-det-nodraft`: Strata's recipe for byte-identical repeats (`docs/DETAILS.md`): `--adapt-swaps 0 --pcie-frac 0`, a fixed `--expert-cache 5978` (the count `auto` gave), `--prompt-cache 0`, `STRATA_IQ_MT_MIN=1`, `STRATA_PREFILL_CPU_SHARE=0`, and `STRATA_ADAPT_FETCH=0` (the sm_120 default that "does not give bit-for-bit the same text"). Config `strata-1gpu-det-nodraft.json`.
- `strata-det-mtp`: the same with the MTP drafter (`--expert-cache 5349`, what `auto` gave with the draft layer).
- `strata-det-skip7`: `strata-det-nodraft` with `STRATA_ROUTE_TAIL_SKIP=7`.

**Expected:** every answer correct; the deterministic configs identical across repetitions; the skip changes some longer answers.

**Actual** (`runs/11-quality/quality.md`; = same text as llama.cpp, ≠ different, ✓ check passed, ~ the two repetitions differ):

| prompt | llama.cpp | Strata defaults | Strata deterministic | + MTP | + skip 7 |
|---|---|---|---|---|---|
| arith, capital, primes, sort, time, author, german | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| json | = ✓ | ≠ ✓ ~ | ≠ ✓ | ≠ ✓ | ≠ ✓ |
| code | = ✓ | ≠ ✓ ~ | ≠ ✓ | = ✓ | = ✓ |
| explain | = ✓ | = ✓ | = ✓ | = ✓ | ≠ ✓ |

- All 100 answers pass their check.
- llama.cpp and the three deterministic Strata configs give the same text on both repetitions. Strata with its defaults does not: json and code each changed between the two repetitions (one of the two matched llama.cpp).
- What differs, and where:
  - json: the first tag `"archivist"` instead of `"archaeologist"` (Strata without drafting, with or without the skip); with MTP all three tags differ (`"astronomer", "novelist", "gardener"`). Still valid JSON with the required keys.
  - code: `cleaned = ''.join(c.lower() for c in s if c.isalnum())` instead of a list comprehension; both correct.
  - explain, with the skip only: from character 361 of 434, "some violet light is absorbed by the upper atmosphere, causing us to per…" instead of "some violet is absorbed by the upper atmosphere, resulting in the sky ap…".

**Verdict:**
- Strata gives the same answers as llama.cpp on short, factual prompts, and equally correct but not identical text on longer ones: the first different token comes where two continuations are nearly equally likely (an occupation in a list, a coding idiom).
- With Strata's defaults the same request can end differently from one run to the next. Strata documents the causes (adaptive cache, PCIe share, CPU prompt share, `ADAPT_FETCH`), and its recipe removes them: the deterministic configs repeat exactly.
- MTP drafting changes text here because it changes the expert cache size (5,349 against 5,978 slots: different experts computed on the GPU or the CPU, which round differently), not because drafted tokens are accepted unchecked.
- The route tail skip changed one of ten answers late in a sentence. Ten prompts cannot measure the size of its quality cost (Strata's own figure: KL 0.0126); they show it is not free and not dramatic at this file's precision.

## 12. Long context and the four-turn job (2026-11-25)

**Why:** prompt speed at the lengths an agent sends, decode with a long context behind it, and a job timed in seconds rather than tokens per second.

**Client:** `scripts/long-bench.py` (new, shared; started by `bench-run.sh` with `BENCH_CLIENT` and `LONG_MODE`). The document is Strata's own docs at v0.2.21 (`docs/*.md` in sorted order, 1,527,726 characters), cut to the target length, with one invented sentence in the middle: "The passphrase for the archive is obsidian-2824." Every long prompt ends with the question for it.

**Configuration:** the fastest exact setup of steps 8–10, the helper cache on two GPUs without drafting, with `--max-context 131072 --kv-resident 32768` (KV streaming: setup turns it on from 64K when RAM allows; at 128K it needs 54.7 GB by setup's function). Two configs:
- `strata-helper-nodraft-128k.json`: caches off, for the prompt timings.
- `strata-helper-nodraft-128k-cache.json`: Strata's default prompt and conversation caches, for the job.

### 12a. First attempt: a prompt too long for the context

The token count was estimated from the first 16,000 characters of the corpus (3.69 characters per token). The "32K" prompt had 36,771 tokens: the later docs have more code and tables. Its three requests are valid (TTFT 16.9–17.2 s, about 2,170 t/s, needle found) and kept as `runs/12-long/attempt1-*`. The 128K prompt was over 131,072 tokens and Strata answered 400 (it rejects, never truncates). **Fix:** the client now calibrates on 16 slices spread evenly over the document prefix itself, twice, and aims 5 % below the target; a 400 is recorded instead of ending the run.

### 12b. Long prompts

**Command:** `LONG_MODE=ttft BENCH_CLIENT=scripts/long-bench.py scripts/bench-run.sh runs/12-long strata-helper-nodraft-128k 3 -- … --config configs/strata-helper-nodraft-128k.json …`

**Actual** (3 requests each, the same prompt, nothing reused):

| Prompt | TTFT s | Prompt t/s (server) | Decode t/s | Needle |
|---|---|---|---|---|
| 32,804 tokens (105,425 characters) | 15.37–15.60 | 2,110–2,137 | 65.8–71.2 | 3 of 3 |
| 126,586 tokens (421,579 characters) | 61.47–61.53 | 2,059–2,064 | 68.4–72.2 | 3 of 3 |

The log: "KV streaming: 32768 of 131072 cells per QSA layer in VRAM, the K/V in 1.55 GiB of pinned RAM"; at 126K about 67 % of the KV block reads hit VRAM. Peak VRAM 15,042 / 15,342 MiB, RSS 46.0 GiB, lowest MemAvailable 14.7 GB.

### 12c. The four-turn job

**Command:** `LONG_MODE=job BENCH_CLIENT=scripts/long-bench.py scripts/bench-run.sh runs/12-long strata-helper-nodraft-128k-cache 3 -- … --config configs/strata-helper-nodraft-128k-cache.json …`

A 32,810-token document with the same needle, then four turns in one conversation: a five-bullet summary (up to 400 tokens), a list of `STRATA_` variables (400), a bash script (400), the passphrase (32). Each run starts with a random session tag, so it cannot reuse the previous run.

**Actual:**

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Whole job | 33.4 s | 32.8 s | 33.1 s |
| Turn 1 (summary): TTFT, tokens out | 15.70 s, 400 | 15.75 s, 400 | 15.70 s, 400 |
| Turns 2–4: TTFT | 0.36–0.38 s | 0.38–0.42 s | 0.38–0.40 s |
| Tokens reused in turns 2–4 | 33,209 / 33,650 / 33,942 | 33,211 / 33,623 / 33,914 | 33,211 / 33,650 / 33,941 |
| Needle | found | found | found |

Decode during the job: 62–73 t/s. The script turn wrote a plausible script, but for the Intel SYCL build (`ONEAPI_DEVICE_SELECTOR`, `build-sycl/strata`): the first matching example in the document, not this machine. That is the model's answer, not a speed issue.

**Verdict:**
- Prompt reading stays above 2,000 t/s up to 126K tokens; a 126K-token prompt takes 61.5 s before the first token (the vLLM video's 27B setup: 247 s for 249K).
- Decode does not slow down with a long context behind it (68–72 t/s at 126K against 66.9 at 4K), thanks to KV streaming.
- The needle came back in all nine long requests and all three jobs.
- In the job, the first turn (reading the document) is half of the total; with the prompt cache the later turns start in 0.4 s. Of the 33 s, about 16 s are prompt reading and about 17 s are generating 1,064–1,092 tokens.

## 13. Two GPUs, every mode (2026-11-25)

**Why:** the user's main question, added after step 12: the best way to run this model on both cards, and what two cards give over one, per use. Same client, prompts and model as before; caches off for prompt and decode timings, on for the job.

### 13a. llama.cpp on two GPUs at long context, and the job

**Config:** step 7's two-GPU "best" with `-c 131072`. At 131K the MoE cache of 23,552 MiB no longer fits: "projected to use 36517 MiB of device memory vs. 31417 MiB of free device memory", then out of memory for a 2,837 MiB compute buffer (the KV cache alone takes 1,536 + 384 MiB per card). With `--moe-cache-mib 17408` (23,552 minus the overshoot and 512 MiB per card) it loads: caches 8,825 / 8,566 MiB, compute buffers 2,837 / 2,524 MiB, peak VRAM 15,544 / 15,312 MiB (`runs/13a-llamacpp-long/probe/`).

**Commands:** `LONG_MODE=ttft` and `LONG_MODE=job` with `BENCH_CLIENT=scripts/long-bench.py scripts/bench-run.sh runs/13a-llamacpp-long … -- llama-server -m $M <base> -c 131072 -cmoe --moe-cache-mib 17408 --moe-cache-cpu-share 0 -lm dio -b 4096 -ub 4096` (`<base>` with `-c 8192` replaced).

**Actual** (`runs/13a-llamacpp-long/summary.md`): 32,804 tokens: TTFT 39.7 s (828 t/s); 126,574 tokens: 153.4 s (826 t/s); needle 6 of 6. The job: 65.3, 62.2, 65.1 s, the first turn 37.6–39.9 s, the later turns 0.5–0.7 s (llama-server reuses the slot's KV with `cache_prompt`); needle 3 of 3. Decode in the long-prompt runs covers only the 8-token answer and is too noisy to report.

### 13b. llama.cpp's other split modes

**Commands:** `scripts/bench-run.sh runs/13b-llamacpp-modes <label> 5 -- llama-server -m $M <base> … -sm {row,tensor}`.

**Actual:**
- `-sm row`, with or without the MoE cache: the model does not load ("device CUDA0 does not support split buffers").
- `-sm tensor` with the MoE cache: loads the weights, then "MoE cache requires a GPU backend". (Before that: "llama_params_fit is not implemented for SPLIT_MODE_TENSOR" and "enabling flash_attn since it is required for SPLIT_MODE_TENSOR".)
- `-sm tensor` with plain `-cmoe` (experts in RAM, no cache) runs, against `-sm layer` with the same options (`runs/13b-llamacpp-modes/summary.md`):

| Split | Prompt t/s | Decode t/s, prose / JSON | VRAM MiB (GPU0 / GPU1) |
|---|---|---|---|
| `-sm tensor -cmoe` | 94.5 / 94.2 | 20.6 / 21.0 | 4,370 / 4,370 |
| `-sm layer -cmoe` | 627.4 / 614.7 | 22.8 / 22.9 | 3,776 / 3,334 |
| `-sm layer`, MoE cache 23,552 MiB (step 7 "best") | 837.6 / 841.9 | 53.1 / 52.9 | 15,784 / 15,046 |

**Verdict:** on these cards with this model, llama.cpp's only good two-GPU mode is the layer split with the MoE cache. The tensor split cannot use the cache and reads prompts 6.6× slower than the layer split.

### 13c. Strata's layer split at long context, and the job

**Configs:** `strata-2gpu-nodraft-128k.json` (step 10's layer split, `--max-context 131072 --kv-resident 32768`, caches off), `strata-2gpu-nodraft-128k-cache.json` (the same with Strata's caches, for the job), and `strata-1gpu-nodraft-128k.json` (one GPU, the same context settings) for what the second card adds.

**Actual** (`runs/13c-strata-long/summary.md`; needle found in every request and job):

| Prompt | llama.cpp, 2 GPUs | Strata, 1 GPU | Strata, helper cache (12b) | **Strata, layer split** |
|---|---|---|---|---|
| 32,804 tokens: TTFT | 39.7 s | 15.5 s | 15.5 s | **10.5 s** (3,156 t/s) |
| 126,6xx tokens: TTFT | 153.4 s | 61.8 s | 61.5 s | **36.3 s** (3,513 t/s) |
| Four-turn job (3 runs) | 62.2–65.3 s | – | 32.8–33.4 s | **28.7–29.4 s** |

Decode with 126K behind it stays at 66–70 t/s on the layer split (as on the helper cache).

**Verdict:**
- **For long prompts the second card matters:** the layer split reads 1.5× (32K) to 1.7× (126K) faster than one GPU. In step 10, with 4K prompts, it added only 6 %: a long prompt is read in large chunks, and each card streams only its own layers' missing experts.
- The helper cache reads prompts on CUDA0 alone (`docs/SECOND_GPU.md`: "Prompt prefill still uses CUDA0"), and its long-prompt times equal one GPU's.
- Against llama.cpp on two GPUs: 4.2× faster to the first token at 126K, and the job in 45 % of the time.

### 13d. Strata's layer split: split point and hand-off

**Configs:** step 10's `strata-2gpu-nodraft.json` with `"layer_split": "20"` or `"32"` instead of `"auto"` (which chose 26), and with `"STRATA_SPLIT_OVERLAP": "1"` (overlapped hand-off between the cards).

**Actual** (`runs/13d-strata-split/summary.md`; step 10 auto: 64.8 / 64.8 decode, 1,878 prompt):

| Split | CUDA1 runs | CUDA1 cache | Decode t/s, prose / JSON | Prompt t/s |
|---|---|---|---|---|
| 20 | layers 20–47 | 6,227 slots (11.12 GiB) | 64.6 / 64.7 | 1,881 |
| 32 | layers 32–47 | 6,476 slots (12.10 GiB) | 64.8 / 64.4 | 1,875 |
| auto (26) + overlapped hand-off | layers 26–47 | 6,311 slots | 64.9 / 64.8 | 1,874 |

**Verdict:** none of them changes anything beyond 0.5 %. The docs say the overlap "helps AMD; no gain on NVIDIA", which holds here. With 97 % cache hits on two cards, where the split falls does not matter for one stream.

### 13e. Several requests at once

**Why:** with `"parallel": N` Strata decodes N conversations in the same verify windows (`docs/BATCHING.md`), and on a layer split it pipelines groups of slots, so the two cards work on different conversations at the same time instead of waiting for each other. That is where a second card could pay most.

**Client:** `oai-bench.py --concurrency 1,2,4` (`BENCH_CONCURRENCY`): N requests sent together per repetition, each prompt starting with its own tag (no shared prefix), 3 repetitions per level on one server start. "Total t/s" = all generated tokens / the group's wall time, from the first request sent to the last stream closed (so it includes prompt reading). Two workloads:
- **long prompts:** the prose prompt (4,315 tokens), 256 tokens out each: what a few agents sending documents at once see.
- **short:** a new prompt kind of about 50 tokens ("Write a long story, at least 1,500 words …"), 512 tokens out: decode alone.

**Configs:** `"parallel": 4` added to the one-GPU, layer split and helper configs (`strata-*-par4.json`, no drafting, and MTP for the two two-GPU modes); `"parallel": 8` on the layer split (`strata-2gpu-nodraft-par8.json`). llama.cpp: the two-GPU "best" with `-np 4 -c 32768` (8,192 per slot); its MoE cache had to shrink to 22,016 MiB (23,552 projected 31,149 of 31,417 MiB and failed on the compute buffer, `runs/13e-concurrency-short/probe/`).

**The engine's batch setup** (layer split, 4 slots): "--batch-groups auto: 2 groups of 2 slots", "4 slot sessions on CUDA0 (0.29 GiB each)", the same on CUDA1; with 8 slots "2 groups of 4 slots".

**Actual, short prompts** (total t/s, 512 tokens per stream; `runs/13e-concurrency-short/summary.md`):

| Configuration | 1 stream (decode t/s) | 2 at once | 4 at once | 8 at once |
|---|---|---|---|---|
| llama.cpp, 2 GPUs, `-np 4` | 52.1 | 85.5 | 117.1 | – |
| Strata, 1 GPU | 56.6 | 73.8 | 104.5 | – |
| Strata, helper cache | 67.6 | 90.0 | 132.8 | – |
| Strata, helper cache + MTP | 95.5 | 88.9 | 130.7 | – |
| **Strata, layer split** | 65.3 | **113.1** | **173.1** | **229.4 ± 14.7** |
| Strata, layer split + MTP | 94.1 | 112.4 | 172.2 | – |

**Actual, long prompts** (total t/s including prompt reading, 256 tokens per stream; `runs/13e-concurrency/summary.md`):

| Configuration | 2 at once | 4 at once |
|---|---|---|
| llama.cpp, 2 GPUs, `-np 4` | 32.1 | 37.2 |
| Strata, 1 GPU | 39.5 | 52.4 |
| Strata, helper cache | 44.3 | 58.4 |
| **Strata, layer split** | **46.8** | **61.7** |

The prompts of concurrent requests are read one after another (the per-stream TTFTs of a group of four on the layer split: 2.3, 6.9, 9.3, 11.6 s), and a stream that already decodes slows while another prompt is read, so per-stream decode rates in this workload are not comparable; the totals are. With 4.3K-token prompts and 256-token answers, prompt reading is most of the wall time.

**Verdict:**
- **Several requests are where the second card pays most.** With 4 short requests at once the layer split delivers 173 t/s, 1.66× one GPU (104.5) and 1.30× the helper cache (132.8); with 8, about 230 t/s, 3.5× a single stream. The pipelined batch groups keep both cards busy.
- The helper cache scales less: CUDA0 still runs every layer of every slot.
- MTP helps a single stream (+44 % on the layer split) and nothing from two streams on: "the pipelined path does not run the slots' MTP drafts" (`docs/BATCHING.md`); `--batch-mtp` with `--batch-groups 1` would, and was not tried.
- llama.cpp batches too (117 t/s with 4 streams), but Strata's layer split delivers 48 % more with 4 short requests and 66 % more with 4 long ones.
