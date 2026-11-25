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
