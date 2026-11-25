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
