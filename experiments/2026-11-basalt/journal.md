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

## 3. Model 1 (2026-11-26)

**Command:** `scripts/hf-fetch.sh jesdga/Qwen3.8-Flash-Next-Basalt aa76f46bb60bdfed4a4521da16d9952552917e2b ~/models/qwen3.8-flash-next-basalt-gsq-rco-iq3xxs Qwen3.8-FN-GSQ-RCO-IQ3_XXS.basalt`

**Actual:** 79,865,545,428 bytes in 2,287 s (about 35 MB/s), SHA-256 `c20595c2…b365165` OK from the disk (O_DIRECT, 51 s).

**Model 2** (UD-IQ3_XXS for step 9) started right after: `Qwen3.8-FN-UD-IQ3_XXS.basalt` at the current head `fc721830e519ab52012d372e89b90d9a7df30ff2` (the pack unchanged since `aa76f46`), 87,602,085,467 bytes, SHA-256 `b59b3500…1dd9a11` by the Hub API.

## 4. Smoke test, one GPU (2026-11-26)

**Command** (in `~/basalt/v0.5.7/src`): `bin/basalt serve --config ~/ai-homelab-v2/experiments/2026-11-basalt/configs/basalt-1gpu-nodraft.json --python .venv/bin/python`, then one short request (logs in `runs/4-smoke/`).

**Expected:** 0.5.7 loads the GSQ-RCO pack (format v2); no lossy-flag warnings; no experts on the SSD.

**Actual:**
- **The pack loads.** Ready after 18 s: "loaded 43.3GiB 14.0s cuda0=13.9GiB ram=29.4GiB", the model's name "Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS".
- "experts live in VRAM 10.8 GiB (27%), peer GPU 0.0 GiB (0%), RAM 29.1 GiB (73%), file (SSD) 0.0 GiB (0%); 40.0 GiB in all". Expert cache 6,661 slots (10.83 GiB) pre-filled from `data/profiles/qwen4exp-48x512.prof`; Strata's was 5,978 slots on the same card. The prompt path borrows 4,767 of them (7.68 GiB). Resident RAM mode: 37.44 GiB page-locked.
- `[server] auto ram-experts=40.3GB pool-affinity=auto pool=19`: 19 CPU expert workers by Basalt's rule (Strata: 13).
- PCIe probe 20.2 GB/s; "this CPU has no AVX-512: the expert kernels run on AVX-2". Two warnings "fallback: dense_tc projection on native_mmvq (types 4 / 21)" and "(types 8 / 21)": two dense projection types of this older pack take a slower path. No lossy-flag warnings.
- VRAM 15,168 MiB on GPU 0; RAM 43 GB used, 18 GB available.

**Two client changes:**
- The first request was refused: `404 "The model 'x' does not exist"`. Basalt checks the request's `model` field; llama.cpp and Strata ignore it. Every Basalt config now has `"alias": "bench"`, and all three clients send `"model": "bench"` (`oai-bench.py` already did; `long-bench.py` and `quality-check.py` changed).
- The server wrote `basalt-1gpu-nodraft.totals.json` (persistent token counts) next to the config, in the repository. `"totals-file": "none"` in every config's `server` block turns it off.

**The request:** "Vienna is the capital of Austria." (Strata and llama.cpp: "The capital of Austria is Vienna."), 54 tokens of which most were thinking. Server line: "pp=65/0c 1.0s tg=54 64.8t/s ttft=1.0s end=stop hit=71.5% pcie=0.0%"; `timings` as in Strata (`prompt_per_second` 63.4, `predicted_per_second` 64.8, `draft_n` 0). The server also logged "thinking budget 100 tokens, output cap 200 (answer-room auto)": Basalt caps thinking by default; the clients turn thinking off, so it does not apply to the measurements.

**Stopping:** `bin/basalt serve` runs the server as `python -m serve.server` with the engine as its child; SIGTERM to the server: "stopping … stopped", RAM and VRAM back to idle. `scripts/bench-run.sh` stops the whole process group, so it works unchanged. The engine log is in `~/.cache/basalt/<id>/engine.log`, copied next to each run.

**Verdict:** works. 64.8 t/s decode on a short prompt without drafting (Strata's smoke test: 51.1 t/s, also mostly thinking tokens).

## 6–7. One GPU, and two GPUs in peer and split mode (2026-11-26)

**Why:** the same comparison as `2026-11-strata` steps 8 and 10: exact (no drafting) and with the drafter, one card and both, on the standard prose and JSON prompts.

**Commands** (in `experiments/2026-11-basalt`, `B=~/basalt/v0.5.7/src`):

```bash
scripts/bench-run.sh runs/6-7-modes <run> 5 -- env -C $B bin/basalt serve --config configs/basalt-<run>.json --python .venv/bin/python
```

for `<run>` = `1gpu-nodraft`, `1gpu-mtp`, `peer-nodraft`, `peer-mtp`, `split-nodraft`, `split-mtp`. The UD-IQ3_XXS download ran at the same time (about 0.3 of a CPU core, NVMe writes, nothing on the GPUs).

**Engine logs:** Basalt keeps one `engine.log` per model in `~/.cache/basalt/<id>/` and appends every start to it; it was split at each "basalt: system:" line into `runs/6-7-modes/<run>.engine.log`.

**Placement** (engine logs):
- One GPU: expert cache 6,661 slots (10.83 GiB), 6,017 with the MTP drafter; "experts live in VRAM 10.8 GiB (27%) … RAM 29.1 GiB (73%), file (SSD) 0.0 GiB".
- Peer: "gpus: peer; RTX 5060 Ti (nvidia-smi 0) main, share 0.50; RTX 5060 Ti (nvidia-smi 1) peer, share 0.50 (bandwidth)". GPU 0 holds the dense weights, the KV and 6,270 experts (10.2 GiB); "peer GPU 1: 8458 experts, 13.61 GiB" (9,038 / 14.55 GiB with MTP). Experts: 26 % on GPU 0, 34 % on GPU 1, 40 % in RAM. "peer GPU 1 computes its experts' rows of each prompt chunk up to 2559 tokens". With MTP only 582 MiB stay free on GPU 1. `peer-exclusive` off (RAM holds all experts).
- Split: "layers 0-23, 8317 experts (12.0 GiB); … layers 24-47, 6314 experts (11.4 GiB)" (7,307 on GPU 0 with MTP), cut by free VRAM. "pipelined windows: on"; with MTP, every guessed window was held ("4 launched, 4 held").
- No lossy-flag warnings, nothing on the SSD. Peer mode used the most RAM: RSS 47 GiB, lowest MemAvailable 5.8 GB (split: 38–40 GiB, 12.5–14.2 GB).

**Actual** (`runs/6-7-modes/summary.md`; all requests read their whole prompt and ran to 256 tokens):

| Run | Kind | TTFT s | Prompt t/s | Decode t/s | Drafts accepted |
|---|---|---|---|---|---|
| 1 GPU, no drafting | prose | 2.32 | 1,871.7 | 72.3 ± 0.2 | – |
| 1 GPU, no drafting | JSON | 2.37 | 1,838.9 | 71.7 ± 0.3 | – |
| 1 GPU, MTP | prose | 2.36 | 1,845.6 | 104.8 ± 1.8 | 612 of 873 (70 %) |
| 1 GPU, MTP | JSON | 2.41 | 1,813.5 | 170.5 ± 7.7 | 1,041 of 1,291 (81 %) |
| peer, no drafting | prose | 2.31 | 1,875.6 | 77.4 ± 0.3 | – |
| peer, no drafting | JSON | 2.36 | 1,850.4 | 78.4 ± 0.3 | – |
| peer, MTP | prose | 2.34 | 1,855.3 | 113.1 ± 2.3 | 618 of 955 (65 %) |
| peer, MTP | JSON | 2.39 | 1,826.2 | 193.3 ± 10.1 | 1,042 of 1,303 (80 %) |
| split, no drafting | prose | 2.34 | 1,855.0 | 71.0 ± 1.4 | – |
| split, no drafting | JSON | 2.37 | 1,839.4 | 71.5 ± 0.2 | – |
| split, MTP | prose | 2.48 | 1,752.3 | 107.4 ± 4.1 | 620 of 940 (66 %) |
| split, MTP | JSON | 2.51 | 1,739.4 | 184.5 ± 15.4 | 1,026 of 1,212 (85 %) |

**Against Strata on the same weights** (Strata steps 8, 10, 10b; Strata's MTP ran `--spec 4`, Basalt's default is 8):

| | Strata | Basalt | Basalt against Strata |
|---|---|---|---|
| 1 GPU, exact, prose / JSON | 62.8 / 62.9 | 72.3 / 71.7 | +15 % / +14 % |
| 1 GPU, MTP, prose / JSON | 77.8 / 118.2 | 104.8 / 170.5 | +35 % / +44 % |
| 2 GPUs, exact (best mode each: Strata helper cache, Basalt peer) | 66.9 / 66.6 | 77.4 / 78.4 | +16 % / +18 % |
| 2 GPUs, MTP, prose (Strata helper + MTP, Basalt peer) | 93.4 | 113.1 | +21 % |
| 2 GPUs, MTP, JSON (Strata pipelined split, Basalt peer) | 159.2 | 193.3 | +21 % |
| Prompt t/s, 1 GPU | 1,779 | 1,872 | +5 % |

**Verdict:**
- Basalt is faster than Strata in every configuration: +14–18 % exact, +21–44 % with the drafter. Part of the drafter gap is the window (8 against 4 tokens); exact decoding has no drafter and still gains 15 %.
- **Basalt's best two-card mode for one chat is peer**: +7 % exact and +8 % (prose) / +13 % (JSON) with MTP over one card. Split is no better than one card without drafting (71.0 against 72.3) and in between with MTP.
- As with Strata, the second card adds little to one chat: 7–13 %.

## 11. Two GPUs, every mode (2026-11-26)

**Why:** the main question, with the matrix of `2026-11-strata` step 13 and the same clients. Run as one batch, `experiments/2026-11-basalt/scripts/run-matrix.sh`; every run through `scripts/bench-run.sh`, which now saves the run's part of Basalt's appended engine log (`BENCH_ENGINE_LOG`) as `<run>.engine.log`. The UD-IQ3_XXS download ran during the first half (paused 08:38–08:49 at the user's request).

### 11a. Long context and the four-turn job

**Configs:** the no-drafting configs with `"max-context": 131072` (`basalt-{1gpu,peer,split}-nodraft-128k.json`), and the same without `"prompt-cache": 0` for the job (Basalt's default of 6 checkpoints). No `kv-resident`: Basalt's own elastic KV handles a long context on one card and in peer mode ("each session backs its first 32,768 cells at start … more cells come from the expert cache when a request needs them"); split keeps its KV on its cards.

**Actual** (`runs/11a-long/summary.md`; needle found in all 18 long requests and all 9 jobs):

| | Basalt 1 GPU | Basalt peer | Basalt split | Strata best (layer split) | llama.cpp, 2 GPUs |
|---|---|---|---|---|---|
| 32,804 tokens: TTFT | 13.5 s (2,430 t/s) | **10.5 s** (3,131) | 10.6 s (3,107) | 10.5 s | 39.7 s |
| 126,586 tokens: TTFT | 54.6 s (2,320) | **34.7 s** (3,649) | 35.8 s (3,544) | 36.3 s | 153.4 s |
| Four-turn job | 30.7–31.0 s | **25.2–25.8 s** | 27.0–28.7 s | 28.7–29.4 s | 62.2–65.3 s |

Decode with 126K behind it: 75–79 t/s on one card and split, 87–88 in peer mode. Peer mode at 128K left the least RAM: lowest MemAvailable 3.5 GB (RSS 47 GiB).

### 11b. Shares

**Configs:** split with `"gpu-share": "0.4,0.6"` (layers 0–17 / 18–47) and `"0.6,0.4"` (0–28 / 29–47); peer with `"gpu-share": "1,1"` (also interleaves the hottest experts between the cards; 8,414 experts on GPU 1).

**Actual** (`runs/11b-shares/summary.md`): split 70.4 / 71.2 decode (default cut 71.0), prompt 1,804 / 1,860; peer 77.6 / 78.5 (default 77.4 / 78.4). **Verdict:** no effect, as the split point had none in Strata.

### 11c. Several requests at once

**Configs:** `"parallel": 4` (one card, peer, split; peer and split also with MTP) and `"parallel": 8` (peer, split). Peer with 8 slots did not start: "error: capturing slot 4's solo windows failed: verify: instantiate: out of memory" on GPU 0. With `"vram-reserve-mib": 1500` it starts (`basalt-peer-{nodraft,mtp}-par8-res1500.json`).

**Actual, short prompts** (512 tokens per stream, total t/s; `runs/11c-concurrency-short/summary.md`):

| | 1 request (decode t/s) | 2 at once | 4 at once | 8 at once |
|---|---|---|---|---|
| Basalt, 1 GPU | 70.4 | 96.2 | 125.2 | – |
| Basalt peer | 77.7 | 107.4 | 145.8 | 189.8 |
| Basalt split | 70.8 | 91.0 | 120.6 | 147.4 |
| **Basalt peer + MTP** | **121.2** | **144.2** | **173.4** | **200.5** |
| Basalt split + MTP | 115.7 | 138.4 | 164.7 | – |
| Strata layer split (step 13e) | 65.3 | 113.1 | 173.1 | ≈ 229 |
| Strata layer split + MTP | 94.1 | 112.4 | 172.2 | – |
| llama.cpp, 2 GPUs | 52.1 | 85.5 | 117.1 | 151.2 |

**Actual, 4.3K-token prompts** (total t/s including prompt reading; `runs/11c-concurrency/summary.md`): one card 41.8 / 54.4 (2 / 4 at once), peer 44.0 / 56.4, split 43.2 / 58.2. Strata's layer split: 46.8 / 61.7.

**Verdict:**
- Basalt keeps the MTP drafter when several requests run; Strata's pipelined batches do not draft. So Basalt peer + MTP is the best at 1–2 requests (121 / 144 t/s against Strata's 94 / 113) and level with Strata at 4 (173).
- Basalt's split does not pipeline batches across the cards the way Strata's layer split does: from 4 requests on without drafting, Strata's layer split is ahead (173 against 121–146 at 4; ≈ 229 against 190–200 at 8).
- With long prompts both engines are prompt-bound and within 10 % of each other.

### 11d. Basalt's best two-GPU setup per use

| Use | Best Basalt setup | Basalt | Strata's best | llama.cpp, 2 GPUs |
|---|---|---|---|---|
| One chat, exact | peer | 77.4 t/s | 66.9 | 53.1 |
| One chat, MTP, prose / JSON | peer + MTP | 113.1 / 193.3 t/s | 93.4 / 159.2 | – |
| 126K-token prompt | peer | 34.7 s | 36.3 s | 153.4 s |
| Four-turn job | peer | 25.5 s | 29.0 s | 64.2 s |
| 2 / 4 requests (short) | peer + MTP, 4 slots | 144 / 173 t/s | 113 / 173 | 86 / 117 |
| 8 requests (short) | peer + MTP, 8 slots, `vram-reserve-mib` 1500 | 200.5 t/s | ≈ 229 | 151 |

The all-round Basalt setup is **peer + MTP** (`basalt-peer-mtp-par4.json`): the fastest single chat measured on this machine, the fastest long prompt and job, and level with Strata at four requests. Strata's layer split is still ahead from about four to eight simultaneous requests.

## Control run (2026-11-26)

Step 6's one-GPU run again with no download running (`runs/6-7-modes-control/`): 72.5 / 71.9 decode, 1,876 / 1,846 prompt, against 72.3 / 71.7 and 1,872 / 1,839 during the download. The download did not affect the numbers.

## 8. Route tail skip on (2026-11-26)

**Configs:** `basalt-1gpu-nodraft-skip7.json` and `basalt-peer-mtp-skip7.json`: `"route-tail-skip": 7` (Strata's rank). The start warns: "--route-tail-skip 7 drops low-ranked missed experts", and every request reports what it dropped.

**Actual** (`runs/8-skip/summary.md`):

| | Skip off | Skip 7 | Gain | Dropped |
|---|---|---|---|---|
| 1 GPU, no drafting, prose / JSON | 72.3 / 71.7 | 72.8 / 72.9 | +1 % / +2 % | "27200 of 49770 routed entries no GPU held dropped (54.7%); 1.5% of all 1761120 entries" |
| peer + MTP, prose / JSON | 113.1 / 193.3 | 115.6 / 209.7 | +2 % / +8 % | "91786 of 195323 … (47.0%); 6.2% of all 1474560 entries" |

**Verdict:** as in Strata: half of the misses are dropped, 1.5–6 % of all routed experts, for 1–8 % more speed. The skip stays off.

## 9. The UD-IQ3_XXS pack (2026-11-26)

**Configs:** `basalt-ud-*.json`, the step 6/7 configs with the UD-IQ3_XXS pack (Unsloth UD-IQ3_XXS experts, 46.8 GiB, Basalt's recommended pack for 16 GB cards; top-1 88.67 %, KL 0.1188 against GSQ-RCO IQ3_XXS's 84.03 %, 0.2404 by Basalt's card).

**Placement:** one card: "experts live in VRAM 10.1 GiB (21%) … RAM 36.8 GiB (79%), file (SSD) 0.0 GiB"; peer: GPU 0 9.2 GiB, "peer GPU 1: 7693 experts, 14.54 GiB", RAM 23.1 GiB; split: layers 0–22 / 23–47, 21.1 GiB on the cards. Nothing on the SSD in any mode. Peer mode left the least RAM: RSS 55 GiB, lowest MemAvailable 5.1 GB.

**Actual** (`runs/9-ud/summary.md`; GSQ-RCO pack from step 6/7 for comparison):

| | GSQ-RCO IQ3_XXS | UD-IQ3_XXS | Change |
|---|---|---|---|
| 1 GPU, no drafting, prose / JSON | 72.3 / 71.7 | 68.2 / 68.2 | −6 % / −5 % |
| 1 GPU, MTP | 104.8 / 170.5 | 98.1 / 169.9 | −6 % / 0 % |
| peer, no drafting | 77.4 / 78.4 | 72.6 / 72.2 | −6 % / −8 % |
| peer + MTP | 113.1 / 193.3 | 112.3 / 200.9 | −1 % / +4 % |
| split + MTP | 107.4 / 184.5 | 105.3 / 182.0 | −2 % / −1 % |
| Prompt t/s, 1 GPU | 1,872 | 1,756 | −6 % |

**Verdict:** the better-quality pack costs 0–8 % speed (17 % more expert bytes per token) and still fits in RAM on 64 GB, in every mode. With peer + MTP it is as fast as the GSQ-RCO pack. For daily use on this machine UD-IQ3_XXS is the better choice; the cross-engine numbers stay on GSQ-RCO, the weights Strata ran.

## 10. Quality check (2026-11-26)

**Configs:** `1gpu-nodraft` (defaults), `1gpu-nodraft-det` (`"adapt-swaps": 0`, Basalt's advice for identity checks), `1gpu-nodraft-det-skip7`, `peer-mtp`, `peer-mtp-det` (`"adapt-swaps": 0, "peer-adapt-swaps": 0`). Same client and prompts as Strata's step 11 (`BENCH_CLIENT=scripts/quality-check.py`).

**Actual** (`runs/10-quality/quality-vs-llamacpp.md`; llama.cpp's answers from `2026-11-strata` as the reference):

| prompt | llama.cpp | Strata det. | Basalt defaults | Basalt det. | + skip 7 | peer + MTP | peer + MTP det. |
|---|---|---|---|---|---|---|---|
| 7 short ones | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| json | = ✓ | ≠ ✓ | ≠ ✓ ~ | ≠ ✓ | ≠ ✓ | ≠ ✓ ~ | = ✓ |
| code | = ✓ | ≠ ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| explain | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |

- 100 of 100 Basalt answers correct. Nine of ten prompts give llama.cpp's text in every Basalt setting, one more than Strata (code).
- The JSON answer varies: the tags (`"astronomer", "novelist", "gardener"` / `"archivist" …` / `"archaeologist" …`) and once only the layout (pretty-printed instead of one line). With the defaults the two repetitions differed; with `adapt-swaps 0` they repeat, as Basalt documents.
- The skip changed the JSON tags only.

**Verdict:** Basalt's answers are as correct as llama.cpp's and Strata's, and closer to llama.cpp's than Strata's.

## 12. Write-up (2026-11-26)

README: status, the three-engine table, Basalt's two-GPU modes, the UD pack, findings, the recommended two-GPU setup, what was not measured. `docs/stack.md`: the Basalt build and the two packs. Repository README: status.
