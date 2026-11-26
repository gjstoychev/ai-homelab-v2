# Basalt: Qwen3.8-Flash-Next on 2× RTX 5060 Ti, against Strata on the same weights

Status: **done** (2026-11-26), after [`2026-11-strata`](../2026-11-strata/). Recommended setup: [below](#the-recommended-two-gpu-setup). Every step with the exact command, expected vs actual: [`journal.md`](journal.md).

## Question

[Basalt](https://github.com/jesdga95/basalt) is a fork of Strata whose hot paths were rewritten for Blackwell (sm_120). How fast does it run Qwen3.8-Flash-Next on `formula` (2× RTX 5060 Ti 16 GB, 64 GB RAM) with nothing skipped, against Strata on the same source quant with the same client and prompts? Which of its two-card modes (`peer`, `split`) suits two equal 16 GB cards? And how much does its recommended 16 GB pack (UD-IQ3_XXS) gain over the older one?

The public claims are about 1.4× Strata's decode with both engines exact (93 against 66 tok/s on one RTX 5080) and about 25 % less time on a real coding job. The only two-card RTX 5060 Ti number is 2,370 prompt / 161.3 decode tok/s on an IQ3_S pack, with no version, mode or RAM given (basalt.jgulfo.com community table).

**Main question (as in `2026-11-strata`, added 2026-11-26): the best way to run this model on both cards with Basalt, and how it compares with Strata's best two-GPU setup** (layer split + MTP + 4 slots: 87.6 / 133.7 t/s for one chat, 172 t/s for four, a 126K-token prompt in 36 s, the four-turn job in 29 s).

## Pinned versions

Nothing is updated during the experiment. Basalt releases every two days, has no backward-compatibility shims (0.5.7 turned `gpu: auto` and a GPU list without `gpu-mode` into errors) and names no stable release.

| Part | Version | Source |
|---|---|---|
| Basalt | **v0.5.7** (commit `400b411`, 2026-11-24) | built from source at the tag |
| Model 1 | **`Qwen3.8-FN-GSQ-RCO-IQ3_XXS.basalt`**, HF `jesdga/Qwen3.8-Flash-Next-Basalt` at revision **`aa76f46bb60bdfed4a4521da16d9952552917e2b`** (the last one with the GSQ-RCO packs; removed in `451e426` on 2026-11-07). 79,865,545,428 bytes, SHA-256 `c20595c2dd1c444b465f916447b835c1f30189ba1286932427115dac7b365165`, format v2 (re-uploaded 2026-11-01). Experts 40.0 GiB | Hugging Face |
| Model 2 | **`Qwen3.8-FN-UD-IQ3_XXS.basalt`** at the current head **`fc721830e519ab52012d372e89b90d9a7df30ff2`** (pack added in `aa76f46`, unchanged since). 87.6 GB, experts 46.8 GiB. Built from Unsloth's UD-IQ3_XXS, with a tuned MTP drafter | Hugging Face |

Model 1 is the same ISTA-DASLab GSQ-RCO IQ3_XXS that `2026-11-strata` runs as a GGUF, so it gives the like-for-like comparison. Basalt's card no longer lists it. No source says 0.5.7 refuses it: the format is still v2, and the current `docs/precision/qwen38-flash-next.md` still measures IQ3_XXS and IQ3_S packs. Step 4 tests that first; if it is refused, the cross-engine comparison falls back to model 2 against Strata's UD-IQ3_XXS (if `2026-11-strata` step 13 worked) and is labelled as different files otherwise.

| Pack | Experts | Top-1 against Q8_0 | Mean KL | Source |
|---|---|---|---|---|
| GSQ-RCO IQ3_XXS | 40.0 GiB | 84.03 % | 0.2404 | Basalt card at `aa76f46` |
| UD-IQ3_XXS | 46.8 GiB | 88.67 % | 0.1188 | Basalt card at `fc72183` |

### Host stack: no changes needed

| Layer | On `formula` | Basalt needs | Source of the requirement |
|---|---|---|---|
| Kernel | 7.0.0-43-generic | Linux x86-64 with AVX2 | README |
| NVIDIA driver | 580.178.04 (open module) | R580 or newer; release archives built with CUDA 13.0.2 need 580.95.05 or newer. Known bad: an older 615 build looped every reply | `docs/platforms.md`, 0.5.6 notes |
| CUDA toolkit | 13.0 | `CUDA 13.0 or newer` (CMake stops below 13.0) | `CMakeLists.txt` |
| GPU architecture | sm_120 | one compute capability, 12.0 or 12.1; FP4 targets build as `sm_120a-real` | `cmake/target_arch.cmake` |
| Compiler / CMake | gcc 13.3 / 3.28 | GCC 13+, CMake 3.24+ | README |
| NCCL, P2P | not used | not mentioned; peers move data over PCIe and pinned RAM | `docs/configuration/gpus.md` |
| Python | 3.12.3 | 3.10+, `requirements.txt` pinned (numpy 2.5.3 on 3.12, jinja2 3.1.6, regex 2026.9.10, pyyaml 6.0.3, tqdm 4.70.1, pillow 12.3.0, pypdf 7.0.1, psutil 7.2.2) in a venv | `requirements.txt` |
| RAM | 64 GB | enough for the experts the GPUs do not hold; `ram-reserve-mib` 2048 plus 2 GiB stay free; anything beyond is read from the pack on the SSD (logged as "N experts on SSD") | `docs/running.md` |

Running after `2026-11-strata` on the same stack keeps the engines as the only difference. The driver and CUDA stay as they are until both experiments are done (see "Afterwards").

## Settings that change the output

| Setting | Default in 0.5.7 | Here |
|---|---|---|
| `route-tail-skip`, `route-prior`, `relaxed-drafting` | off; each named in a start warning when on | off, checked in the log; `route-tail-skip` on only in step 8 |
| `spec` (MTP drafter in the pack) | 8 | `0` and `suffix-draft` `0` for the baseline, the default as its own step |
| `kv` | `fp16` for Flash-Next | `int8` (Strata's default; measured lossless by Basalt) |
| `adapt-swaps`, `peer-adapt-swaps` | adaptive | default for speed runs, `0` for the quality check |
| Peer mode prompt numerics | about 0.01 KLD from one card | recorded; `--diag-main-card-bits` not used |

With drafting on and `relaxed-drafting` off, Basalt's output is the same as plain decoding (275 of 275 greedy prompts in its docs).

## Method

The same as `2026-11-strata`: `scripts/oai-bench.py` on `/v1/chat/completions`, prose and structured prompts reported separately, one warm-up and 5 measured requests, the same samplers, the same ten quality prompts and the same four-turn job. A run is rejected if the log reports experts on the SSD (the Basalt community bench refuses such runs too), a page-locking failure, swap or an unexpected setting.

## Steps

1. **Disk.** Archive the Strata model files if space is short (see "Disk").
2. **Build.** `scripts/build-basalt.sh v0.5.7`: clone at the tag into `~/basalt/v0.5.7/src`; `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBASALT_MODELS=qwen4exp` (Flash-Next only; "the build fetches nothing"); `cmake --build build -j`. The binaries land in the source tree's `bin/` (`basalt`, `basalt-bench`, `basalt-probe`). `build-info.txt` as for llama.cpp; venv with `requirements.txt`, passed to `serve` with `--python`.
3. **Model 1.** `hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-GSQ-RCO-IQ3_XXS.basalt --revision aa76f46…` into `~/models/qwen3.8-flash-next-basalt/`; SHA-256 against the value above.
4. **Smoke test, one GPU.** `bin/basalt serve --config basalt-gsq-rco-iq3xxs.json --python …`, config with `"version": 2`, `"gpu": "0"`, `"kv": "int8"`. Does 0.5.7 load the pack? From the log: lossy flag warnings (none expected), expert cache, RAM tier, "experts on SSD" (none expected), drafter.
5. **Plan for two cards without loading.** `--gpu-plan` for `peer` and `split`: experts per card, what stays in RAM.
6. **One GPU:** (a) `spec 0`, `suffix-draft 0`; (b) default drafting.
7. **Two GPUs:** `"gpu": "0,1"` with `"gpu-mode": "peer"` and then `"split"`, each with 6a and 6b. If the start prints "prefill peer: the peer's buffers do not fit", raise `peer-reserve-mib` and record it.
8. **`route-tail-skip` on** with the fastest exact setting: the speed it adds, for comparison with Strata's step 9.
9. **Model 2 (UD-IQ3_XXS):** download at `fc72183`, then 6b and the fastest two-card mode. Watch the RAM tier: 46.8 GiB of experts minus what the GPUs hold, against about 50–55 GiB usable.
10. **Quality check** with the ten prompts, against llama.cpp and Strata's answers.
11. **Two GPUs, every mode** (the same matrix as `2026-11-strata` step 13), with the shared clients:
    - a. Long context: TTFT and needle at 32K and 128K, and the four-turn job, for `peer`, `split` and one GPU.
    - b. `gpu-share` and the split point (uneven shares), `peer-reserve-mib`, and whatever two-card options Basalt 0.5.7 documents beyond those.
    - c. Several requests at once (1, 2, 4, 8; `"parallel"`, which "needs `kv` `int8`" for Flash-Next) on one GPU, `peer` and `split`, short prompts and 4.3K-token prompts.
    - d. The best Basalt setup per use, next to Strata's and llama.cpp's.
12. **Write-up:** the cross-engine table (llama.cpp, Strata, Basalt on the same weights, same prompts), findings, `docs/stack.md`, README index.

## Disk

| | Size |
|---|---|
| Model 1 (GSQ-RCO IQ3_XXS pack) | 79.9 GB |
| Model 2 (UD-IQ3_XXS pack) | 87.6 GB |
| Basalt build and venv | a few GB |

About 270 GB are free on the NVMe after `2026-11-strata` even if its GGUF stays, enough for both packs (about 100 GB left). The archive has about 59 GB free after step 2 of `2026-11-strata`; the Strata GGUF (75.8 GB) only moves there after more space is made, which is the user's call.

## Afterwards: the vLLM experiment

The next experiment reproduces the dual RTX 5060 Ti vLLM recipe for Qwen3.8-27B (video *Best Ways to Run Qwen3.8-27B on Dual RTX 5060 Ti Cards*, RepoChad, 2026-09-25; repository `skrodahl/qwen38-27B-dual-rtx5060`). Its versions are pinned on purpose and are not to be replaced with the newest ones: newer revisions of the NVFP4 checkpoint dropped the MTP head, so the latest one silently loses speculative decoding and its speedup. Pinned there: checkpoint `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` at `0cc27958cefbbe231782ec8511de8c4eb5233348`, vLLM 0.30.0 for the FP8-KV results, driver 595.91.07 with CUDA 13.2. That needs a driver upgrade on `formula`, which is why it comes after both engine experiments: the upgrade changes the stack under them.

## Results

Every number traces to a file in `runs/` and a step in the journal. All runs: kernel 7.0.0-43, driver 580.178.04, CUDA 13.0.88, Basalt v0.5.7 (`400b411`, built from source for sm_120, Flash-Next only), the GSQ-RCO IQ3_XXS pack at `aa76f46` unless stated, route tail skip off unless stated; Strata and llama.cpp numbers from [`2026-11-strata`](../2026-11-strata/) on the same weights (the GGUF the pack was built from), with the same clients and prompts (temperature 0, thinking off, nothing reused unless stated). No page-cache drops and no CPU power (runs made without sudo).

**Three engines, the best setup of each, on this machine:**

| Use | llama.cpp v1.0.0 | Strata v0.2.21 | **Basalt v0.5.7** | Basalt setup |
|---|---|---|---|---|
| One chat, 1 GPU, exact (decode t/s, prose / JSON) | 45.2 / 42.4 | 62.8 / 62.9 | **72.3 / 71.7** | one card |
| One chat, 1 GPU, MTP | – | 77.8 / 118.2 | **104.8 / 170.5** | one card |
| One chat, 2 GPUs, exact | 53.1 / 52.9 | 66.9 / 66.6 | **77.4 / 78.4** | peer |
| One chat, 2 GPUs, MTP | – | 93.4 / 159.2 | **113.1 / 193.3** | peer |
| Prompt, 4.3K tokens (t/s) | 872 (1 GPU) | 1,878 (2 GPUs) | 1,876 (peer) | |
| 126K-token prompt, time to first token | 153.4 s | 36.3 s | **34.7 s** | peer |
| Four-turn job over a 32K document | 64.2 s | 29.0 s | **25.5 s** | peer |
| 2 / 4 requests at once (short, total t/s) | 86 / 117 | 113 / 173 | **144 / 173** | peer + MTP, 4 slots |
| 8 requests at once | 151 | **≈ 229** | 200.5 | peer + MTP, 8 slots, `vram-reserve-mib` 1500 |
| 4 requests at once, 4.3K-token prompts | 37 | **62** | 58 | split |
| Quality: correct / same text as llama.cpp (of 10) | 10 / 10 | 10 / 8 | 10 / 9 | |

**Basalt's two-GPU modes** (decode t/s, 4.3K-token prompt; steps 6, 7, 11):

| Mode | Exact | MTP, prose | MTP, JSON | 4 requests (short) | 126K prompt |
|---|---|---|---|---|---|
| One card | 72.3 | 104.8 | 170.5 | 125 | 54.6 s |
| peer | 77.4 | 113.1 | 193.3 | 146 (exact) / 173 (MTP) | 34.7 s |
| split | 71.0 | 107.4 | 184.5 | 121 / 165 | 35.8 s |
| `gpu-share` 0.4,0.6 / 0.6,0.4 (split), 1,1 (peer) | 70.4–77.6 | – | – | – | – |

**The UD-IQ3_XXS pack** (better quality, 46.8 GiB of experts): 0–8 % slower than GSQ-RCO IQ3_XXS; peer + MTP 112.3 / 200.9 t/s; fits in RAM in every mode.

## Findings

1. **Basalt is the fastest engine here for one user**, on the same weights as Strata and llama.cpp: +14–18 % over Strata exact, +21–44 % with the drafter, 1.4–1.7× llama.cpp exact. It also reads prompts slightly faster (126K tokens in 34.7 s).
2. **Peer is Basalt's best two-GPU mode**: the main card runs every layer, the second card holds 34 % of the experts and computes their rows, and reads part of long prompts. One chat gains 7–13 % over one card, a long prompt 1.6×.
3. **For several requests the engines trade places.** Basalt keeps the MTP drafter in batches (peer + MTP: 144 t/s for 2, 173 for 4, 200 for 8); Strata's layer split pipelines batches across the cards but without drafts (113 / 173 / 229). Up to four requests Basalt is ahead or level; at eight Strata's layer split is ahead.
4. **Split mode does not pay on these cards for Basalt** (no better than one card for one chat, behind peer for batches). Shares change nothing.
5. **The route tail skip** drops half of the misses (1.5–6 % of all routed experts) for 1–8 % speed, the same picture as Strata. Off by default in both and stays off.
6. **Memory:** peer mode needs the most RAM (lowest MemAvailable 3.5 GB at 128K context, 5.1 GB with the UD pack); 8 slots in peer mode need `vram-reserve-mib` 1500 or GPU 0 runs out of memory while capturing the slots' windows.
7. **Pitfalls:** Basalt checks the request's `model` name (alias the config), writes a token-totals file next to the config (`"totals-file": "none"`), appends every start to one engine log per model, and has no `--version` flag.

## The recommended two-GPU setup

Basalt v0.5.7 with `"gpu": "0,1"`, `"gpu-mode": "peer"`, the pack's MTP drafter (default `spec` 8), `"kv": "int8"`, `"parallel": 4`, Basalt's default prompt cache, and the UD-IQ3_XXS pack for quality ([`configs/basalt-ud-peer-mtp.json`](configs/basalt-ud-peer-mtp.json) plus `"parallel": 4`; measured as [`configs/basalt-peer-mtp-par4.json`](configs/basalt-peer-mtp-par4.json) on the GSQ-RCO pack). For long contexts set `"max-context": 131072`. For a server that mostly takes eight or more simultaneous requests, Strata's layer split with `"parallel": 8` delivered more ([`2026-11-strata`](../2026-11-strata/README.md#the-recommended-two-gpu-setup)).

## Not measured

Long mixed workloads, sampled decoding, `relaxed-drafting` and `route-prior`, the NVFP4 packs (they need more RAM than 64 GB for all experts), more than 8 requests, context beyond 128K, the UD pack at long context and under concurrency, CPU power and cold-cache load times.
