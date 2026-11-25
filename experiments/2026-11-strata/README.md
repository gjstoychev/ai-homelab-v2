# Strata: Qwen3.8-Flash-Next on 2× RTX 5060 Ti, the full model

Status: **planned** (2026-11-25). Every step with the exact command, expected vs actual: [`journal.md`](journal.md).

## Question

How fast does [Strata](https://github.com/Niko1221/Strata) run Qwen3.8-Flash-Next on `formula` (2× RTX 5060 Ti 16 GB, 64 GB RAM) when it runs the **whole model**, nothing skipped? How does that compare with llama.cpp on the same GGUF file, and what do speculative decoding and the route tail skip add?

**Main question (added 2026-11-25): what is the best way to run this model on both cards, and how much more do two cards give than one?** For a single chat, for structured output, for long prompts, and for several requests at once, with each engine's two-GPU modes.

The second half of the comparison is [`2026-11-basalt`](../2026-11-basalt/): Basalt, a fork of Strata rewritten for Blackwell, on the same source quant with the same client and prompts.

## Why "the whole model" matters

Qwen3.8-Flash-Next routes every token through 10 of 512 experts in each of its 48 layers. Strata keeps the most-used experts in VRAM and the rest in RAM. Its **route tail skip** leaves out missed experts ranked 7th or lower, so a token is computed from 6–9 experts instead of 10. It was on by default for 12/16 GB NVIDIA cards from engine 0.1.42 (2026-10-10) until Strata 0.2.17, which turned it off. Public Strata vs Basalt numbers mixed runs with and without it: on one RTX 5080 Strata went from 80 to 66 tok/s with the skip off, and Basalt ran 93 without it. Strata's own measurements show the quality cost growing with the file's precision (UD-Q4_K_XL KL 0.056 → 0.086 with the skip). Every comparison here starts with the skip off, set explicitly, and the run log is checked for it.

Sources for this section: the video *Basalt Runs Qwen 3.8 Flash Next At 224 Token/sec* (Kai, 2026-11-18) and Strata's `docs/DETAILS.md` and `docs/QUANTS.md`.

## Pinned versions

Nothing is updated during the experiment. Both engines release every few days and break things between releases (according to the video above, Strata engine 0.1.58 raised one owner's RAM at load from 68 to almost 115 GB and Linux killed it; Basalt 0.5.7 made two-GPU configs without `gpu-mode` an error), so a run is only comparable with runs on the same build.

| Part | Version | Source |
|---|---|---|
| Strata | **v0.2.21** (commit `d46f5f2`, 2026-11-24), engines 0.1.67 | built from source at the tag |
| Model | **ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, `IQ3_XXS/`**, revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (the folder's only commit is `fb6d866`, 2026-09-15). 2 shards: 47 GB (experts and dense weights) + 28.8 GB (n-gram table, read from the SSD) | Hugging Face |
| MTP draft layer | about 6 GB, fetched and packed with Strata's `terminal/tools/mtp_*.py`; source recorded at the step | Strata tools |
| Baseline engine | llama.cpp v1.0.0 (build 12359, commit `b1b770d9`), already in `~/llama.cpp/v1.0.0` | `scripts/build-llamacpp.sh` |

### Host stack: no changes needed

| Layer | On `formula` | Strata needs | Source of the requirement |
|---|---|---|---|
| Kernel | 7.0.0-43-generic | none stated | — |
| NVIDIA driver | 580.178.04 (open module) | 580 or newer for CUDA 13 engines | `docs/INSTALL.md` |
| CUDA toolkit | 13.0 (nvcc 13.0.88) | CUDA 13; **not 13.2.0/13.2.1** (nvcc 13.2.51 miscompiles kernels for sm_120); 13.0, 13.1 and 13.2.2 are safe | `docs/TROUBLESHOOTING.md` |
| Compiler | gcc 13.3 | C++20; CUDA 13.0 refuses gcc newer than 14 | `docs/SOURCE_INSTALL.md` |
| CMake / Ninja | 3.28 / installed | CMake 3.24+ | `engine/CMakeLists.txt` |
| GPU architecture | sm_120 | default build target ("RTX 50 only") | `docs/SOURCE_INSTALL.md` |
| NCCL, P2P | not used | the layer split hands off through pinned RAM | `docs/MULTI_GPU.md` |
| Python | 3.12.3, `python3.12-venv` installed | 3.10+, `terminal/requirements.txt` (pinned) in a venv | `terminal/requirements.txt` |
| RAM | 64 GB, 63.5 GB available | experts + about 10 GB: 43 + 10 = 53 GB, fully resident | `docs/MODELS.md` |
| Memory guards | `systemd-oomd` not installed | 0.2.17 notes: systemd-oomd can kill Strata while loading | release notes |
| Hugging Face CLI | not installed | for the pinned downloads | step 1 |

`ulimit -l` is about 8 GB. Strata pins its expert arena through CUDA, which does not count against `RLIMIT_MEMLOCK`, and no Strata doc mentions it. llama.cpp's `-lm dio` pinned more than that in `2026-11-llamacpp-flags`. Step 6 checks the startup log for "only N of M GiB could be page-locked" anyway.

## Why this model file

| Candidate | For | Against | Decision |
|---|---|---|---|
| **GSQ-RCO IQ3_XXS** (ISTA-DASLab) | Strata's recommended default, so its most tested path; experts 43 GB fit RAM fully; the same source quant exists as a Basalt pack (`Qwen3.8-FN-GSQ-RCO-IQ3_XXS.basalt`, HF revision `aa76f46`), so both experiments run the same weights | lowest quality of the options: top-1 84.46 %, mean KL 0.202 against Q8_0 (Strata `QUANTS.md`) | **used** |
| Unsloth UD-IQ4_XS | best quality of the options (KL 0.101) | Strata pins the pre-3.1 upload (revision `38bb39e`), Basalt's pack is built from Unsloth's 3.1 re-upload (2026-11-04), so not the same weights; 55 GiB experts exceed Strata's 64 GB budget (47 GiB), the rest is read from the SSD every token | no |
| Unsloth UD-IQ3_XXS | Basalt's recommended pack for 16 GB cards | not in Strata's model list (the engine refuses unsupported tensor formats at start); same revision mismatch as above | optional last step |

The quantization is a speed comparison between engines, not a quality ranking of quants. The quality check (step 11) only asks whether each engine and setting gives the same answers as the baseline.

## Settings that change the output

Each is set explicitly in every run and recorded in its `.cmd` file.

| Setting | Default in 0.2.21 | Here |
|---|---|---|
| Route tail skip `STRATA_ROUTE_TAIL_SKIP` | off since 0.2.17 | `0`, set explicitly; `7` only in step 9 |
| MTP draft layer `--mtp` | on in setup's config | off for the baseline, on as its own step |
| Lookup drafting (suffix/prompt lookup) | on, even without `--mtp`; no documented switch for "fully off" | recorded; whether `--spec 0` turns it off is tested in step 6 |
| `--spec N --spec-min-p P` | setup: `4` / `0.5` | setup's values when drafting is on |
| KV cache `--kv` | `int8` | `int8` (Basalt runs `int8` too) |
| CPU prompt share | on where it applies | default, recorded; `STRATA_PREFILL_CPU_SHARE=0` only for the quality check |
| Expert cache adaptation, PCIe share | on | default for speed runs; off (`--adapt-swaps 0 --pcie-frac 0`, fixed `--expert-cache N`, `STRATA_IQ_MT_MIN=1`, `--prompt-cache 0`) for the quality check |

Speculative decoding verifies every token, so it changes speed, not the answer. Its speed depends on the text: drafts are accepted more often in structured output than in prose (124.9 against 78.3 tok/s for one vLLM setup; 218.6 against 158.2 on one Basalt row). Every measurement therefore runs a prose and a structured prompt and reports them separately, never averaged.

## Method

- **Client:** a new `scripts/oai-bench.py`, the same for llama.cpp, Strata and Basalt. It talks to the OpenAI-compatible `/v1/chat/completions` endpoint with streaming, temperature 0, a fixed seed and 256 output tokens, and measures on the client side: time to first token, prompt tokens per second (from the server's `usage`), decode tokens per second between the first and the last token, and total time. The prompt text continues the lighthouse prompt of `2026-11-llamacpp-flags` (about 4,300 tokens) for prose; the structured prompt asks for a fixed JSON schema over the same text.
- **Repetitions:** one warm-up and 5 measured requests per configuration, mean ± standard deviation, as in `2026-11-llamacpp-flags`.
- **Per run:** load time, peak RAM, VRAM per GPU, GPU power and CPU package power (the samplers of `2026-11-llamacpp-flags/scripts/`), and the server's startup log, which shows the expert cache, the pinned RAM, the KV type and the drafting state.
- **Rejected runs:** OOM, swap use, "could not be page-locked", experts read from the SSD, or a setting that differs from the `.cmd` file. They stay in the journal and are rerun.
- **A real job, in minutes:** a fixed four-turn conversation over a long document (prompt about 32K tokens, growing each turn), timed from the first request to the last token. Headline tok/s overstate what an agent feels: in the video's coding task Basalt was 1.6× faster in tok/s but only 25 % faster in job time.

## Steps

1. **Tools.** By hand: a venv with a pinned `huggingface_hub` for `hf download`.
2. **Model.** Archive `qwen3.8-flash-next-iq3xxs` (batiai, used in `2026-11-llamacpp-flags`) to `/mnt/models` with `scripts/move-model.sh`. Download `IQ3_XXS/` at the pinned revision to `~/models/qwen3.8-flash-next-gsq-rco-iq3xxs/`, check sizes and SHA-256 against Hugging Face's LFS hashes, and write `REVISION` and `SHA256SUMS`.
3. **Build.** `scripts/build-strata.sh v0.2.21`: clone at the tag into `~/strata/v0.2.21/src`, build with `-DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_BUILD_TESTS=OFF`, target `strata`. Write `build-info.txt` like `build-llamacpp.sh` does (commit, toolchain, CMake cache, GPU code in the binary, linked libraries). Python venv with `terminal/requirements.txt`. Strata's `setup.sh` and `install.sh` are **not run**: they can install packages through apt. What setup does for this model is read from its source and repeated by hand, with each command in the journal.
4. **Model files for the engine.** The pack and tokenizer for IQ3_XXS as setup would prepare them (from step 3's reading), and the MTP draft layer (`mtp_fetch.py`, `mtp_pack.py`, `mtp_rt.py`).
5. **Server config.** `strata-gsq-rco-iq3xxs.json` with `"exe": "build/strata"`, `"env": {"STRATA_ROUTE_TAIL_SKIP": "0"}`, started with `terminal/.venv/bin/python terminal/serve/server.py --engine strata --config … --port 8080`.
6. **Smoke test, one GPU.** One short request. From the startup log: the route tail skip state, expert cache size, pinned RAM, KV type, drafting. Check whether `--spec 0` is accepted and turns drafting off.
7. **Baseline: llama.cpp on the same GGUF**, one GPU and two (layer split): llama.cpp's default (`--fit`) and the best configuration of `2026-11-llamacpp-flags` with the MoE cache re-tuned for this file. Same client and prompts.
8. **Strata, one GPU:** (a) no MTP, (b) MTP with setup's `--spec 4 --spec-min-p 0.5`.
9. **Strata, the route tail skip on** (`STRATA_ROUTE_TAIL_SKIP=7`) with 8b's settings: what it adds in speed, and whether the answers of step 11 change.
10. **Strata, two GPUs** (`"gpu": [0,1]`, `"layer_split": "auto"`): 8a and 8b again. The log shows the split (for example "layers 0-18 (CUDA0), 19-47 (CUDA1)").
11. **Quality check.** Ten fixed prompts (facts, arithmetic, code, a JSON schema), greedy, with determinism settings on. Answers compared across llama.cpp, Strata skip off and Strata skip on. The aim is to catch broken or drifting output, not to rank quants.
12. **Long context and the real job,** with the fastest exact configuration of steps 8–10: time to first token at 32K and 128K (3 runs each), and the four-turn job (3 runs).
13. **Two GPUs, every mode** (added after step 12; the main question):
    - a. llama.cpp's two-GPU "best" at long context (32K, 128K) and in the four-turn job: the reference for 12.
    - b. llama.cpp's other split modes with the MoE cache: `-sm row`, `-sm tensor`, and an uneven `-ts`.
    - c. Strata's layer split at long context (32K, 128K) against the helper cache of step 12: whether the second card helps prompt reading.
    - d. Strata's layer split tuned: a manual split point against `auto`, `STRATA_SPLIT_OVERLAP=1`.
    - e. Several requests at once (2 and 4): aggregate tokens per second for llama.cpp (`-np`) and Strata (`"parallel"`) in their two-GPU modes.
    - f. The result: the best two-GPU setup per use (one chat, structured output, long prompts, several requests) and what the second card adds in each.
14. **Optional: Unsloth UD-IQ3_XXS** through `iq_pack.py`, to see whether the engine accepts it (format check at start). If it does, the same file is the second Basalt model and both experiments get a second like-for-like pair.
15. **Write-up:** results table, findings, `docs/stack.md` (Strata build), README index.

## Disk

| | Size | Where |
|---|---|---|
| GSQ-RCO IQ3_XXS GGUF | 75.8 GB | `~/models/` |
| MTP draft layer | about 6 GB | `~/strata/v0.2.21/` |
| Strata build and venv | a few GB | `~/strata/v0.2.21/` |
| batiai IQ3_XXS (archived in step 2) | 80 GB | `/mnt/models/` (139 GB free before) |

The NVMe has 279 GB free before step 2 and about 280 GB after it (80 GB out, 82 GB in).

## Results

Preliminary (steps 7–13; the write-up is step 15). Every number traces to `runs/` and the journal.

**One GPU** (same weights, 4.3K-token prompt, 256 tokens out):

| Engine | Prompt t/s | Decode t/s, prose | Decode t/s, JSON |
|---|---|---|---|
| llama.cpp v1.0.0, best (`-cmoe`, 9.25 GiB MoE cache, `-lm dio`, `-ub 4096`) | 872 | 45.2 | 42.4 |
| Strata, exact (no drafting, nothing skipped) | 1,779 | 62.8 | 62.9 |
| Strata, MTP | 1,738 | 77.8 | 118.2 |

**Two GPUs, the best setup per use:**

| Use | Best setup | Result | Second card adds | llama.cpp, two GPUs |
|---|---|---|---|---|
| One chat, exact | Strata helper cache | 66.9 t/s | +7 % | 53.1 t/s |
| One chat, MTP, prose / JSON | Strata helper cache + MTP / layer split + MTP + pipeline windows | 93.4 / 159.2 t/s | +20 % / +35 % | – |
| 126K-token prompt | Strata layer split | first token after 36.3 s | 1.7× | 153.4 s |
| Four-turn job over a 32K document | Strata layer split | 29.0 s | – | 64.2 s |
| 4 / 8 requests at once (short prompts) | Strata layer split + 4 / 8 slots | 173 / ≈ 229 t/s total | 1.66× (4) | 117 / 151 t/s |
| All-round, one config | Strata layer split + MTP + 4 slots | 87.6 / 133.7 t/s one chat, 172 t/s for 4 | | |

Quality (step 11): 100 of 100 answers correct on llama.cpp and every Strata setting; short factual answers identical, longer ones differ at near-ties.

## Findings

Pending.
