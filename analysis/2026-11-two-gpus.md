# Two RTX 5060 Ti 16 GB: every engine, every two-GPU mode (November 2026)

A cross-experiment analysis of five experiments run on `formula` (i7-14700, 64 GB DDR5-6000, 2× RTX 5060 Ti 16 GB on CPU lanes, Gen5 x8 each, no P2P) between 2026-11-20 and 2026-11-27. The question: **which configuration gets the most tokens per second out of each engine when both cards work at once, and how do the best configurations compare?**

Sources (every number below is in one of them; all were rechecked against the raw run files on 2026-11-27):

| Experiment | Engines | Models |
|---|---|---|
| [`2026-11-boardswap`](../experiments/2026-11-boardswap/) | llama.cpp (llama-server, llama-bench), Ollama | dense 4B–32B, Qwen3 30B-A3B, Qwen3.8 27B (GGUF Q4_K_M) |
| [`2026-11-llamacpp-flags`](../experiments/2026-11-llamacpp-flags/) | llama.cpp | Qwen3.8-Flash-Next (batiai IQ3_XXS), DeepSeek-V4-Flash (UD-IQ1_M) |
| [`2026-11-strata`](../experiments/2026-11-strata/) | Strata, llama.cpp | Qwen3.8-Flash-Next (GSQ-RCO IQ3_XXS) |
| [`2026-11-basalt`](../experiments/2026-11-basalt/) | Basalt | Qwen3.8-Flash-Next (GSQ-RCO and UD IQ3_XXS packs) |
| [`2026-11-vllm-27b-tp2`](../experiments/2026-11-vllm-27b-tp2/) | vLLM, Basalt, llama.cpp (step 17) | Qwen3.8-27B (NVFP4; GGUF Q4_K_M for llama.cpp) |

## 1. The answer

**Engines can only be ranked per model.** Not every engine ran every model, and a 27B dense model, an 80B-class MoE and an 8B model are different problems on 16 GB cards. Where two engines did run the same model, the weights and prompts are the same (marked otherwise).

Which engine ran which model, on two GPUs:

| Engine | Qwen3.8-Flash-Next (MoE) | Qwen3.8-27B (dense) | dense 8B–32B, Qwen3 30B-A3B |
|---|---|---|---|
| llama.cpp | yes (`2026-11-strata`, `2026-11-llamacpp-flags`) | yes (vLLM step 17; Boardswap with short prompts) | yes (Boardswap) |
| Strata | yes | **cannot**: v0.2.21's setup builds packs only for MoE families (Flash-Next variants, Qwen3.6-35B-A3B, HY4, Inkling) and its Qwen path accepts only the `qwen35moe` architecture; the 27B is `qwen35` (dense) | not supported |
| Basalt | yes | yes (split mode; `peer` refused) | not run |
| vLLM | not run | yes | not run |
| Ollama | not run | yes (Boardswap, short prompts, layer split only) | yes (Boardswap) |

### Best two-GPU configuration per engine

| Engine | Model | Best two-GPU configuration | One chat, t/s (prose / JSON) | Several requests, total t/s | Prompt |
|---|---|---|---|---|---|
| **Basalt** | Qwen3.8-Flash-Next | `peer` + the pack's MTP drafter, int8 KV, `parallel` 4 | **113.1 / 193.3** | 173 (4, short), 200.5 (8, short) | 1,855 t/s; 126K in 34.7 s (no drafter) |
| **Strata** | Qwen3.8-Flash-Next | layer split + MTP + `parallel` 4 (one config for everything); for one chat only: helper cache + MTP (prose), layer split + MTP + pipeline windows (JSON) | 87.6 / 133.7 (all-round); 93.4 / 159.2 (best per kind) | 172 (4, short); **229 (8, short, no drafter)** | 1,794 t/s; 126K in 36.3 s (no drafter) |
| **llama.cpp** | Qwen3.8-Flash-Next | layer split, `-cmoe` + MoE cache 23,552 MiB, `-lm dio`, `-b/-ub 4096` | 53.1 / 52.9 (no drafting) | 117 (4, short), 151 (8, short) | 838 t/s; 126K in 153.4 s |
| **vLLM** | Qwen3.8-27B | TP=2 + MTP k=4: fp8 KV on v0.30.0 for one chat; the NVFP4-KV recipe (v0.32.0) with `max-num-seqs` 4 for several requests | **64.8 / 116.4** | **94.0 / 120.2** (4, 4.3K prompts, NVFP4 KV) | 3,126 t/s at 4K; 252K in 272 s (k=6) |
| **Basalt** | Qwen3.8-27B | `split` + DFlash 2 drafter, int8 KV | 46.6 / **128.0** | 84.1 / 103.3 (4, 4.3K prompts; 163,840 context) | **4,272 t/s at 4K; 252K in 114.5 s** |
| **llama.cpp** | Qwen3.8-27B (Q4_K_M) | `-sm tensor` + MTP, default batch | 55.5 / 100.8 | 49.0 / 57.6 (4, 4.3K prompts) | 1,320 t/s at 4K; 252K in 360 s (q8_0 KV) |
| **llama.cpp** | dense 8B–32B | `-sm tensor` | 8B 130.3, 14B 74.3, 32B 35.8 | not measured | 8B 4,518 t/s |
| **Ollama** | dense 8B–32B, 27B | layer split (`OLLAMA_SCHED_SPREAD=1`; no tensor split) | 8B 82.2, 14B 44.0, 32B 20.1; 27B + MTP 42.8 | not measured | 8B 3,751 t/s |

### Winner per model and use

| Model | One chat | Structured output (JSON) | Long prompts | Several requests |
|---|---|---|---|---|
| Qwen3.8-Flash-Next (MoE) | **Basalt** peer + MTP: 113 t/s prose, 1.2× Strata's best, 2.1× llama.cpp's | **Basalt** peer + MTP: 193 t/s | **Basalt** peer: 126K in 34.7 s (Strata 36.3, llama.cpp 153.4) | up to 4: Basalt ≈ Strata (173 each); 8: **Strata** layer split (229 against 200.5) |
| Qwen3.8-27B (dense) | **vLLM** TP=2 + MTP: 64.8 t/s prose (llama.cpp 55.5, Basalt 46.6) | **Basalt** split + DFlash 2: 128.0 t/s (vLLM 116.4, llama.cpp 100.8) | **Basalt** split: 2.4× vLLM at 32K, 2.4–2.5× at 252K (llama.cpp 3.1× slower than Basalt) | **vLLM** TP=2, 4 slots (NVFP4-KV recipe): 94.0 prose (Basalt 84.1, llama.cpp 49.0) |
| dense 8B–32B | **llama.cpp** tensor split: 1.6–1.7× one card (8B–14B); Ollama cannot split tensors | same | llama.cpp tensor split | not measured |

The same ordering holds when both engines read **one file** (step 16 of the vLLM experiment: `nvidia/Qwen3.8-27B-NVFP4` in both): vLLM +46 % plain decode and +28 % drafted prose, Basalt +15 % drafted JSON and 2.4–2.5× on prompts of 32K and more.

## 2. The math: how to read these numbers

### What each number measures

| Number | Formula | Where it comes from |
|---|---|---|
| **Decode t/s** (one stream) | (generated tokens − 1) / (time of last token − time of first token) | the client (`scripts/oai-bench.py`), the same for every engine |
| **Total t/s** (several requests) | all streams' generated tokens / wall time of the group (first request sent → last stream closed) | the client; includes the time spent reading the prompts |
| **Prompt t/s** | prompt tokens / time to first token, or the server's own figure | server-measured for llama.cpp, Strata, Basalt; client-measured for vLLM (about 0.5 % lower for the same work) |
| **TTFT** | time from request to first content token | the client |
| **Acceptance length** | 1 + accepted draft tokens / drafts | the server's speculative-decoding counters |

All Flash-Next and 27B runs (llama.cpp's included, step 17) use the same prompts: about 4,300 tokens in (a prose task and a JSON task on the same document), 256 tokens out, greedy, thinking off, nothing reused between requests. "Short" concurrency runs use a 63-token prompt. Boardswap's llama.cpp and Ollama runs use a ~540-token prompt (llama-server, Ollama) or llama-bench's synthetic 512-token prompt and 128 tokens.

### Drafting: tokens per second = steps per second × acceptance length

Speculative decoding (MTP, DFlash 2, lookup) proposes several tokens and the model checks them in one step. A step costs a bit more than a plain step, and yields on average the acceptance length in tokens. vLLM's k sweep shows both parts exactly: one step takes **25.3 ms + 2.64 ms per draft token** (k = 1–7), so steps/s falls from 42.07 (k=0) to 27.84 (k=4) to 22.81 (k=7), while the tokens per step depend on the text: 2.29 for prose, 4.22 for JSON at k=4. Hence

- prose: 27.84 × 2.29 ≈ 64 t/s, JSON: 27.84 × 4.22 ≈ 117 t/s (measured 63.9 and 117.6),
- **prose and JSON must always be reported apart**: the same server runs 1.8× faster on JSON because JSON repeats the prompt's structure and the drafter guesses it,
- the best k depends on the text: prose peaks at k=2–3 (67.9 t/s), JSON keeps rising to k=7 (133.9 t/s).

### Why a second card adds little to one chat (and a lot to other things)

Decoding one token reads every active weight once. On these cards that is a memory-bandwidth problem (448 GB/s each), not a compute problem. What the second card can do depends on how the model is split:

| Split | What runs where | One chat | Prompts | Several requests |
|---|---|---|---|---|
| **Layer split / pipeline** (llama.cpp `-sm layer`, Ollama, Strata layer split, Basalt `split`, vLLM PP=2) | each card holds half the layers; for every token the cards take turns | **no faster** than one card for a dense model (llama.cpp 8B: 82.3 against 82.4 t/s); the gain on MoE comes from the extra VRAM for experts | can overlap chunks: Basalt split reads 32K 2.4× faster than vLLM TP | both cards busy at once: the best total throughput (Strata 8 requests: 229 t/s) |
| **Tensor split** (llama.cpp `-sm tensor`, vLLM TP=2) | every layer split across both cards; each card reads half the weights per token, then the halves are summed (all-reduce) | up to 2×; measured 1.6–1.7× on dense 8B–14B (llama.cpp; the 32B has no one-card baseline), because every layer waits for an all-reduce through host memory (no P2P on these cards) | the all-reduce is about half of every prompt read (vLLM profile: 56–57 % of a 4,220-token prefill) | yes, batched |
| **Peer** (Basalt) / **helper cache** (Strata) | the main card runs every layer; the second card holds experts (Basalt: 34 % of them) and computes their rows | +7–13 % (Basalt), +7–35 % (Strata, more with MTP) | Basalt peer reads part of long prompts on the second card: 126K 1.6× faster than one card | Basalt: +16 % at 4 short requests without drafting |

For the MoE model the bottleneck is different: Flash-Next's experts (40 GiB at IQ3_XXS) do not fit 32 GB of VRAM, so the engines keep the most-used experts on the GPUs and compute misses on the CPU. With 90 % of routed experts already in one card's cache (Strata), a second card's extra expert capacity buys only a few per cent; what it buys more of is parallel work (prompts, batches).

### Why "4 requests" totals differ so much between short and long prompts

Total t/s counts the prompt reading in the wall time. Example, Strata layer split, 4 requests of 4,326 prompt tokens and 256 output tokens each: 1,024 tokens out over a measured 61.7 t/s is 16.6 s of wall time, of which reading the four prompts at 1,878 t/s takes about 9.2 s. With 63-token prompts the same configuration reaches 173.1 t/s. **Totals are only comparable at the same prompt length**, and this analysis keeps the two apart.

### How to read a speed-up

A "1.6×" in this document is a ratio of the same metric on the same prompts, two configurations of the same engine and model unless stated (for example 126K TTFT on one card / on two cards). Percentages are (new − old) / old. Differences smaller than the run-to-run spread (the ± after a mean, usually 1–8 % with drafting, under 1 % without) are not findings.

## 3. Per engine

### 3.1 llama.cpp (v1.0.0 for the MoE runs, build 12464 for the dense runs)

**Qwen3.8-Flash-Next, GSQ-RCO IQ3_XXS** (`2026-11-strata`, steps 7 and 13; t/s prose / JSON):

| GPUs | Configuration | Prompt t/s | Decode t/s |
|---|---|---|---|
| 1 | `--fit` (default) | 278.0 | 28.0 / 27.3 |
| 1 | best: `-cmoe`, MoE cache 9,472 MiB, `-lm dio`, `-b/-ub 4096` | 871.7 | 45.2 / 42.4 |
| 2 | `--fit` (default) | 423.5 | 36.0 / 35.9 |
| 2 | **best: layer split, `-cmoe`, MoE cache 23,552 MiB, `-lm dio`, `-b/-ub 4096`** | 837.6 | **53.1 / 52.9** |
| 2 | same with `-ts 45,55` | 827.6 | 53.2 / 52.3 |
| 2 | layer split, `-cmoe`, no MoE cache | 627.4 | 22.8 / 22.9 |
| 2 | tensor split (`-sm tensor`, no MoE cache possible) | 94.5 | 20.6 / 21.0 |
| 2 | row split | does not load | |

Second card, best against best: decode +17 % / +25 %, prompt −4 %. Long prompts (2 GPUs, best, 131K context): 32K in 39.7 s, 126K in 153.4 s (826 t/s). Several requests (`-np 4` / `-np 8`): 4.3K prompts 32.1 (2) and 37.2 (4) t/s total; short prompts 85.5 (2), 117.1 (4), 151.4 (8).

**The setting-by-setting series** (`2026-11-llamacpp-flags`, two GPUs, ~4,260-token prompt, server-measured; Qwen batiai IQ3_XXS, DeepSeek-V4-Flash UD-IQ1_M):

| Step | Options added | Qwen prompt / generation t/s | DeepSeek prompt / generation t/s |
|---|---|---|---|
| 0a | naive layer offload (`-fit off -ngl N`) | 334.6 / 16.8 | 131.8 / 6.4 |
| 0b | default `--fit` | 340.9 / 38.8 | 145.7 / 14.5 |
| 1 | `-cmoe` | 202.7 / 24.4 | 93.7 / 10.7 |
| 2 | + MoE cache (tuner's size) | 232.5 / 48.1 | 112.6 / 18.7 |
| 2b | + twice the tuner's cache | 258.0 / 54.4 | 126.4 / 21.4 |
| 3 | + `-lm dio` | 320.2 / 57.8 | not possible (experts exceed RAM) |
| 4 | + `-b/-ub 4096` | 804.3 / 58.0 | 230.5 / 19.9 |
| best | | **803.1 / 59.5** (CPU share 0) | **244.4 / 20.4** |

The same model is about 12 % faster at generation with the batiai quant (59.5, server-measured) than with GSQ-RCO (53.1, client-measured): different quants and meters, not a different engine.

**Dense models** (`2026-11-boardswap`, after the swap, llama-server, ~540-token prompt; generation / prompt t/s):

| Model | 1 card | 2 cards, layer split | 2 cards, tensor split | tensor / 1 card |
|---|---|---|---|---|
| Llama 3.1 8B Q4_K_M | 82.4 / 3,745 | 82.3 / 4,023 | **130.3 / 4,518** | 1.58× |
| Qwen3 8B | 77.6 / 3,614 | 77.5 / 3,895 | **123.6 / 4,261** | 1.59× |
| Qwen3 14B | 43.9 / 2,010 | 43.9 / 2,150 | **74.3 / 2,613** | 1.69× |
| Qwen3 32B | – | 20.1 / 927 | **35.8 / 1,276** | (needs 2 cards) |
| Qwen3.8 27B | – | 24.2 / 838 | 40.4 / 986 | (needs 2 cards) |
| Qwen3.8 27B + MTP | – | 52.1 / 824 | **77.8 / 951** | |
| Qwen3 30B-A3B (MoE) | – | 143.8 / 4,023 | **158.2** / 3,622 | |

**Qwen3.8-27B with this analysis' prompts** (`2026-11-vllm-27b-tp2` step 17; llama.cpp v1.0.0, the same Q4_K_M GGUF; 4.3K-token prompt, 256 tokens; prose / JSON t/s):

| Configuration | Decode | TTFT 4K / 32K | 4 requests, total | 252K needle |
|---|---|---|---|---|
| tensor split | 40.3 / 40.4 | 3.04 / 24.15 s | 49.4 / 49.3 | |
| **tensor split + MTP** | **55.5 / 100.8** | 3.19 / 25.56 s | 49.0 / 57.6 | 359.9 s (q8_0 KV) |
| tensor split + MTP, `-b/-ub 4096` | 57.9 / 100.0 | 3.34 / 26.83 s | | |
| layer split + MTP | 37.4 / 66.9 | 3.06 / 24.95 s | | |

At 540 tokens the same tensor split + MTP decodes at 77.8 t/s (Boardswap); at 4.3K, 55.5. Decode slows with context, and prose drafts worse than the short benchmark text. MTP's output equals the plain output on all ten quality prompts.

### 3.2 Ollama (0.32.14, `2026-11-boardswap`)

Ollama splits by layers only (`OLLAMA_SCHED_SPREAD=1`), so on two cards it runs at one card's speed and is worth it only when a model needs both cards' memory. It generates within 1 % of llama-server in the same mode (Llama 3.1 8B 82.2 against 82.3; Qwen3 32B 20.1 against 20.1), reads prompts 2–7 % slower, and loads in 15–33 s against 2–12 s. On Qwen3.8 27B its default MTP gives 42.8 t/s (llama-server, same layer split: 52.1; tensor split + MTP: 77.8).

### 3.3 Strata (v0.2.21, engine 0.1.67; Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS)

| GPUs | Mode | Exact (prose / JSON) | MTP prose / JSON | Prompt t/s |
|---|---|---|---|---|
| 1 | one card | 62.8 / 62.9 | 77.8 / 118.2 | 1,779 |
| 1 | one card, lookup drafting | 62.7 / 77.4 | | 1,775 |
| 1 | one card, route tail skip 7 | 62.7 / 63.8 | 81.7 / 125.0 | 1,772 |
| 2 | layer split (auto, cut at 26) | 64.8 / 64.8 | 86.2 / 132.4 | 1,878 |
| 2 | layer split, cut at 20 / 32, overlapped hand-off | 64.6–64.9 | | 1,872–1,882 |
| 2 | layer split + MTP + pipeline windows 2 | | 90.6 / **159.2** | 1,777 |
| 2 | … with the draft layer on CUDA0 | | 77.9 / 156.5 | 1,759 |
| 2 | helper cache (`--expert-cache-device1 auto --remote-expert-opt`) | **66.9 / 66.6** | **93.4** / 138.3 | 1,757 |
| 2 | **recommended: layer split + MTP + `parallel` 4** | | 87.6 / 133.7 | 1,794 |

Long prompts: 32K in 15.5 s (1 GPU) / 10.5 s (2 GPUs), 126K in 61.8 s / **36.3 s** (1.70×). The four-turn job over a 32K document: 29.0 s (2 GPUs).

Several requests (total t/s at 2 / 4 / 8):

| Configuration | 4.3K prompts | Short prompts |
|---|---|---|
| 1 GPU, `parallel` 4 | 39.5 / 52.4 | 73.8 / 104.5 |
| 2 GPUs, layer split, `parallel` 4 | 46.8 / **61.7** | 113.1 / **173.1** |
| 2 GPUs, helper cache, `parallel` 4 | 44.3 / 58.4 | 90.0 / 132.8 |
| 2 GPUs, layer split + MTP, `parallel` 4 | | 112.4 / 172.2 |
| 2 GPUs, layer split + MTP + `--batch-mtp` | | 106.6 / 129.7 |
| 2 GPUs, layer split, `parallel` 8 | | – / – / **229.4** |

Second card: one chat +7 % exact, +20 % / +35 % with MTP; 126K prompt 1.70×; 4 short requests 1.66×. MTP stops helping once several requests run (the pipelined batch path does not draft).

### 3.4 Basalt (v0.5.7)

**Qwen3.8-Flash-Next, GSQ-RCO IQ3_XXS pack** (`2026-11-basalt`):

| GPUs | Mode | Exact (prose / JSON) | MTP prose / JSON | Prompt t/s | 126K TTFT | 4 short requests, exact / MTP |
|---|---|---|---|---|---|---|
| 1 | one card | 72.3 / 71.7 | 104.8 / 170.5 | 1,872 | 54.6 s | 125.2 / – |
| 2 | **peer** | **77.4 / 78.4** | **113.1 / 193.3** | 1,876 | **34.7 s** | **145.8 / 173.4** |
| 2 | split | 71.0 / 71.5 | 107.4 / 184.5 | 1,855 | 35.8 s | 120.6 / 164.7 |
| 2 | split, `gpu-share` 0.4,0.6 / 0.6,0.4 | 70.4–71.3 | | 1,781–1,860 | | |
| 2 | peer, `gpu-share` 1,1 | 77.6 / 78.5 | | 1,845 | | |
| 1 / 2 | route tail skip 7 (one card exact / peer + MTP) | 72.8 / 72.9 | 115.6 / 209.7 | 1,875 / 1,858 | | |
| 1 / 2 | UD-IQ3_XXS pack (one card / peer) | 68.2 / 68.2 → 72.6 / 72.2 | 98.1 / 169.9 → 112.3 / 200.9 | 1,756 / 1,750 | | |

Eight short requests: split 147.4, peer 189.8, **peer + MTP 200.5** (peer needs `vram-reserve-mib` 1500 for 8 slots). 4.3K prompts, 4 requests, no drafter: one card 54.4, peer 56.4, split 58.2. The four-turn job: one card 30.9 s, split 27.8 s, **peer 25.4 s**.

Second card (peer against one card): one chat +7–13 %, prompt rate unchanged, 126K prompt 1.57×, 4 short requests +16 % (exact), job 1.22×.

**Qwen3.8-27B** (`2026-11-vllm-27b-tp2`, steps 10, 13, 15, 16; split mode only, `peer` refused: the main card would have to hold the whole 16 GiB model):

| Weights | Configuration | Context | No drafter | DFlash 2, prose / JSON | TTFT 4K / 32K (no drafter) | 252K needle |
|---|---|---|---|---|---|---|
| QUASAR NVFP4 | split, `parallel` 1 | 262,144 | 24.95 | 46.6 / **128.0** | 0.940 / 5.193 s | **114.5 s** (drafter on) |
| nvidia NVFP4 (matched) | split, `parallel` 1 | 200,000 (drafter), 254,000 (no drafter) | 21.93 | 41.8 / 111.1 | 1.101 / 5.897 s | 110.0 s (no drafter) |

Several requests (QUASAR, 4.3K prompts, total t/s at 1 / 2 / 4): no drafter, 262,144: 22.9 / 39.5 / 64.2; DFlash 2, 163,840: 39.6 / 57.5 / 84.1 (prose), 89.6 / 90.3 / 103.3 (JSON).

### 3.5 vLLM (Qwen3.8-27B, two GPUs only: the model does not fit one card)

Pinned checkpoint `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090@0cc2795`, v0.30.0, fp8 KV, driver 595.91.07 unless stated:

| Configuration | Decode prose / JSON | TTFT 0.5K / 4K / 32K | KV pool | 252K needle |
|---|---|---|---|---|
| TP=2, MTP off | 42.35 / 42.33 | 0.184 / 1.310 / 12.404 s | 351,618 | |
| **TP=2, MTP k=4** | **64.76 / 116.44** | 0.197 / 1.347 / 13.112 s | 278,927 | 271.95 s (k=6) |
| TP=2, MTP k=4, NVFP4 KV (v0.32.0, util 0.94; the published recipe) | 61.75 / 113.79 | – / 1.369 / 13.133 s | 474,032 | 279.09 s |
| TP=2, matched checkpoint (`nvidia/Qwen3.8-27B-NVFP4`), MTP off / k=4 | 32.06 / 53.64 / 96.79 | 0.225 / 1.557 / 14.263 s | 266,853 | 272.34 s (MTP off) |

MTP draft length (phase A, k sweep, 640 tokens; prose / JSON / mixed t/s): k=0 42.1 · k=1 60.1 / 69.6 / 62.6 · k=2 **67.9** / 91.6 / 72.1 · k=3 67.6 / 107.2 / **73.6** · k=4 63.9 / 117.6 / 73.0 · k=5 59.4 / 123.0 / 68.9 · k=6 56.2 / 127.7 / 62.8 · k=7 53.0 / **133.9** / 58.5.

Two GPUs, other ways (step 8, the NVFP4 recipe, thinking off; total t/s at 1 / 2 / 4 requests, 4.3K prompts):

| Configuration | Prose | JSON | Decode per stream, 1 request | 32K TTFT |
|---|---|---|---|---|
| TP=2, `max-num-seqs` 1 | 44.1 / 43.9 / 44.5 | 67.5 / 68.7 / 68.9 | 58.2 / 108.1 | 13.16 s |
| TP=2, `max-num-seqs` 2 | 44.2 / 68.6 / 70.0 | 68.5 / 96.9 / 97.1 | 58.1 / 110.7 | 13.14 s |
| **TP=2, `max-num-seqs` 4** | **45.6 / 69.4 / 94.0** | 67.6 / 97.2 / **120.2** | 60.7 / 108.1 | 13.16 s |
| PP=2, `max-num-seqs` 4, 65,536 context | 31.3 / 53.2 / 86.0 | 52.2 / 81.2 / 128.9 | 36.0 / 66.8 | **9.29 s** |

(Total t/s at 1 request includes the prompt; decode per stream does not.) TP=2 gives each stream 1.7× the speed of PP=2; PP=2 reads prompts 1.4× faster and catches up at 4 requests (JSON 128.9 against 120.2), but holds 65,536 tokens of context against 262,144.

## 4. Head to head, best against best

### 4.1 Qwen3.8-Flash-Next (MoE; same GSQ-RCO IQ3_XXS weights in all three)

| | llama.cpp best | Strata best | Basalt best |
|---|---|---|---|
| One chat, no drafting | 53.1 / 52.9 | 66.9 / 66.6 (helper) | **77.4 / 78.4** (peer) |
| One chat, drafting | – (no MTP for this model in llama.cpp v1.0.0) | 93.4 (helper + MTP) / 159.2 (layer split + MTP + pipeline windows) | **113.1 / 193.3** (peer + MTP) |
| Prompt, 4.3K tokens | 838 t/s | **1,878** | 1,876 |
| 126K-token prompt, TTFT | 153.4 s | 36.3 s | **34.7 s** |
| Four-turn job, 32K document | 64.2 s | 29.0 s | **25.4 s** |
| 4 requests, 4.3K prompts, no drafting | 37.2 | **61.7** | 58.2 (split) |
| 4 requests, short prompts | 117.1 | 173.1 | **173.4** (peer + MTP) |
| 8 requests, short prompts | 151.4 | **229.4** | 200.5 |
| Peak VRAM per card | 15,784 / 15,046 MiB | 15,030 / 14,934 MiB (recommended) | 15,372 / 15,532 MiB (peer + MTP) |

Basalt against Strata, one chat: +16 % / +18 % exact, +21 % / +21 % drafted (best against best), the prompt rate equal. Basalt against llama.cpp: 1.5× exact, 2.1× / 3.7× drafted, 2.2× on prompts. Strata's layer split serves eight short requests best (229 t/s); at four they tie.

### 4.2 Qwen3.8-27B (dense)

| | vLLM best (TP=2) | Basalt best (split) | llama.cpp best (tensor split + MTP, step 17) |
|---|---|---|---|
| Weights | NVFP4 (pinned / matched) | NVFP4 (QUASAR / matched) | GGUF Q4_K_M |
| One chat, no drafting | **42.35** / 32.06 | 24.95 / 21.93 | 40.30 |
| One chat, drafting, prose | **64.76** / 53.64 | 46.57 / 41.76 | 55.47 |
| One chat, drafting, JSON | 116.44 / 96.79 | **127.99** / 111.13 | 100.84 |
| TTFT 4K / 32K (no drafting) | 1.310 / 12.404 s | **0.940 / 5.193 s** | 3.036 / 24.150 s |
| 252K needle | 271.95 s | **114.5 s** | 359.85 s (q8_0 KV) |
| 4 requests, 4.3K prompts, prose / JSON | **94.0** / **120.2** (NVFP4-KV recipe, `max-num-seqs` 4) | 84.1 / 103.3 (QUASAR, 163,840 context) | 49.0 / 57.6 (`-np 4`) |
| Context held | 262,144 (fp8 KV) | 262,144 (int8 KV) | 262,144 (q8_0 KV) |
| Drafting changes the text (10 prompts) | 3–4 of 10 | 0 of 10 | 0 of 10 |

vLLM against Basalt on the matched checkpoint: +46 % plain decode, +28 % drafted prose; Basalt +15 % drafted JSON and 1.4× / 2.4× / 2.5× faster prompts at 4K / 32K / 252K.

Strata is not in this table because it cannot run the dense 27B (section 1). llama.cpp ran other weights (Q4_K_M against NVFP4), so its column compares engines and quantizations together; prompts, client and driver are the same. It decodes at 86–87 % of vLLM's speed, above Basalt on prose and below on JSON, holds the full context, and its drafted output does not change; it reads prompts 2.3× slower than vLLM at 4K and 3.1× slower than Basalt at 252K, which also caps its several-request totals.

### 4.3 Dense models up to 32B

Only llama.cpp and Ollama ran them. llama.cpp's tensor split gives 1.58–1.69× one card (79–88 % of the ideal 2× after the board swap) and is the only mode where the second card speeds up one chat; Ollama's layer split equals one card.

## 5. What did not matter (or barely)

| Factor | Effect | Source |
|---|---|---|
| Driver 580.178.04 → 595.91.07 | vLLM speed unchanged (0.5–5 %, inside the spread), KV pools +0.9–2.3 % (now equal to the source's); Strata and Basalt within 0.4 % on prompts | vLLM 9a–9e |
| 150 W power cap (from 180 W) | decode unchanged, 32K prompt +1.4 %, 252K needle +0.4 % (fp8) and +2.9 % (NVFP4, the only load that drew more than 150 W) | vLLM 8b, 14 |
| GPU1 on chipset Gen4 x4 → CPU Gen5 x8 | layer split and one card within ±1 %; tensor split generation +7 to +21 %, prompt +16 to +39 % | Boardswap |
| `NCCL_P2P_DISABLE` | none: these cards have no P2P; NCCL goes through host memory either way | vLLM 9d |
| NCCL protocol | the default is already the fastest (7.5 GB/s; LL 3.0, LL128 4.3) | vLLM 9g |
| Split point, `gpu-share`, `-ts` | none within the spread | Strata 13d, 13b; Basalt 11b |
| Route tail skip | 0–8 % faster for 1.5–6 % of routed experts left out; changed one of ten answers in Strata | Strata 9, Basalt 8 |
| `--batch-mtp` (Strata) | slower (129.7 against 172.2 for 4 requests) | Strata 13e |
| Twice the MoE cache tuner's size (llama.cpp) | +13–14 % generation: the tuner sizes the cache from the first GPU only | llama.cpp flags 2b |

## 6. Gaps and runs worth adding

1. ~~llama.cpp tensor split + MTP on Qwen3.8-27B with the 4.3K prompts~~: done (vLLM experiment, step 17). Still open: Ollama on the 27B with these prompts, and llama.cpp on NVFP4 weights (no GGUF of them).
2. **Drafted several-request runs with 4.3K prompts on Flash-Next** (Strata and Basalt ran their long-prompt concurrency without drafting).
3. **vLLM on Flash-Next**: not run (vLLM's MoE path with experts beyond 32 GB of VRAM was not part of these experiments).
4. vLLM's MTP config on the matched checkpoint at the source's util 0.95 with 180,000 context (vLLM 16a).
5. Distribution-level quality (KL against BF16) for the quantizations and KV types; the ten-prompt checks only show that no setting broke an answer.

## 7. Recommended setups

| Want | Engine and model | Configuration file |
|---|---|---|
| Fastest single user on Flash-Next | Basalt peer + MTP | [`basalt-peer-mtp-par4.json`](../experiments/2026-11-basalt/configs/basalt-peer-mtp-par4.json) (UD pack for quality: `basalt-ud-peer-mtp.json`) |
| Many users (8+) on Flash-Next | Strata layer split, `parallel` 8 | [`2026-11-strata` recommended setup](../experiments/2026-11-strata/README.md#the-recommended-two-gpu-setup) |
| Flash-Next with llama.cpp | layer split + MoE cache 23,552 MiB, `-lm dio`, `-b/-ub 4096`, CPU share 0 | [`2026-11-llamacpp-flags`](../experiments/2026-11-llamacpp-flags/README.md) |
| Qwen3.8-27B, one chat | vLLM TP=2, fp8 KV, MTP k=4 (k=2–3 for prose only) | [`fp8-mtp4.args`](../experiments/2026-11-vllm-27b-tp2/configs/fp8-mtp4.args) |
| Qwen3.8-27B, several users | vLLM TP=2, NVFP4 KV, MTP k=4, `max-num-seqs` 4 (fp8 KV with 4 slots was not measured) | [`nvfp4-mtp4-seqs4.args`](../experiments/2026-11-vllm-27b-tp2/configs/nvfp4-mtp4-seqs4.args) |
| Qwen3.8-27B with long documents or JSON | Basalt split + DFlash 2 | [`basalt27-split.json`](../experiments/2026-11-vllm-27b-tp2/configs/basalt27-split.json) |
| Dense models up to 32B | llama-server `-sm tensor` (custom image with NCCL 2.32.6), + MTP where the model has a head | [`2026-11-boardswap` engines](../experiments/2026-11-boardswap/engines.md) |
