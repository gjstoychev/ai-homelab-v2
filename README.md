# ai-homelab-v2

An open bench for local LLM inference on a two-node home lab: inference engines, drivers, flags and configurations, each tried on real hardware, measured and written up, including the mistakes.

Write-ups are published at https://gshell.dev. The structured curriculum (Linux fundamentals, Ansible, observability, platform) is [ai-homelab](https://github.com/gjstoychev/ai-homelab), paused while this repo runs.

## Nodes

- `formula` — ASUS ROG Maximus Z690 Formula, i7-14700, 64 GB DDR5, 2× RTX 5060 Ti 16 GB (both on CPU lanes, Gen5 x8, no P2P)
- `hero` — ASUS ROG Maximus Z690 Hero, i5-13400, 32 GB DDR5, two GPUs of its own (being rebuilt)

Full component list and topologies: [`docs/hardware.md`](docs/hardware.md). Kernel, driver, CUDA, NCCL and engine versions, with a changelog: [`docs/stack.md`](docs/stack.md).

## Experiments

| Experiment | Question | Status | Write-up |
|---|---|---|---|
| [`2026-11-boardswap`](experiments/2026-11-boardswap/) | What does moving the second GPU from chipset x4 to CPU x8 lanes change? Ollama vs llama.cpp on two cards | done | [Two GPUs, x8/x4 vs x8/x8: does the slot matter?](https://gshell.dev/blog/x8x4-vs-x8x8-local-llm/) |
| [`2026-11-llamacpp-flags`](experiments/2026-11-llamacpp-flags/) | How fast do 80 GB MoE models run on 32 GB of VRAM, one llama.cpp setting at a time? | done | [llama.cpp: 80 GB MoE models on 32 GB of VRAM](https://gshell.dev/blog/llamacpp-moe-beyond-vram/) |
| [`2026-11-strata`](experiments/2026-11-strata/) | How fast does Strata run Qwen3.8-Flash-Next with the whole model (no route tail skip), against llama.cpp on the same GGUF, and what is the best way to use both GPUs? | done | |
| [`2026-11-basalt`](experiments/2026-11-basalt/) | Is Basalt, Strata's Blackwell fork, faster on the same weights, and which two-card mode suits two 16 GB cards? | planned | |

Each experiment directory has a `README.md` (question, setup with the full stack, results, findings), a `journal.md` (every step: why, the exact command, expected vs actual) and the scripts and raw output behind every number.

## Layout

| Path | Content |
|---|---|
| `experiments/<yyyy-mm>-<name>/` | One experiment each |
| `docs/` | Hardware and software stack |
| `scripts/` | Scripts shared by experiments, such as `build-llamacpp.sh` |
| `work/` | Disposable scratch, not in git |
