date: Wed Nov 25 08:38:56 PM UTC 2026
label: strata-2gpu-nodraft-128k-cache, reps: 3
kernel: 7.0.0-43-generic
driver: 580.178.04
gpu: 0, NVIDIA GeForce RTX 5060 Ti, 00000000:01:00.0, 180.00 W
gpu: 1, NVIDIA GeForce RTX 5060 Ti, 00000000:02:00.0, 180.00 W
page cache dropped: no (BENCH_SUDO not set)
cpu power: not measured (BENCH_SUDO not set)
command: env -C /home/llama/strata/v0.2.21/src/terminal .venv/bin/python serve/server.py --engine strata --config /home/llama/ai-homelab-v2/experiments/2026-11-strata/configs/strata-2gpu-nodraft-128k-cache.json --port 8080
