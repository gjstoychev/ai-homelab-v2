date: Wed Nov 25 09:20:42 PM UTC 2026
label: llamacpp-2gpu-best-np8, reps: 3
kernel: 7.0.0-43-generic
driver: 580.178.04
gpu: 0, NVIDIA GeForce RTX 5060 Ti, 00000000:01:00.0, 180.00 W
gpu: 1, NVIDIA GeForce RTX 5060 Ti, 00000000:02:00.0, 180.00 W
page cache dropped: no (BENCH_SUDO not set)
cpu power: not measured (BENCH_SUDO not set)
command: /home/llama/llama.cpp/v1.0.0/build/bin/llama-server -m /home/llama/models/qwen3.8-flash-next-gsq-rco-iq3xxs/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --host 127.0.0.1 --port 8080 -np 8 --cache-ram 0 -t 8 -c 65536 -lv 4 -cmoe --moe-cache-mib 20480 --moe-cache-cpu-share 0 -lm dio -b 4096 -ub 4096
