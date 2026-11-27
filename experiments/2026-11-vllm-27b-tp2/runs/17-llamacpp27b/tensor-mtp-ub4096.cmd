date: Fri Nov 27 09:26:55 AM UTC 2026
label: tensor-mtp-ub4096, reps: 5
kernel: 7.0.0-44-generic
driver: 595.91.07
gpu: 0, NVIDIA GeForce RTX 5060 Ti, 00000000:01:00.0, 180.00 W
gpu: 1, NVIDIA GeForce RTX 5060 Ti, 00000000:02:00.0, 180.00 W
page cache dropped: no (BENCH_SUDO not set)
cpu power: not measured (BENCH_SUDO not set)
command: /home/llama/llama.cpp/v1.0.0/build/bin/llama-server -m /usr/share/ollama/.ollama/models/blobs/sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d --host 127.0.0.1 --port 8080 -np 1 --cache-ram 0 -t 8 -c 40960 -lv 4 -ngl 99 -fa on -sm tensor -b 4096 -ub 4096 --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-backend-sampling
