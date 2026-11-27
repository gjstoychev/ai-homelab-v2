Runs of the matched-checkpoint configs that loaded the WRONG model: MODEL_DIR was not exported, so
scripts/vllm-serve.sh mounted its default, the pinned gittensor checkpoint (qwen3.8-27b-nvfp4-0cc2795; see the
"model:" line in each .out). Not part of the matched comparison. Kept because they are still measurements of the
pinned checkpoint with MTP k=4 above util 0.95 (CUDA-graph estimate on, the vLLM default):
  0.965-oom/       util 0.965, 262144: started; first request OOM (114 MiB asked, 102.6 MiB free)
  0.96-oom/        util 0.96, 262144: started; first request OOM (58 MiB asked, 42.6 MiB free)
  256000-oom/      util 0.965, max-model-len 256000: same OOM as 0.965 (KV pool 292,800 tokens, against 292,514 at 262144)
  0.96-seqs1-oom/  util 0.96, max-num-seqs 1: same OOM (58 MiB, 40.6 MiB free)
  0.965-nomtp-ran/ matched-nomtp.args, util 0.965, no MTP: ran the whole baseline (42.4 t/s decode)
The pinned checkpoint runs MTP k=4 at util 0.95 without a problem (steps 4, 9b).
