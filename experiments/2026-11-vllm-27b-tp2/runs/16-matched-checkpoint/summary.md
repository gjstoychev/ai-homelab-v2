# Step 16: matched checkpoint (nvidia/Qwen3.8-27B-NVFP4 482ca0f3), vLLM 0.30.0 TP=2 and Basalt split

Generated with `scripts/baseline-summary.py` (vllm/, basalt/) plus the needle files. Journal step 16.

## vLLM

| label | kind | n | thinking | tokens (reasoning) | TTFT s | decode t/s | range | acc. len | per position |
|---|---|---|---|---|---|---|---|---|---|
| matched-nomtp | prose | 5 | off | 256 (0) | 1.577 ± 0.003 | 32.06 ± 0.01 | 32.0–32.1 | – | – |
| matched-nomtp | structured | 5 | off | 256 (0) | 1.581 ± 0.001 | 32.05 ± 0.00 | 32.0–32.0 | – | – |
| matched-nomtp-repro | prose | 3 | on | 800 (800) | 1.584 ± 0.000 | 32.01 ± 0.01 | 32.0–32.0 | – | – |
| matched-nomtp-repro | structured | 3 | on | 800 (800) | 1.609 ± 0.001 | 32.01 ± 0.00 | 32.0–32.0 | – | – |
| matched-nomtp-repro | mixed | 3 | on | 800 (800) | 1.610 ± 0.001 | 32.01 ± 0.00 | 32.0–32.0 | – | – |
| matched-mtp4 | prose | 5 | off | 256 (0) | 1.644 ± 0.017 | 53.64 ± 2.53 | 50.6–57.5 | 2.31 | 0.68 / 0.37 / 0.19 / 0.07 |
| matched-mtp4 | structured | 5 | off | 256 (0) | 1.649 ± 0.019 | 96.79 ± 4.84 | 91.0–101.9 | 4.21 | 0.96 / 0.82 / 0.75 / 0.68 |
| matched-mtp4-repro | prose | 3 | on | 800 (800) | 1.649 ± 0.019 | 55.65 ± 1.10 | 54.4–56.6 | 2.40 | 0.66 / 0.38 / 0.22 / 0.14 |
| matched-mtp4-repro | structured | 3 | on | 800 (800) | 1.674 ± 0.001 | 69.89 ± 3.31 | 66.5–73.1 | 3.01 | 0.78 / 0.56 / 0.40 / 0.28 |
| matched-mtp4-repro | mixed | 3 | on | 800 (800) | 1.663 ± 0.020 | 65.00 ± 4.02 | 61.2–69.2 | 2.80 | 0.72 / 0.49 / 0.34 / 0.25 |

| label | prompt tokens | n | TTFT s | prompt t/s (tokens / TTFT) | needle |
|---|---|---|---|---|---|
| matched-nomtp | 578 | 3 | 0.225 ± 0.015 | 2582 ± 178 | 3/3 |
| matched-nomtp | 4210 | 3 | 1.557 ± 0.012 | 2703 ± 22 | 3/3 |
| matched-nomtp | 32267 | 3 | 14.263 ± 0.008 | 2262 ± 1 | 3/3 |
| matched-mtp4 | 577 | 3 | 0.237 ± 0.024 | 2453 ± 263 | 3/3 |
| matched-mtp4 | 4211 | 3 | 1.597 ± 0.016 | 2637 ± 27 | 3/3 |
| matched-mtp4 | 32267 | 3 | 14.952 ± 0.006 | 2158 ± 1 | 3/3 |

## Basalt

| label | kind | n | thinking | tokens (reasoning) | TTFT s | decode t/s | range | acc. len | per position |
|---|---|---|---|---|---|---|---|---|---|
| matched-split | prose | 5 | off | 256 | 1.156 ± 0.002 | 41.76 ± 2.61 | 39.5–45.8 | – | – |
| matched-split | structured | 5 | off | 256 | 1.161 ± 0.001 | 111.13 ± 8.35 | 98.2–119.3 | – | – |
| matched-split-repro | prose | 3 | on | 800 | 1.169 ± 0.001 | 47.74 ± 2.33 | 45.6–50.2 | – | – |
| matched-split-repro | structured | 3 | on | 800 | 1.169 ± 0.001 | 72.71 ± 6.76 | 67.3–80.3 | – | – |
| matched-split-repro | mixed | 3 | on | 800 | 1.169 ± 0.001 | 51.00 ± 2.35 | 48.6–53.3 | – | – |
| matched-split-nodraft | prose | 5 | off | 256 | 1.112 ± 0.000 | 21.93 ± 0.00 | 21.9–21.9 | – | – |
| matched-split-nodraft | structured | 5 | off | 256 | 1.112 ± 0.000 | 21.93 ± 0.00 | 21.9–21.9 | – | – |
| matched-split-nodraft-repro | prose | 3 | on | 800 | 1.120 ± 0.000 | 21.32 ± 0.00 | 21.3–21.3 | – | – |
| matched-split-nodraft-repro | structured | 3 | on | 800 | 1.122 ± 0.000 | 21.32 ± 0.00 | 21.3–21.3 | – | – |
| matched-split-nodraft-repro | mixed | 3 | on | 800 | 1.121 ± 0.000 | 21.32 ± 0.01 | 21.3–21.3 | – | – |

| label | prompt tokens | n | TTFT s | prompt t/s (tokens / TTFT) | needle |
|---|---|---|---|---|---|
| matched-split | 579 | 3 | 0.212 ± 0.001 | 2737 ± 7 | 3/3 |
| matched-split | 4212 | 3 | 1.145 ± 0.001 | 3679 ± 3 | 3/3 |
| matched-split | 32269 | 3 | 6.126 ± 0.007 | 5268 ± 6 | 3/3 |
| matched-split-nodraft | 579 | 3 | 0.204 ± 0.000 | 2837 ± 3 | 3/3 |
| matched-split-nodraft | 4212 | 3 | 1.101 ± 0.000 | 3825 ± 1 | 3/3 |
| matched-split-nodraft | 32267 | 3 | 5.897 ± 0.004 | 5471 ± 4 | 3/3 |

## 252K needle

Decode after the needle is over an 8-token answer: indicative only.

| engine, config | context | prompt tokens | n | TTFT s | prompt t/s | decode t/s after | found |
|---|---|---|---|---|---|---|---|
| vLLM matched-nomtp | 262,144 | 252875–252877 | 3 | 272.34 ± 0.01 | 929 | 36.9 | 3/3 |
| Basalt split, no drafter | 254,000 | 252886–252889 | 3 | 109.99 ± 0.14 | 2299 | 16.8 | 3/3 |
