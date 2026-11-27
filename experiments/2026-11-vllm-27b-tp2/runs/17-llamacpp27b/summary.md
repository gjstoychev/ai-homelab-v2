| label | kind | n | thinking | tokens (reasoning) | TTFT s | decode t/s | range | acc. len | per position |
|---|---|---|---|---|---|---|---|---|---|
| tensor-mtp | prose | 5 | off | 256 | 3.271 ± 0.005 | 55.47 ± 1.16 | 54.3–57.3 | – | – |
| tensor-mtp | structured | 5 | off | 256 | 3.281 ± 0.001 | 100.84 ± 1.03 | 99.5–101.7 | – | – |
| tensor-mtp-repro | prose | 3 | on | 800 | 3.404 ± 0.002 | 60.01 ± 1.72 | 58.4–61.8 | – | – |
| tensor-mtp-repro | structured | 3 | on | 800 | 3.413 ± 0.002 | 80.95 ± 11.61 | 74.0–94.3 | – | – |
| tensor-mtp-repro | mixed | 3 | on | 800 | 3.416 ± 0.002 | 70.30 ± 3.57 | 66.3–73.2 | – | – |
| tensor-nomtp | prose | 5 | off | 256 | 3.114 ± 0.004 | 40.30 ± 0.02 | 40.3–40.3 | – | – |
| tensor-nomtp | structured | 5 | off | 256 | 3.122 ± 0.002 | 40.40 ± 0.09 | 40.3–40.5 | – | – |
| tensor-nomtp-repro | prose | 3 | on | 800 | 3.241 ± 0.005 | 40.37 ± 0.03 | 40.3–40.4 | – | – |
| tensor-nomtp-repro | structured | 3 | on | 800 | 3.248 ± 0.002 | 40.39 ± 0.03 | 40.4–40.4 | – | – |
| tensor-nomtp-repro | mixed | 3 | on | 800 | 3.251 ± 0.002 | 40.37 ± 0.04 | 40.3–40.4 | – | – |
| tensor-mtp-ub4096 | prose | 5 | off | 256 | 3.423 ± 0.001 | 57.94 ± 2.62 | 56.2–62.5 | – | – |
| tensor-mtp-ub4096 | structured | 5 | off | 256 | 3.428 ± 0.002 | 99.98 ± 0.51 | 99.1–100.4 | – | – |
| tensor-mtp-ub4096-repro | prose | 3 | on | 800 | 3.544 ± 0.002 | 61.81 ± 2.36 | 60.2–64.5 | – | – |
| tensor-mtp-ub4096-repro | structured | 3 | on | 800 | 3.549 ± 0.002 | 74.07 ± 3.10 | 71.5–77.5 | – | – |
| tensor-mtp-ub4096-repro | mixed | 3 | on | 800 | 3.572 ± 0.002 | 67.36 ± 1.78 | 65.5–69.1 | – | – |
| layer-mtp | prose | 5 | off | 256 | 3.128 ± 0.002 | 37.44 ± 1.41 | 36.0–39.3 | – | – |
| layer-mtp | structured | 5 | off | 256 | 3.137 ± 0.004 | 66.88 ± 0.57 | 66.3–67.6 | – | – |
| layer-mtp-repro | prose | 3 | on | 800 | 3.242 ± 0.010 | 40.22 ± 0.89 | 39.3–41.1 | – | – |
| layer-mtp-repro | structured | 3 | on | 800 | 3.243 ± 0.003 | 49.49 ± 1.59 | 47.7–50.7 | – | – |
| layer-mtp-repro | mixed | 3 | on | 800 | 3.243 ± 0.003 | 51.39 ± 2.28 | 50.0–54.0 | – | – |

| label | prompt tokens | n | TTFT s | prompt t/s (tokens / TTFT) | needle |
|---|---|---|---|---|---|
| tensor-mtp | 579 | 3 | 0.598 ± 0.002 | 968 ± 5 | 3/3 |
| tensor-mtp | 4211 | 3 | 3.191 ± 0.005 | 1320 ± 2 | 3/3 |
| tensor-mtp | 32266 | 3 | 25.560 ± 0.007 | 1262 ± 0 | 3/3 |
| tensor-nomtp | 578 | 3 | 0.580 ± 0.004 | 997 ± 8 | 3/3 |
| tensor-nomtp | 4211 | 3 | 3.036 ± 0.003 | 1387 ± 1 | 3/3 |
| tensor-nomtp | 32268 | 3 | 24.150 ± 0.005 | 1336 ± 0 | 3/3 |
| tensor-mtp-ub4096 | 580 | 3 | 0.528 ± 0.001 | 1097 ± 1 | 3/3 |
| tensor-mtp-ub4096 | 4211 | 3 | 3.339 ± 0.005 | 1261 ± 2 | 3/3 |
| tensor-mtp-ub4096 | 32268 | 3 | 26.834 ± 0.017 | 1203 ± 1 | 3/3 |
| layer-mtp | 578 | 3 | 0.695 ± 0.003 | 831 ± 6 | 3/3 |
| layer-mtp | 4212 | 3 | 3.062 ± 0.022 | 1375 ± 10 | 3/3 |
| layer-mtp | 32267 | 3 | 24.950 ± 0.038 | 1293 ± 2 | 3/3 |
