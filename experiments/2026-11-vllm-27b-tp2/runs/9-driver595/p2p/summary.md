| label | kind | n | thinking | tokens (reasoning) | TTFT s | decode t/s | range | acc. len | per position |
|---|---|---|---|---|---|---|---|---|---|
| fp8-nomtp-p2pon | prose | 3 | off | 256 (0) | 1.324 ± 0.001 | 42.35 ± 0.01 | 42.3–42.4 | – | – |
| fp8-nomtp-p2pon | structured | 3 | off | 256 (0) | 1.329 ± 0.001 | 42.32 ± 0.01 | 42.3–42.3 | – | – |
| fp8-nomtp-p2pon-repro | prose | 3 | on | 800 (800) | 1.334 ± 0.000 | 42.26 ± 0.00 | 42.3–42.3 | – | – |
| fp8-nomtp-p2pon-repro | structured | 3 | on | 800 (800) | 1.350 ± 0.002 | 42.25 ± 0.01 | 42.2–42.3 | – | – |
| fp8-nomtp-p2pon-repro | mixed | 3 | on | 800 (800) | 1.353 ± 0.001 | 42.25 ± 0.00 | 42.2–42.2 | – | – |
| fp8-nomtp-p2psys | prose | 3 | off | 256 (0) | 1.325 ± 0.002 | 42.37 ± 0.01 | 42.4–42.4 | – | – |
| fp8-nomtp-p2psys | structured | 3 | off | 256 (0) | 1.330 ± 0.001 | 42.34 ± 0.01 | 42.3–42.4 | – | – |
| fp8-nomtp-p2psys-repro | prose | 3 | on | 800 (800) | 1.335 ± 0.001 | 42.27 ± 0.01 | 42.3–42.3 | – | – |
| fp8-nomtp-p2psys-repro | structured | 3 | on | 800 (800) | 1.351 ± 0.001 | 42.26 ± 0.01 | 42.3–42.3 | – | – |
| fp8-nomtp-p2psys-repro | mixed | 3 | on | 800 (800) | 1.352 ± 0.000 | 42.27 ± 0.01 | 42.3–42.3 | – | – |

| label | prompt tokens | n | TTFT s | prompt t/s (tokens / TTFT) | needle |
|---|---|---|---|---|---|
| fp8-nomtp-p2pon | 578 | 3 | 0.185 ± 0.010 | 3131 ± 187 | 3/3 |
| fp8-nomtp-p2pon | 4212 | 3 | 1.308 ± 0.008 | 3219 ± 20 | 3/3 |
| fp8-nomtp-p2pon | 32268 | 3 | 12.413 ± 0.003 | 2600 ± 1 | 3/3 |
| fp8-nomtp-p2psys | 578 | 3 | 0.185 ± 0.011 | 3140 ± 201 | 3/3 |
| fp8-nomtp-p2psys | 4210 | 3 | 1.306 ± 0.007 | 3224 ± 18 | 3/3 |
| fp8-nomtp-p2psys | 32267 | 3 | 12.418 ± 0.006 | 2598 ± 1 | 3/3 |
