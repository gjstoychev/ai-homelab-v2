| run | kind | streams | n | load s | TTFT s | prompt t/s | decode t/s | total t/s | drafts | VRAM MiB | RSS GiB | short |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11c-concurrency-short/1gpu-nodraft-par4-c1 | short | 1 | 3 | 14.8 | 1.10 ± 0.02 | 58.4 ± 1.1 | 70.4 ± 0.6 | – | 0/0 | – | – | 0 |
| 11c-concurrency-short/1gpu-nodraft-par4-c2 | short | 2 | 6 | 14.8 | 0.76 ± 0.48 | 192.2 ± 66.9 | 51.7 ± 0.9 | 96.2 ± 4.1 | 0/0 | – | – | 0 |
| 11c-concurrency-short/1gpu-nodraft-par4-c4 | short | 4 | 12 | 14.8 | 2.22 ± 0.83 | 180.5 ± 86.9 | 36.3 ± 1.7 | 125.2 ± 5.7 | 0/0 | – | – | 0 |
| 11c-concurrency-short/peer-nodraft-par4-c1 | short | 1 | 3 | 18.9 | 0.80 ± 0.00 | 80.0 ± 0.3 | 77.7 ± 0.2 | – | 0/0 | – | – | 0 |
| 11c-concurrency-short/peer-nodraft-par4-c2 | short | 2 | 6 | 18.9 | 0.65 ± 0.67 | 319.2 ± 137.6 | 57.4 ± 2.4 | 107.4 ± 9.4 | 0/0 | – | – | 0 |
| 11c-concurrency-short/peer-nodraft-par4-c4 | short | 4 | 12 | 18.9 | 1.57 ± 0.59 | 302.4 ± 151.4 | 41.2 ± 1.4 | 145.8 ± 4.6 | 0/0 | – | – | 0 |
| 11c-concurrency-short/split-nodraft-par4-c1 | short | 1 | 3 | 16.4 | 0.88 ± 0.00 | 73.2 ± 0.3 | 70.8 ± 1.4 | – | 0/0 | – | – | 0 |
| 11c-concurrency-short/split-nodraft-par4-c2 | short | 2 | 6 | 16.4 | 1.49 ± 0.75 | 66.2 ± 14.6 | 52.7 ± 3.8 | 91.0 ± 3.9 | 0/0 | – | – | 0 |
| 11c-concurrency-short/split-nodraft-par4-c4 | short | 4 | 12 | 16.4 | 2.75 ± 1.40 | 65.3 ± 13.5 | 36.4 ± 3.2 | 120.6 ± 3.1 | 0/0 | – | – | 0 |
| 11c-concurrency-short/peer-mtp-par4-c1 | short | 1 | 3 | 19.9 | 0.24 ± 0.00 | 291.5 ± 5.2 | 121.2 ± 2.9 | – | 805/1201 (67 %) | – | – | 0 |
| 11c-concurrency-short/peer-mtp-par4-c2 | short | 2 | 6 | 19.9 | 0.37 ± 0.18 | 318.6 ± 83.4 | 77.0 ± 2.4 | 144.2 ± 3.7 | 1522/2361 (64 %) | – | – | 0 |
| 11c-concurrency-short/peer-mtp-par4-c4 | short | 4 | 12 | 19.9 | 0.78 ± 0.36 | 316.4 ± 102.2 | 47.1 ± 1.3 | 173.4 ± 2.2 | 2586/3544 (73 %) | – | – | 0 |
| 11c-concurrency-short/split-mtp-par4-c1 | short | 1 | 3 | 16.8 | 0.24 ± 0.00 | 303.1 ± 5.6 | 115.7 ± 4.2 | – | 806/1193 (68 %) | – | – | 0 |
| 11c-concurrency-short/split-mtp-par4-c2 | short | 2 | 6 | 16.8 | 0.39 ± 0.18 | 287.6 ± 23.8 | 73.6 ± 4.3 | 138.4 ± 8.6 | 1573/2339 (67 %) | – | – | 0 |
| 11c-concurrency-short/split-mtp-par4-c4 | short | 4 | 12 | 16.8 | 0.87 ± 0.48 | 237.2 ± 48.3 | 45.1 ± 1.3 | 164.7 ± 3.8 | 2649/3513 (75 %) | – | – | 0 |
| 11c-concurrency-short/split-nodraft-par8-c8 | short | 8 | 24 | 16.4 | 4.91 ± 2.51 | 66.1 ± 9.5 | 22.9 ± 2.3 | 147.4 ± 2.7 | 0/0 | – | – | 0 |
| 11c-concurrency-short/peer-nodraft-par8-res1500-c8 | short | 8 | 24 | 18.9 | 2.10 ± 0.74 | 339.7 ± 111.7 | 26.4 ± 0.7 | 189.8 ± 3.8 | 0/0 | – | – | 0 |
| 11c-concurrency-short/peer-mtp-par8-res1500-c8 | short | 8 | 24 | 19.9 | 1.25 ± 0.60 | 344.9 ± 68.0 | 26.9 ± 0.6 | 200.5 ± 1.7 | 141/183 (77 %) | – | – | 0 |
