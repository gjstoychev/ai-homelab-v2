| configuration | requests at once | kind | full groups | total t/s | short groups | TTFT s mean (max) | decode t/s per stream | KV pool |
|---|--:|---|--:|--:|--:|---|--:|--:|
| split-nodraft-par4 | 1 | prose | 3 | 22.9 | 0 | 0.95 (0.95) | 24.9 | – |
| split-nodraft-par4 | 1 | structured | 3 | 22.9 | 0 | 0.96 (0.96) | 24.9 | – |
| split-nodraft-par4 | 2 | prose | 3 | 39.5 | 0 | 1.62 (2.94) | 22.6 | – |
| split-nodraft-par4 | 2 | structured | 3 | 38.4 | 0 | 1.80 (2.95) | 22.4 | – |
| split-nodraft-par4 | 4 | prose | 3 | 64.2 | 0 | 2.72 (4.96) | 19.6 | – |
| split-nodraft-par4 | 4 | structured | 3 | 64.1 | 0 | 2.74 (5.00) | 19.6 | – |
| split-par4-163k | 1 | prose | 3 | 39.6 | 0 | 1.00 (1.00) | 46.6 | – |
| split-par4-163k | 1 | structured | 3 | 89.6 | 0 | 1.01 (1.01) | 137.9 | – |
| split-par4-163k | 2 | prose | 3 | 57.5 | 0 | 1.87 (3.11) | 38.3 | – |
| split-par4-163k | 2 | structured | 3 | 90.3 | 0 | 2.06 (3.13) | 79.3 | – |
| split-par4-163k | 4 | prose | 3 | 84.1 | 0 | 2.87 (5.28) | 28.7 | – |
| split-par4-163k | 4 | structured | 3 | 103.3 | 0 | 3.41 (5.32) | 42.4 | – |

| configuration | prompt tokens | TTFT s | needle |
|---|--:|--:|--:|
| split-nodraft-par4 | 32266 | 5.20 | 2/2 |
| split-par4-163k | 32266 | 5.62 | 2/2 |
