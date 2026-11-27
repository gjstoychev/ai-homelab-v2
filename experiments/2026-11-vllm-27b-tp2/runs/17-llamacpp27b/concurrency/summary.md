| configuration | requests at once | kind | full groups | total t/s | short groups | TTFT s mean (max) | decode t/s per stream | KV pool |
|---|--:|---|--:|--:|--:|---|--:|--:|
| tensor-mtp-np4 | 1 | prose | 3 | 33.0 | 0 | 3.28 (3.29) | 57.0 | – |
| tensor-mtp-np4 | 1 | structured | 3 | 44.1 | 0 | 3.32 (3.32) | 102.6 | – |
| tensor-mtp-np4 | 2 | prose | 3 | 42.8 | 0 | 6.37 (6.68) | 46.5 | – |
| tensor-mtp-np4 | 2 | structured | 3 | 52.8 | 0 | 6.42 (6.79) | 80.1 | – |
| tensor-mtp-np4 | 4 | prose | 3 | 49.0 | 0 | 10.62 (13.68) | 27.5 | – |
| tensor-mtp-np4 | 4 | structured | 3 | 57.6 | 0 | 10.90 (13.73) | 44.6 | – |
| tensor-nomtp-np4 | 1 | prose | 3 | 27.1 | 0 | 3.11 (3.11) | 40.2 | – |
| tensor-nomtp-np4 | 1 | structured | 3 | 27.0 | 0 | 3.15 (3.15) | 40.4 | – |
| tensor-nomtp-np4 | 2 | prose | 3 | 38.7 | 0 | 6.04 (6.32) | 35.6 | – |
| tensor-nomtp-np4 | 2 | structured | 3 | 38.5 | 0 | 6.07 (6.40) | 35.5 | – |
| tensor-nomtp-np4 | 4 | prose | 3 | 49.4 | 0 | 9.89 (12.78) | 25.3 | – |
| tensor-nomtp-np4 | 4 | structured | 3 | 49.3 | 0 | 10.26 (12.83) | 26.0 | – |

| configuration | prompt tokens | TTFT s | needle |
|---|--:|--:|--:|
| tensor-mtp-np4 | 32267 | 25.63 | 2/2 |
| tensor-nomtp-np4 | 32269 | 24.19 | 2/2 |
