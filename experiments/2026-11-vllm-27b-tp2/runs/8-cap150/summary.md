| configuration | requests at once | kind | full groups | total t/s | short groups | TTFT s mean (max) | decode t/s per stream | KV pool |
|---|--:|---|--:|--:|--:|---|--:|--:|
| tp2-seqs2 | 1 | prose | 3 | 45.5 | 0 | 1.43 (1.44) | 60.8 | 463,166 |
| tp2-seqs2 | 1 | structured | 3 | 66.7 | 0 | 1.45 (1.45) | 106.6 | 463,166 |
| tp2-seqs2 | 2 | prose | 3 | 70.0 | 0 | 2.21 (2.87) | 51.6 | 463,166 |
| tp2-seqs2 | 2 | structured | 3 | 95.6 | 0 | 2.31 (2.88) | 88.2 | 463,166 |
| tp2-seqs2 | 4 | prose | 3 | 69.3 | 0 | 5.83 (10.42) | 47.7 | 463,166 |
| tp2-seqs2 | 4 | structured | 3 | 96.8 | 0 | 4.86 (8.15) | 86.0 | 463,166 |

| configuration | prompt tokens | TTFT s | needle |
|---|--:|--:|--:|
| tp2-seqs2 | 32268 | 13.32 | 2/2 |
