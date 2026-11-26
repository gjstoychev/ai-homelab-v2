| configuration | requests at once | kind | full groups | total t/s | short groups | TTFT s mean (max) | decode t/s per stream | KV pool |
|---|--:|---|--:|--:|--:|---|--:|--:|
| pp2p41-len64k-seqs4 | 1 | prose | 3 | 31.3 | 0 | 1.08 (1.08) | 36.0 | 107,817 |
| pp2p41-len64k-seqs4 | 1 | structured | 3 | 52.2 | 0 | 1.08 (1.08) | 66.8 | 107,817 |
| pp2p41-len64k-seqs4 | 2 | prose | 3 | 53.2 | 0 | 1.58 (2.11) | 33.0 | 107,817 |
| pp2p41-len64k-seqs4 | 2 | structured | 3 | 81.2 | 0 | 1.59 (2.12) | 56.8 | 107,817 |
| pp2p41-len64k-seqs4 | 4 | prose | 3 | 86.0 | 0 | 2.59 (3.89) | 28.5 | 107,817 |
| pp2p41-len64k-seqs4 | 4 | structured | 3 | 128.9 | 0 | 2.50 (3.68) | 50.1 | 107,817 |
| tp2-seqs1 | 1 | prose | 3 | 44.1 | 0 | 1.42 (1.43) | 58.2 | 469,957 |
| tp2-seqs1 | 1 | structured | 3 | 67.5 | 0 | 1.43 (1.43) | 108.1 | 469,957 |
| tp2-seqs1 | 2 | prose | 3 | 43.9 | 0 | 4.35 (7.48) | 57.8 | 469,957 |
| tp2-seqs1 | 2 | structured | 2 | 68.7 | 1 | 3.32 (5.29) | 109.4 | 469,957 |
| tp2-seqs1 | 4 | prose | 3 | 44.5 | 0 | 10.04 (19.12) | 58.8 | 469,957 |
| tp2-seqs1 | 4 | structured | 2 | 68.9 | 1 | 6.99 (12.74) | 110.6 | 469,957 |
| tp2-seqs2 | 1 | prose | 3 | 44.2 | 0 | 1.40 (1.42) | 58.1 | 463,166 |
| tp2-seqs2 | 1 | structured | 3 | 68.5 | 0 | 1.43 (1.43) | 110.7 | 463,166 |
| tp2-seqs2 | 2 | prose | 3 | 68.6 | 0 | 2.36 (2.83) | 52.0 | 463,166 |
| tp2-seqs2 | 2 | structured | 2 | 96.9 | 1 | 2.28 (2.84) | 86.2 | 463,166 |
| tp2-seqs2 | 4 | prose | 3 | 70.0 | 0 | 5.88 (10.25) | 49.4 | 463,166 |
| tp2-seqs2 | 4 | structured | 2 | 97.1 | 1 | 4.58 (8.02) | 87.5 | 463,166 |
| tp2-seqs4 | 1 | prose | 3 | 45.6 | 0 | 1.41 (1.42) | 60.7 | 446,867 |
| tp2-seqs4 | 1 | structured | 3 | 67.6 | 0 | 1.43 (1.43) | 108.1 | 446,867 |
| tp2-seqs4 | 2 | prose | 3 | 69.4 | 0 | 2.28 (2.83) | 51.2 | 446,867 |
| tp2-seqs4 | 2 | structured | 2 | 97.2 | 1 | 2.37 (2.84) | 88.3 | 446,867 |
| tp2-seqs4 | 4 | prose | 3 | 94.0 | 0 | 3.73 (5.66) | 38.0 | 446,867 |
| tp2-seqs4 | 4 | structured | 1 | 120.2 | 2 | 3.74 (5.68) | 62.5 | 446,867 |

| configuration | prompt tokens | TTFT s | needle |
|---|--:|--:|--:|
| tp2-seqs2 | 32268 | 13.14 | 2/2 |
| tp2-seqs1 | 32269 | 13.16 | 2/2 |
| tp2-seqs4 | 32268 | 13.16 | 2/2 |
| pp2p41-len64k-seqs4 | 32267 | 9.29 | 2/2 |
