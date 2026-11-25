| run | kind | n | load s | TTFT s | prompt t/s | decode t/s | drafts | VRAM MiB | RSS GiB | short |
|---|---|---|---|---|---|---|---|---|---|---|
| 10b-strata-2gpu-modes/helper-nodraft | prose | 5 | 15.8 | 2.47 ± 0.02 | 1757.4 ± 13.4 | 66.9 ± 0.2 | 0/0 | 15042 / 15344 | 43.4 | 0 |
| 10b-strata-2gpu-modes/helper-nodraft | structured | 5 | 15.8 | 2.51 ± 0.00 | 1741.6 ± 0.9 | 66.6 ± 0.4 | 0/0 | 15042 / 15344 | 43.4 | 0 |
| 10b-strata-2gpu-modes/helper-mtp | prose | 5 | 16.8 | 2.52 ± 0.01 | 1721.1 ± 8.0 | 93.4 ± 3.5 | 602/1054 (57 %) | 15154 / 15344 | 42.9 | 0 |
| 10b-strata-2gpu-modes/helper-mtp | structured | 5 | 16.8 | 2.54 ± 0.00 | 1715.9 ± 3.4 | 138.3 ± 1.3 | 876/1037 (84 %) | 15154 / 15344 | 42.9 | 0 |
| 10b-strata-2gpu-modes/2gpu-mtp-pipe2 | prose | 5 | 17.3 | 2.45 ± 0.00 | 1777.0 ± 0.7 | 90.6 ± 1.5 | 591/1069 (55 %) | 15002 / 14910 | 44.4 | 0 |
| 10b-strata-2gpu-modes/2gpu-mtp-pipe2 | structured | 5 | 17.3 | 2.46 ± 0.00 | 1778.9 ± 0.5 | 159.2 ± 4.1 | 866/1032 (84 %) | 15002 / 14910 | 44.4 | 0 |
| 10b-strata-2gpu-modes/2gpu-mtp-pipe2-stage0 | prose | 5 | 17.8 | 2.47 ± 0.00 | 1758.8 ± 1.3 | 77.9 ± 1.1 | 582/1062 (55 %) | 15004 / 14908 | 44.4 | 0 |
| 10b-strata-2gpu-modes/2gpu-mtp-pipe2-stage0 | structured | 5 | 17.8 | 2.48 ± 0.00 | 1757.5 ± 0.8 | 156.5 ± 5.4 | 876/1028 (85 %) | 15004 / 14908 | 44.4 | 0 |
