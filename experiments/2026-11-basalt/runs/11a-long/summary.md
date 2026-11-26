| run | prompt tokens | TTFT s (mean) | prompt t/s (mean) | needle |
|---|---|---|---|---|
| 1gpu-nodraft-128k | 32804 | 13.5 | 2430 | 3/3 |
| 1gpu-nodraft-128k | 126586 | 54.6 | 2320 | 3/3 |
| 1gpu-nodraft-128k-cache | job | 31.0, 31.04, 30.74 | turn 1 TTFT 13.726, 13.613, 13.618 | 3/3 |
| peer-nodraft-128k | 32804 | 10.5 | 3131 | 3/3 |
| peer-nodraft-128k | 126586 | 34.7 | 3649 | 3/3 |
| peer-nodraft-128k-cache | job | 25.37, 25.17, 25.79 | turn 1 TTFT 10.719, 10.561, 10.579 | 3/3 |
| split-nodraft-128k | 32804 | 10.6 | 3107 | 3/3 |
| split-nodraft-128k | 126586 | 35.8 | 3544 | 3/3 |
| split-nodraft-128k-cache | job | 27.67, 28.67, 26.99 | turn 1 TTFT 10.768, 10.609, 10.611 | 3/3 |
