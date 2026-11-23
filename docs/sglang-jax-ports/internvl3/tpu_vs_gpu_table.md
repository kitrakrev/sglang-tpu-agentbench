#### InternVL3-2B-hf (896x672 image, 3,328 image tokens)

| c | device | out tok/s | TTFT p50 / p99 ms | ITL p50 ms | $/1M out | errors |
|---|---|---|---|---|---|---|
| 1 | v6e-1 | 177 | 258 / 275 | 3.6 | 4.24 | 0 |
| 1 | A100 | 36 | 73 / 131 | 26.6 | 19.10 | 0 |
| 1 | H100 | 63 | 46 / 48 | 15.6 | 17.46 | 0 |
| 4 | v6e-1 | 415 | 717 / 782 | 4.5 | 1.81 | 0 |
| 4 | A100 | 150 | 85 / 135 | 26.1 | 4.62 | 0 |
| 4 | H100 | 246 | 63 / 98 | 15.8 | 4.46 | 0 |
| 8 | v6e-1 | 550 | 1,053 / 1,412 | 6.7 | 1.36 | 0 |
| 8 | A100 | 301 | 98 / 178 | 25.8 | 2.31 | 0 |
| 8 | H100 | 480 | 66 / 120 | 16.2 | 2.29 | 0 |
| 16 | v6e-1 | 654 | 1,639 / 2,573 | 11.9 | 1.15 | 0 |
| 16 | A100 | 555 | 113 / 234 | 28.0 | 1.25 | 0 |
| 16 | H100 | 947 | 74 / 175 | 16.1 | 1.16 | 0 |

Notes: A100: server failed with CUDA graphs; this run uses --disable-cuda-graph; H100: ran on NVIDIA H200 (Modal upgraded the H100 request; priced as H100). server failed with CUDA graphs; this run uses --disable-cuda-graph

Source: results/ports/tpu_vs_gpu.md (built by compare_tpu_gpu.py)
