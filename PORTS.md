# New model ports for SGLang-JAX (sgl_jax) on TPU

Four model families that upstream `sgl-project/sglang-jax` (main @ `b3e3e16`) did not support,
added in the fork `kitrakrev/sglang-jax` (one branch per port, each based on upstream main;
`ports-validate` merges them for testing). Not yet proposed upstream.

| Port | Branch | Architectures | Validated checkpoints | Status |
|---|---|---|---|---|
| Gemma 3 (text) | `add-gemma3` | `Gemma3ForCausalLM`; text-only `Gemma3ForConditionalGeneration` weights | gemma-3-1b-it, gemma-3-4b-it | working |
| GLM-4 dense | `add-glm4-dense` | `Glm4ForCausalLM` | GLM-4-9B-0414 | working |
| InternVL 3 (VLM) | `add-internvl3` | `InternVLForConditionalGeneration` (Qwen2 LM) | InternVL3-2B-hf | working |
| Qwen2-Audio (audio) | `add-qwen2-audio` | `Qwen2AudioForConditionalGeneration` | Qwen2-Audio-7B-Instruct | working |

Gemma 3 *vision* is not ported: Gemma 3 uses bidirectional attention among image tokens inside the
LM, which needs custom-mask plumbing through the FA backend that sgl_jax does not have yet.
InternVL 3 was chosen as the VLM instead (causal LM, so it can match HF exactly).

## What each port adds
- **Gemma 3**: q/k Gemma RMSNorm, per-layer-type RoPE (local 10k / global 1M), sliding window on
  5 of 6 layers, no soft-capping; new `rope_type="linear"` in `get_rope` (4B+ global layers).
- **GLM-4**: sandwich norms, fused `gate_up_proj` split at load, partial (0.5) interleaved
  (GPT-J) RoPE, q/k/v bias with bias-free `o_proj`.
- **InternVL 3**: InternViT (CLS + learned positions, layer scale) + 0.5x pixel shuffle +
  MLP projector on the existing Qwen2 backbone; processor adapter (crops -> patches); maps HF
  `rope_type="dynamic"` to unscaled RoPE (dynamic NTK only rescales past `max_position_embeddings`).
- **Qwen2-Audio**: Whisper-style encoder with padded-frame key masking, 2x pooling, projector;
  first mel-spectrogram encoder on the in-model multimodal path (adds a `MelInputSpec` precompile
  dummy and `audio_data` to `Engine.generate`); soundfile+scipy audio decoding.

## Evidence

### 1. Exact parity on CPU (fp32, tiny random checkpoints vs HF transformers 5.12)
Random-init checkpoints with each real model's config shape (layer types, RoPE, sliding window,
tokenizer, processor), served through the full `Engine`, greedy-decoded and compared with
`transformers.generate`. Pass = identical tokens and max |logprob diff| < 1e-4.

| Port | Requests | Greedy tokens | max \|dlogp\| | Negative controls (must fail) |
|---|---|---|---|---|
| GLM-4 | 4 text | identical | 2.9e-6 | neox-style RoPE -> MISMATCH |
| Gemma 3 (1B layout) | 4 text, one longer than the window | identical | 6.7e-6 | no sliding window -> MISMATCH |
| Gemma 3 (4B layout, linear RoPE) | 4 text | identical | 3.8e-6 | no linear scaling -> MISMATCH |
| InternVL 3 | 7-crop image, 1-crop image, 2 images, text-only | identical | 2.9e-6 | - |
| Qwen2-Audio | 3.3 s clip, 7.1 s clip, 2 clips, text-only | identical | 2.0e-5 | - |

Encoder-only checks (vision/audio tower vs HF, same weights): InternVL rel max|diff| 1e-6;
Qwen2-Audio 7e-7 at two clip lengths (exercises padded-frame masking).
Scripts: `results/ports/cpu_parity*.py`, `*_encoder_check.py`; JSON in `results/ports/<model>/cpu-tiny/`.
The same checks are committed as tests on each branch (`test/srt/models/test_*_tiny_parity.py`).

### 2. Real checkpoints on TPU v6e-1 (sgl_jax bf16) vs HF transformers (fp32, VM host CPU)
Greedy, 32 new tokens (or until EOS). bf16 vs fp32 drift makes late divergence on near-tied tokens
expected; what matters is the matched prefix and the logprob gap on it.

| Model | Request | Matched tokens | max \|dlogp\| on prefix | Note |
|---|---|---|---|---|
| gemma-3-1b-it | 4 text | 30/30 (EOS), 32/32, 32/32, 32/32 | 0.02 to 0.10 | |
| gemma-3-4b-it | 4 text | 32/32 x 4 | 0.04 to 0.17 | served text-only via `--json-model-override-args` |
| GLM-4-9B-0414 | 4 text | 32, 16, 7, 32 of 32 | 0.01 to 0.06 | divergences are rewordings ("...is prime:" vs "...is prime. A prime number..."; "due to a phenomenon called Rayleigh scattering" vs "due to Rayleigh scattering, a phenomenon") |
| InternVL3-2B-hf | 7-crop image / red square / text | 32/32, 1/6, 32/32 | 0.06, 0.004, 0.06 | square: "The square is red." vs "The color of the square is red." (near-tie) |
| Qwen2-Audio-7B | tone / chirp / text | 8/8 (EOS), 6/6 (EOS), 4/32 | 0.09, 0.15, 0.89 | both audio answers identical ("It is a telephone dial tone.", "The pitch is rising."); the text-only one rewords ("...of light are red, green, and blue" vs "...are red, blue, and green"), with a larger gap on the prefix than the other models, which I have not explained. That path is the existing upstream Qwen2 backbone, and the CPU fp32 test matched it exactly |

Per-request texts: `results/ports/<model>/parity.json`.

### 3. Throughput: agentbench multi-turn agent workload, v6e-1, radix cache on
Same harness as the main study (4 sessions/level x 6 turns, 600-word system prompt, 128 max out).
0 errors on every row.

| Model | c | out tok/s | TTFT p50 / p99 ms | ITL p50 ms | cached prompt tokens (mean) |
|---|---|---|---|---|---|
| gemma-3-1b-it | 1 | 280 | 14.8 / 18.6 | 3.3 | 1,029 |
| gemma-3-1b-it | 16 | 2,820 | 20.4 / 55.0 | 5.0 | 1,070 |
| gemma-3-4b-it | 1 | 134 | 21.3 / 23.1 | 7.2 | 1,083 |
| gemma-3-4b-it | 16 | 1,321 | 27.4 / 59.3 | 11.5 | 1,107 |
| GLM-4-9B-0414 | 1 | 61 | 38.8 / 42.4 | 15.7 | 981 |
| GLM-4-9B-0414 | 16 | 739 | 49.2 / 89.8 | 20.4 | 1,041 |

Full tables (c=1,4,8,16): `results/ports/<model>/bench/`.

### 4. Multimodal load: InternVL 3 and Qwen2-Audio, v6e-1 (`results/ports/bench_mm.py`)
Each request carries a unique image or clip (no encoder-cache hits) and streams 128 output tokens
(`ignore_eos`). TTFT includes host preprocessing, the encoder and prefill. Every concurrency level is
warmed before measuring. 0 errors on every row.

InternVL3-2B-hf, 896x672 PNG (13 crops, 3,328 image tokens per request), `--chunked-prefill-size 4096`:

| c | out tok/s | TTFT p50 / p99 ms | ITL p50 ms | E2E p50 ms |
|---|---|---|---|---|
| 1 | 177 | 258 / 275 | 3.7 | 723 |
| 4 | 415 | 717 / 783 | 4.5 | 1,229 |
| 8 | 550 | 1,053 / 1,412 | 6.7 | 1,845 |
| 16 | 654 | 1,639 / 2,573 | 12.0 | 3,102 |

Qwen2-Audio-7B-Instruct, 10 s WAV (250 audio tokens per request):

| c | out tok/s | TTFT p50 / p99 ms | ITL p50 ms | E2E p50 ms |
|---|---|---|---|---|
| 1 | 72 | 129 / 129 | 13.0 | 1,783 |
| 4 | 206 | 445 / 1,641 | 14.5 | 2,188 |
| 8 | 410 | 641 / 708 | 14.4 | 2,518 |

`*-mmbench-cold/` holds the first pass, which warmed only c=2. It shows XLA compile stalls when a
new batch shape first appears (52 s TTFT p99 at c=4 for InternVL, 28 s at c=8 for audio). Warm
every shape you plan to serve.

### 5. TPU v6e-1 (sgl_jax ports) vs GPU (upstream SGLang v0.5.2), all four ports
Same client and workloads as above: agentbench for the text models, `bench_mm.py` for image and
audio, with the same concurrencies. GPUs are Modal on-demand, one GPU each (`modal/gpu_ports.py`).
GPU device is from `nvidia-smi`. $/1M output tokens use on-demand list prices: v6e $2.70/chip-h,
A100-80GB $2.50/h, H100 $3.95/h. This compares engines as well as hardware: sgl_jax with these new
ports vs mature SGLang CUDA kernels.

#### gemma-3-1b-it (text)

| c | device | out tok/s | TTFT p50 / p99 ms | ITL p50 ms | $/1M out | errors |
|---|---|---|---|---|---|---|
| 1 | v6e-1 | 280 | 15 / 19 | 3.3 | 2.67 | 0 |
| 1 | A100 | 115 | 45 / 51 | 6.6 | 6.03 | 0 |
| 1 | H100 | 123 | 59 / 71 | 5.4 | 8.94 | 0 |
| 4 | v6e-1 | 989 | 16 / 23 | 3.7 | 0.76 | 0 |
| 4 | A100 | 310 | 88 / 123 | 7.3 | 2.24 | 0 |
| 4 | H100 | 308 | 145 / 261 | 5.9 | 3.56 | 0 |
| 8 | v6e-1 | 1,519 | 17 / 32 | 4.3 | 0.49 | 0 |
| 8 | A100 | 546 | 90 / 133 | 7.4 | 1.27 | 0 |
| 8 | H100 | 495 | 168 / 236 | 5.9 | 2.22 | 0 |
| 16 | v6e-1 | 2,820 | 20 / 55 | 5.0 | 0.27 | 0 |
| 16 | A100 | 854 | 124 / 319 | 8.0 | 0.81 | 0 |
| 16 | H100 | 928 | 166 / 622 | 6.2 | 1.18 | 0 |

#### gemma-3-4b-it (text)

| c | device | out tok/s | TTFT p50 / p99 ms | ITL p50 ms | $/1M out | errors |
|---|---|---|---|---|---|---|
| 1 | v6e-1 | 134 | 21 / 23 | 7.2 | 5.59 | 0 |
| 1 | A100 | 71 | 84 / 90 | 12.8 | 9.80 | 0 |
| 1 | H100 | 100 | 63 / 69 | 9.2 | 11.03 | 0 |
| 4 | v6e-1 | 466 | 24 / 32 | 8.0 | 1.61 | 0 |
| 4 | A100 | 214 | 95 / 172 | 16.6 | 3.25 | 0 |
| 4 | H100 | 315 | 62 / 163 | 11.7 | 3.48 | 0 |
| 8 | v6e-1 | 810 | 25 / 40 | 9.3 | 0.93 | 0 |
| 8 | A100 | 347 | 107 / 232 | 21.0 | 2.00 | 0 |
| 8 | H100 | 504 | 63 / 170 | 14.4 | 2.18 | 0 |
| 16 | v6e-1 | 1,321 | 27 / 59 | 11.5 | 0.57 | 0 |
| 16 | A100 | 532 | 143 / 222 | 27.5 | 1.30 | 0 |
| 16 | H100 | 741 | 104 / 192 | 19.9 | 1.48 | 0 |

#### GLM-4-9B-0414 (text)

| c | device | out tok/s | TTFT p50 / p99 ms | ITL p50 ms | $/1M out | errors |
|---|---|---|---|---|---|---|
| 1 | v6e-1 | 61 | 39 / 42 | 15.7 | 12.24 | 0 |
| 1 | A100 | 74 | 41 / 53 | 12.9 | 9.43 | 0 |
| 1 | H100 | 120 | 42 / 53 | 7.6 | 9.15 | 0 |
| 4 | v6e-1 | 202 | 48 / 68 | 17.0 | 3.71 | 0 |
| 4 | A100 | 240 | 51 / 74 | 14.3 | 2.90 | 0 |
| 4 | H100 | 335 | 52 / 83 | 9.8 | 3.28 | 0 |
| 8 | v6e-1 | 404 | 49 / 70 | 18.1 | 1.86 | 0 |
| 8 | A100 | 459 | 51 / 78 | 15.7 | 1.51 | 0 |
| 8 | H100 | 643 | 54 / 114 | 11.3 | 1.71 | 0 |
| 16 | v6e-1 | 739 | 49 / 90 | 20.4 | 1.02 | 0 |
| 16 | A100 | 800 | 57 / 116 | 18.4 | 0.87 | 0 |
| 16 | H100 | 1,057 | 66 / 106 | 13.7 | 1.04 | 0 |

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

#### Qwen2-Audio-7B-Instruct (10 s clip, 250 audio tokens)

| c | device | out tok/s | TTFT p50 / p99 ms | ITL p50 ms | $/1M out | errors |
|---|---|---|---|---|---|---|
| 1 | v6e-1 | 72 | 129 / 129 | 13.0 | 10.44 | 0 |
| 1 | A100 | 93 | 79 / 80 | 10.2 | 7.45 | 0 |
| 1 | H100 | 148 | 70 / 70 | 6.2 | 7.40 | 0 |
| 4 | v6e-1 | 206 | 445 / 1,641 | 14.5 | 3.64 | 0 |
| 4 | A100 | 329 | 202 / 227 | 10.6 | 2.11 | 0 |
| 4 | H100 | 500 | 170 / 180 | 6.7 | 2.19 | 0 |
| 8 | v6e-1 | 410 | 641 / 708 | 14.4 | 1.83 | 0 |
| 8 | A100 | 564 | 358 / 503 | 11.4 | 1.23 | 0 |
| 8 | H100 | 870 | 270 / 307 | 7.1 | 1.26 | 0 |

What the numbers say:
- **Gemma 3 (1B, 4B)**: v6e-1 is ahead across the board, with 2.5 to 3x the A100 throughput,
  3 to 6x lower TTFT, and the lowest $/1M (1B at c=16: $0.27 vs $0.81 A100 and $1.18 H100).
  Upstream SGLang v0.5.2 Gemma 3 on GPU looks under-optimized. The 4B on GPU runs the full
  `Gemma3ForConditionalGeneration` class on text-only prompts.
- **GLM-4-9B**: H100 is fastest (1,057 tok/s at c=16, ITL 13.7 ms). A100 is slightly ahead of
  v6e-1 on throughput. $/1M is nearly equal (c=16: v6e $1.02, A100 $0.87, H100 $1.04). v6e-1 has
  the lowest TTFT p99 at c=8.
- **InternVL3-2B**: on GPU, SGLang v0.5.2 fails CUDA-graph capture for this model, so both GPU
  runs use `--disable-cuda-graph`. That inflates GPU ITL (16 to 28 ms vs 3.6 to 12 ms on v6e-1)
  and understates GPU throughput. The "H100" run was placed on an **H200** (priced as H100 here).
  The GPU is much faster to first token (46 to 113 ms vs 258 to 1,639 ms on v6e-1). The
  sgl_jax vision encoder is unoptimized (plain `dot_product_attention` per crop), which is the
  clearest next optimization for the port.
- **Qwen2-Audio-7B**: GPUs lead (H100 870 tok/s at c=8 vs 410 on v6e-1; TTFT about half). $/1M
  is $1.83 on v6e-1 vs $1.23 to $1.26 on GPU.

## Upstream status (sgl-project/sglang-jax)
- **Gemma 3 text**: existing PR #1100 (hashkanna). No competing PR. A factual comment with TPU
  parity numbers and the single-KV-head reshape sharding bug is drafted in
  `results/ports/upstream/pr1100_comment.md`. It is **not posted**: the automated permission
  check blocked posting from this session, so it needs the user to post it or approve.
  #1100 already has linear RoPE scaling.
- **GLM-4 dense, InternVL 3, Qwen2-Audio**: no existing PRs or issues found. Branches are on the
  current upstream main (`b3e3e16`), lint-clean, and PR titles and bodies are ready in
  `results/ports/upstream/pr_*.md`. **Not opened**, waiting for confirmation.
- **Llama3RotaryEmbedding** (`_compute_inv_freq` overridden but `_inv_freq_np` used, so Llama 3 RoPE
  scaling is not applied): already fixed in open PR #1767, so no separate PR.
- **Gemma 3 vision**: design note only, on branch `add-gemma3-vision`
  (`docs/design/gemma3_vision.md`). It needs bidirectional image-token attention in the LM: a custom
  mask in FA for extend batches, combined with sliding window, plus image-aligned chunked prefill.
  No PR.

## Bugs found by the TPU run (fixed, kept as evidence)
- `gemma3-1b-crash-before-fix/`: with a single KV head, the q/k/v reshape under explicit mesh axes
  moved the tensor sharding onto head_dim and the FA `shard_map` rejected it. Fixed by pinning
  `P("data","tensor",None)` on the reshape (also applied to GLM-4).
- `internvl3-untied-bug/`: InternVL3-*-hf sets `tie_word_embeddings=True` but ships a different
  `lm_head`; HF keeps them untied. The port tied them and diverged after 2 to 4 tokens. Fixed by
  tying only when the checkpoint has no lm_head.
- `*-failed-env/`: environment issues, not port bugs. llguidance rejects tokenizers that have one
  more token than the model vocab (gemma-3-1b, InternVL3-2B), so those runs use
  `--grammar-backend none`. librosa breaks because the repo pins `msgpack-python 0.5.x`, so the
  Qwen2-Audio loader now uses soundfile.

## Reproduce
- CPU parity: `python results/ports/cpu_parity.py {glm4|gemma3|gemma3lin} <dir>`,
  `cpu_parity_vlm.py <dir>`, `cpu_parity_audio.py <dir>` (with the port branch on `PYTHONPATH`).
- TPU: create a v6e-1 VM (`scripts/tpu_up.sh`), install the `ports-validate` branch of
  `kitrakrev/sglang-jax` plus CPU torch, torchvision, soundfile, then
  `bash results/ports/vm_validate.sh [gemma3-1b gemma3-4b glm4 internvl3 qwen2-audio]`.
  Gemma uses the ungated `unsloth/gemma-3-*-it` mirrors.
- Cost: TPU spot v6e-1 for about 50 min (parity) and about 37 min (multimodal metrics), roughly
  $1.5 to $4 (spot to on-demand ceiling). Modal: about 2.5 A100-hours and 1.2 H100/H200-hours,
  roughly $10. Total about $12 to $14. All TPU VMs are deleted, and all `agentbench-gpu-ports`
  Modal apps are stopped.
