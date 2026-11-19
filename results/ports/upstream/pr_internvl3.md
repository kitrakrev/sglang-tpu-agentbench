Title: feat(models): add InternVL 3 (InternVLForConditionalGeneration) with in-model vision

## Motivation

Adds HF-format InternVL 3 (`InternVLForConditionalGeneration`, e.g. `OpenGVLab/InternVL3-2B-hf`, 1B/8B use the same code path) on the in-model multimodal contract. I found no existing PR or issue for InternVL. (`InternVLChatModel`, the remote-code format, is listed in `multimodal_model_archs` but has no model class.)

## Modifications

- `models/internvl.py` (new):
  - InternViT tower: CLS token, learned absolute positions, pre-LN blocks with layer scale, exact GELU.
  - 0.5x pixel shuffle, then the LayerNorm, Linear, GELU, Linear projector.
  - Reuses the existing `Qwen2Model` backbone.
  - Each 448x448 crop is one fixed-size block of 1024 patches in the lane-packed encoder input (`merge_unit=4`), so crops never straddle a lane boundary.
  - `lm_head` handling: tied only when the checkpoint has no `lm_head`. InternVL3-*-hf sets `tie_word_embeddings=True` but ships a different `lm_head`, and HF keeps them untied.
- `multimodal/processors/internvl.py` (new): HF `InternVLProcessor` crops become row-major 14x14 patches in conv-weight order, with one item per image and placeholder spans validated.
- `configs/model_config.py`: register `InternVLForConditionalGeneration` as multimodal.
- `layers/embeddings.py`: `rope_type="dynamic"` maps to unscaled RoPE. HF dynamic NTK only rescales `inv_freq` once a sequence exceeds `max_position_embeddings`, and the served context never exceeds it, so this is exact.
- Tests: `test/srt/models/test_internvl_tiny_parity.py` (+ `tiny_hf_parity_internvl.py`, `internvl_encoder_check.py`).

Not supported yet: video inputs; InternViT-6B variants (qk-norm, RMSNorm); non-Qwen2 language models (the class raises `NotImplementedError` for these).

## Accuracy Tests

**CPU, fp32, tiny random checkpoint with the InternVL3-2B-hf config shape** (real processor and tokenizer), through `Engine` vs `transformers.generate`, greedy:
- Vision tower + projector vs HF `get_image_features` (7-crop image): rel max |diff| 1e-6.
- End to end, 4 requests (7-crop image, 1-crop image, two images in one prompt, text-only): greedy tokens identical, prompt lengths identical, max |logprob diff| 2.9e-6.

**TPU v6e-1, real checkpoint** (bf16) vs HF fp32 on the VM host CPU, greedy, 32 tokens:

| request | matched tokens | max \|dlogp\| on prefix |
|---|---|---|
| 672x448 gradient/checker image (multi-crop), "Describe this image in detail." | 32/32 | 0.059 |
| 300x300 red image, "What color is this square?" | 1/6 | 0.004 (HF "The square is red." vs "The color of the square is red.") |
| text-only | 32/32 | 0.056 |

(Before the `lm_head` fix the text-only request diverged at token 2. That is how the tying issue was found.)

## Benchmarking and Profiling

TPU v6e-1, bf16, `--chunked-prefill-size 4096`. Each request has a unique 896x672 PNG (12 crops + thumbnail = 13 crops, 3,328 image tokens), 128 output tokens with `ignore_eos`. All shapes were warmed before measuring. TTFT includes host preprocessing, the vision encode and prefill. 0 errors.

| concurrency | output tok/s | TTFT p50 / p99 ms | ITL p50 ms |
|---|---|---|---|
| 1 | 177 | 258 / 275 | 3.7 |
| 4 | 415 | 717 / 783 | 4.5 |
| 8 | 550 | 1,053 / 1,412 | 6.7 |
| 16 | 654 | 1,639 / 2,573 | 12.0 |

The encoder uses plain `jax.nn.dot_product_attention` per crop. Prefill of 3,328 image tokens per request (13 crops) dominates TTFT at higher concurrency.

## Checklist

- [x] Please use English, otherwise it will be closed.
- [x] The purpose of the PR, or link existing issues this PR will resolve.
- [x] The test plan: `JAX_PLATFORMS=cpu python -m pytest test/srt/models/test_internvl_tiny_parity.py`; serve with `python -m sgl_jax.launch_server --model-path OpenGVLab/InternVL3-2B-hf --device tpu --attention-backend fa --grammar-backend none` (llguidance rejects this tokenizer: it has one more token than `vocab_size`, which is a pre-existing limitation).
- [ ] (Optional) The necessary documentation update.
