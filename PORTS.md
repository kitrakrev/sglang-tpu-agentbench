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

Full tables (c=1,4,8,16): `results/ports/<model>/bench/`. The VLM and audio ports were checked for
correctness only (agentbench is text-only).

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
- Cost: one spot v6e-1 in europe-west4-a for about 50 min (about $1 to $2.30 at the on-demand ceiling). The VM was deleted afterwards.
