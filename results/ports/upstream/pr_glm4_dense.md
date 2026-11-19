Title: feat(models): add GLM-4 dense (Glm4ForCausalLM)

## Motivation

Adds the dense GLM-4 architecture (`Glm4ForCausalLM`, e.g. `zai-org/GLM-4-9B-0414`, GLM-Z1-9B). sgl_jax currently supports only the MoE variants (`glm4_moe.py`, `glm5_moe.py`). I found no open PR or issue for the dense model.

## Modifications

- `python/sgl_jax/srt/models/glm4.py` (new):
  - sandwich norms (`post_self_attn_layernorm`, `post_mlp_layernorm`);
  - fused checkpoint `gate_up_proj`, split at load time with `split_sizes`;
  - partial (`partial_rotary_factor=0.5`) GPT-J-style interleaved RoPE;
  - q/k/v bias with a bias-free `o_proj`;
  - q/k/v reshapes pin `P("data", "tensor", None)` (same as `gemma4.py`).
- `test/srt/models/test_glm4_tiny_parity.py` + `tiny_hf_parity.py`: CPU parity test against HF transformers on a tiny random checkpoint.

## Accuracy Tests

**CPU, fp32, tiny random checkpoint with the real GLM-4-9B-0414 config shape** (3 layers, 4 heads / 2 KV heads, head_dim 128, real tokenizer), served through `Engine` and compared with `transformers.generate`, greedy, 16 tokens, 4 prompts:
- greedy tokens identical on all prompts; max |logprob diff| 2.9e-6.
- Negative control: switching RoPE to neox style makes the test fail (first divergence at tokens 0 to 2).

**TPU v6e-1, real checkpoint** (`zai-org/GLM-4-9B-0414`, bf16, `--attention-backend fa`) vs HF transformers fp32 on the VM host CPU, greedy, 32 tokens:

| prompt | matched tokens | max \|dlogp\| on matched prefix |
|---|---|---|
| primary colors | 32/32 | 0.029 |
| prime-check function | 16/32 | 0.013 |
| why is the sky blue | 7/32 | 0.018 |
| planets in order | 32/32 | 0.065 |

The two divergences are rewordings of the same answer (bf16 vs fp32 near-ties), e.g. "...due to a phenomenon called Rayleigh scattering" vs "...due to Rayleigh scattering, a phenomenon...".

## Benchmarking and Profiling

TPU v6e-1, bf16, radix cache on, multi-turn agent workload (4 sessions per level x 6 turns, ~1.2k-token prompts sharing a system prompt, 128 max output tokens), 0 errors:

| concurrency | output tok/s | TTFT p50 / p99 ms | ITL p50 ms |
|---|---|---|---|
| 1 | 61 | 38.8 / 42.4 | 15.7 |
| 4 | 202 | 48.4 / 67.5 | 17.0 |
| 8 | 404 | 48.7 / 69.6 | 18.1 |
| 16 | 739 | 49.2 / 89.8 | 20.4 |

## Checklist

- [x] Please use English, otherwise it will be closed.
- [x] The purpose of the PR, or link existing issues this PR will resolve.
- [x] The test plan: `JAX_PLATFORMS=cpu python -m pytest test/srt/models/test_glm4_tiny_parity.py` (needs network for config/tokenizer); serve with `python -m sgl_jax.launch_server --model-path zai-org/GLM-4-9B-0414 --device tpu --attention-backend fa`.
- [ ] (Optional) The necessary documentation update.
