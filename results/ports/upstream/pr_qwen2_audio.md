Title: feat(models): add Qwen2-Audio (Qwen2AudioForConditionalGeneration) with in-model audio

## Motivation

Adds `Qwen2AudioForConditionalGeneration` (`Qwen/Qwen2-Audio-7B-Instruct`, `Qwen2-Audio-7B`). The architecture is already listed in `multimodal_model_archs` but has no model class. This is also the first log-mel (`MelInputSpec`) encoder on the in-model multimodal path. I found no existing PR or issue for Qwen2-Audio.

## Modifications

- `models/qwen2_audio.py` (new):
  - Whisper-style encoder: conv1d stem (k=3, stride 1 then 2), learned positions, pre-LN blocks, exact GELU.
  - Padded mel frames are masked as attention keys, matching HF's `feature_attention_mask` handling.
  - 2x average pool, LayerNorm, linear projector, then the existing `Qwen2Model` backbone.
  - Each clip is a fixed 3000-frame window. Its valid tokens (`((L-1)//2+1-2)//2+1`) are packed back to back per lane; the clip count per lane is bucketed to a power of two to bound recompiles.
- `multimodal/processors/qwen2_audio.py` (new): HF `Qwen2AudioProcessor` adapter that carries the valid mel frame count per item. Audio is decoded with soundfile and scipy resampling, with librosa as a fallback only, because the pinned `msgpack-python 0.5.x` breaks `import librosa` (`TypeError: strict_map_key`).
- `multimodal/in_model/host_orchestration.py`: precompile dummy input for `MelInputSpec` encoders.
- `entrypoints/engine.py`: `audio_data` argument on `Engine.generate` (already on `GenerateReqInput`).
- Tests: `test/srt/models/test_qwen2_audio_tiny_parity.py` (+ `tiny_hf_parity_qwen2_audio.py`, `qwen2_audio_encoder_check.py`).

Not supported yet: clips longer than 30 s (HF truncates the same way).

## Accuracy Tests

**CPU, fp32, tiny random checkpoint with the Qwen2-Audio-7B config shape** (real processor and tokenizer), through `Engine` vs `transformers.generate`, greedy:
- Audio tower + projector vs HF at 3.3 s and 7.1 s clips (exercises padded-frame masking): rel max |diff| 7e-7.
- End to end, 4 requests (3.3 s clip, 7.1 s clip, two clips in one prompt, text-only): greedy tokens identical, prompt lengths identical, max |logprob diff| 2e-5.

**TPU v6e-1, real checkpoint** (bf16) vs HF fp32 on the VM host CPU, greedy, 32 tokens:

| request | matched tokens | outputs |
|---|---|---|
| 4 s 440 Hz tone, "Describe this sound." | 8/8 (EOS) | both: "It is a telephone dial tone." |
| 4 s rising chirp, "Is the pitch rising or constant?" | 6/6 (EOS) | both: "The pitch is rising." |
| text-only | 4/32 | rewording ("...of light are red, green, and blue" vs "...are red, blue, and green"). This path is the existing Qwen2 backbone, and it matches HF exactly in the CPU fp32 test. |

## Benchmarking and Profiling

TPU v6e-1, bf16. Each request has a unique 10 s WAV clip (1,000 mel frames, 250 audio tokens), 128 output tokens with `ignore_eos`. All shapes were warmed before measuring. TTFT includes decode, feature extraction, the audio encode and prefill. 0 errors.

| concurrency | output tok/s | TTFT p50 / p99 ms | ITL p50 ms |
|---|---|---|---|
| 1 | 72 | 129 / 129 | 13.0 |
| 4 | 206 | 445 / 1,641 | 14.5 |
| 8 | 410 | 641 / 708 | 14.4 |

## Checklist

- [x] Please use English, otherwise it will be closed.
- [x] The purpose of the PR, or link existing issues this PR will resolve.
- [x] The test plan: `JAX_PLATFORMS=cpu python -m pytest test/srt/models/test_qwen2_audio_tiny_parity.py`; serve with `python -m sgl_jax.launch_server --model-path Qwen/Qwen2-Audio-7B-Instruct --device tpu --attention-backend fa`, then send `audio_data` to `/generate`.
- [ ] (Optional) The necessary documentation update.
