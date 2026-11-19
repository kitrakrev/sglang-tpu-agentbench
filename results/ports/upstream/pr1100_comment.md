I tested a Gemma 3 text port on TPU v6e-1 (sgl_jax bf16, `--attention-backend fa`, TP=1) against HF transformers 5.12 (fp32, CPU), greedy, 32 new tokens:

- gemma-3-1b-it: 4 prompts, matched 30/30 (EOS), 32/32, 32/32, 32/32 tokens; max |logprob diff| on the matched prefix 0.02 to 0.10
- gemma-3-4b-it (text tower, served with `--json-model-override-args '{"architectures":["Gemma3ForCausalLM"]}'`): 32/32 on all 4 prompts; max |logprob diff| 0.04 to 0.17

One issue that looks like it applies to this PR too: with a single KV head (gemma-3-1b has `num_key_value_heads=1`), the plain `k.reshape(-1, self.num_kv_heads, self.head_dim)` under explicit mesh axes moves the `tensor` sharding onto `head_dim`. The FA backend then fails at startup:

```
ValueError: in_specs passed to shard_map: P('data', 'tensor', None) does not match the specs of the input: P('data', None, 'tensor') for arg: bfloat16[64@data,1,256@tensor]
```

gemma-3-4b (4 KV heads) was not affected. The fix I used is the same pattern `gemma4.py` already uses:

```python
heads = NamedSharding(self.mesh, P("data", "tensor", None))
q = q.reshape(-1, self.num_heads, self.head_dim, out_sharding=heads)
k = k.reshape(-1, self.num_kv_heads, self.head_dim, out_sharding=heads)
v = v.reshape(-1, self.num_kv_heads, self.head_dim, out_sharding=heads)
```

Separately, gemma-3-1b-it fails to start with the default grammar backend (`ValueError: vocab size too small; 262144 vs 262145` from llguidance, because the tokenizer has one more token than `vocab_size`). I ran it with `--grammar-backend none`. That is not specific to this PR.

Happy to run the TPU parity check on this branch if that's useful.
