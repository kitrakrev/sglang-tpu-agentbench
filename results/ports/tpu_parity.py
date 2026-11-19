"""Real-checkpoint parity on TPU: sgl_jax server (bf16, TPU) vs HF transformers (fp32, VM host CPU).

Greedy-decodes N tokens for a few prompts on both and reports how many leading
tokens match and the max |logprob| difference over the matched prefix.
bf16-vs-fp32 drift means late divergence is expected; early divergence or a
large logprob gap on the matched prefix indicates a real bug.

usage: python tpu_parity.py <hf_model_id> <text|vlm|audio> <out.json> [--port 30000]
"""

import argparse
import base64
import io
import json
import time

import numpy as np
import requests
import torch

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("kind", choices=["text", "vlm", "audio"])
ap.add_argument("out")
ap.add_argument("--port", type=int, default=30000)
ap.add_argument("--new", type=int, default=32)
args = ap.parse_args()
URL = f"http://localhost:{args.port}"
SR = 16000

TEXT_PROMPTS = [
    "The three primary colors are",
    "Write a Python function that checks whether a number is prime.",
    "Explain in one paragraph why the sky is blue.",
    "List the planets of the solar system in order from the sun.",
]


def test_images():
    from PIL import Image

    yy, xx = np.mgrid[0:448, 0:672]
    grad = np.stack([xx * 255 // 672, yy * 255 // 448, ((xx // 32 + yy // 32) % 2) * 255], -1)
    img = Image.fromarray(grad.astype("uint8"))
    small = Image.new("RGB", (300, 300), (220, 30, 30))
    return img, small


def test_audio():
    t = np.arange(int(4.0 * SR)) / SR
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    chirp = (0.3 * np.sin(2 * np.pi * (200 + 300 * t) * t)).astype(np.float32)
    return tone, chirp


def b64_png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def b64_wav(wav):
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="WAV")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def build_requests(proc):
    def chat(content):
        return proc.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
        )

    if args.kind == "text":
        tok = getattr(proc, "tokenizer", proc)
        return [
            (tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False), {}, {})
            for p in TEXT_PROMPTS
        ]
    if args.kind == "vlm":
        img, small = test_images()
        return [
            (chat([{"type": "image"}, {"type": "text", "text": "Describe this image in detail."}]),
             {"images": [img]}, {"image_data": [b64_png(img)]}),
            (chat([{"type": "image"}, {"type": "text", "text": "What color is this square?"}]),
             {"images": [small]}, {"image_data": [b64_png(small)]}),
            (chat([{"type": "text", "text": TEXT_PROMPTS[2]}]), {}, {}),
        ]
    tone, chirp = test_audio()
    a = {"type": "audio", "audio_url": "x"}
    return [
        (chat([a, {"type": "text", "text": "Describe this sound."}]),
         {"audio": [tone], "sampling_rate": SR}, {"audio_data": [b64_wav(tone)]}),
        (chat([a, {"type": "text", "text": "Is the pitch rising or constant?"}]),
         {"audio": [chirp], "sampling_rate": SR}, {"audio_data": [b64_wav(chirp)]}),
        (chat([{"type": "text", "text": TEXT_PROMPTS[0]}]), {}, {}),
    ]


def hf_model():
    import transformers

    if args.kind == "text":
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    if args.kind == "vlm":
        return transformers.AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).eval()
    return transformers.Qwen2AudioForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32).eval()


def main():
    from transformers import AutoProcessor, AutoTokenizer

    proc = AutoTokenizer.from_pretrained(args.model) if args.kind == "text" else AutoProcessor.from_pretrained(args.model)
    reqs = build_requests(proc)

    got = []
    for text, _, server_mm in reqs:
        body = {
            # text models: send the exact HF token ids (avoids double-BOS differences)
            **({"input_ids": proc(text, add_special_tokens=False)["input_ids"]}
               if args.kind == "text" else {"text": text}),
            "sampling_params": {"max_new_tokens": args.new, "temperature": 0.0},
            "return_logprob": True,
            **server_mm,
        }
        r = requests.post(f"{URL}/generate", json=body, timeout=600)
        r.raise_for_status()
        meta = r.json()["meta_info"]
        got.append(dict(
            tokens=[x[1] for x in meta["output_token_logprobs"]],
            logprobs=[x[0] for x in meta["output_token_logprobs"]],
            text=r.json()["text"],
        ))

    model = hf_model()
    ref = []
    for text, hf_mm, _ in reqs:
        if args.kind == "text":
            enc = proc(text, return_tensors="pt", add_special_tokens=False)
        else:
            enc = proc(text=[text], return_tensors="pt", **hf_mm)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.new, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True)
        new = out.sequences[0, enc["input_ids"].shape[1]:].tolist()
        lps = [torch.log_softmax(s[0].float(), -1)[t].item() for s, t in zip(out.scores, new)]
        tok = getattr(proc, "tokenizer", proc)
        ref.append(dict(tokens=new, logprobs=lps, text=tok.decode(new, skip_special_tokens=True),
                        hf_seconds=round(time.time() - t0, 1)))

    rows = []
    for i, (r, g) in enumerate(zip(ref, got)):
        n = min(len(r["tokens"]), len(g["tokens"]))
        first_div = next((j for j in range(n) if r["tokens"][j] != g["tokens"][j]), None)
        upto = n if first_div is None else first_div
        diff = float(np.max(np.abs(np.subtract(r["logprobs"][:upto], g["logprobs"][:upto])))) if upto else None
        rows.append(dict(request=i, matched_prefix=upto, of=n, first_divergence=first_div,
                         max_logprob_diff_on_prefix=diff, hf_text=r["text"], sgl_text=g["text"]))
        print(f"req {i}: matched {upto}/{n} tokens, max|dlogp| on prefix={diff}")
        print("  HF :", r["text"][:160].replace("\n", " "))
        print("  SGL:", g["text"][:160].replace("\n", " "))
    json.dump(dict(model=args.model, kind=args.kind, new_tokens=args.new,
                   note="sgl_jax bf16 on TPU v6e-1 vs HF transformers fp32 on VM CPU",
                   requests=rows), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
