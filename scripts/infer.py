"""
Single-image inference from a train.py checkpoint (PyTorch path, no llama.cpp needed).

    python scripts/infer.py --ckpt ckpt/stage1/latest.pt --image cat.jpg --question "What is in the image?"

Prints the answer plus prefill/decode timing so CPU numbers can go straight into the README.
"""

import argparse
import io
import os
import sys
import time
import urllib.request

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ternavlm.data import build_image_processor
from ternavlm.vlm import TernaVLM, TernaVLMConfig


def load_model(ckpt_path: str, device: str) -> TernaVLM:
    ck = torch.load(ckpt_path, map_location="cpu")
    m = ck["cfg"]["model"]
    model = TernaVLM(TernaVLMConfig(
        lm_name=m["lm_name"], vision_name=m["vision_name"], pixel_shuffle=m.get("pixel_shuffle", 2),
        projector_hidden=m.get("projector_hidden", 2048), lora_r=m.get("lora_r", 0),
        lora_alpha=m.get("lora_alpha", 16.0), quantize_act=m.get("quantize_act", True),
        lora_mode=m.get("lora_mode", "joint"), quantize_lm=m.get("quantize_lm", True),
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ))
    model.load_trainable_state_dict(ck["model"])
    return model.to(device).eval()


def build_prompt(model: TernaVLM, question: str) -> torch.Tensor:
    """Exactly the training format of ternavlm/data.py: BOS + 'User: <image>*N\\n{q}\\nAssistant:'.
    The answer is trained as ' ' + text + EOS, so generation continues right after the colon."""
    tok = model.tokenizer
    img_block = model.cfg.image_token * model.num_image_tokens
    ids = [tok.bos_token_id] if tok.bos_token_id is not None else []
    ids += tok.encode(f"User: {img_block}\n{question}\nAssistant:", add_special_tokens=False)
    return torch.tensor([ids])


def load_image(src: str) -> Image.Image:
    if src.startswith("http"):
        return Image.open(io.BytesIO(urllib.request.urlopen(src).read())).convert("RGB")
    return Image.open(src).convert("RGB")


@torch.no_grad()
def answer(model: TernaVLM, proc, image: Image.Image, question: str, max_new_tokens: int = 64) -> tuple[str, dict]:
    device = next(model.parameters()).device
    ids = build_prompt(model, question).to(device)
    pix = proc(images=image, return_tensors="pt")["pixel_values"].to(device, next(model.vision.parameters()).dtype)
    t0 = time.time()
    out = model.generate(ids, torch.ones_like(ids), pix, max_new_tokens=max_new_tokens, do_sample=False,
                         eos_token_id=model.tokenizer.eos_token_id, pad_token_id=model.tokenizer.eos_token_id)
    dt = time.time() - t0
    # with inputs_embeds, recent transformers return only the new tokens; older versions echo the prompt
    new = out[0][ids.shape[1]:] if out.shape[1] > ids.shape[1] and torch.equal(out[0][: ids.shape[1]], ids[0]) else out[0]
    text = model.tokenizer.decode(new, skip_special_tokens=True).strip()
    return text, {"seconds": round(dt, 2), "new_tokens": int(new.numel()), "tok_per_s": round(new.numel() / max(dt, 1e-6), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", default="Describe the image.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    model = load_model(a.ckpt, a.device)
    proc = build_image_processor(model.cfg.vision_name)
    text, stats = answer(model, proc, load_image(a.image), a.question, a.max_new_tokens)
    print("Q:", a.question)
    print("A:", text)
    print(stats)


if __name__ == "__main__":
    main()
