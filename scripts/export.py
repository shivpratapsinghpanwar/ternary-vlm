"""
Merge LoRA into the ternary LM and export for CPU inference.

    python scripts/export.py --ckpt ckpt/stage2/latest.pt --out export/ternavlm

Produces:
  export/ternavlm/lm/          merged HF checkpoint whose linear weights are exactly ternary (-a, 0, +a)
  export/ternavlm/vision/      SigLIP encoder (unchanged)
  export/ternavlm/projector.pt projector weights + config

Next step (manual for now): convert with llama.cpp
  * LM      -> convert_hf_to_gguf.py --outtype tq2_0   (ternary GGUF; TQ1_0/TQ2_0 kernels run on CPU)
  * vision  -> llama.cpp/tools/mtmd conversion for SigLIP-style encoders (as used by SmolVLM)
  * projector: mtmd expects the projector inside the vision GGUF; write it in the SmolVLM layout
    (LayerNorm -> Linear -> GELU -> Linear over pixel-shuffled features) so no C++ changes are needed.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ternavlm.ternary import merge_all
from ternavlm.vlm import TernaVLM, TernaVLMConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    m = ck["cfg"]["model"]
    model = TernaVLM(TernaVLMConfig(
        lm_name=m["lm_name"], vision_name=m["vision_name"], pixel_shuffle=m.get("pixel_shuffle", 2),
        projector_hidden=m.get("projector_hidden", 2048), lora_r=m.get("lora_r", 0),
        lora_alpha=m.get("lora_alpha", 16.0), quantize_act=m.get("quantize_act", True),
        lora_mode=m.get("lora_mode", "joint"), quantize_lm=m.get("quantize_lm", True),
        torch_dtype=torch.float32,
    ))
    model.load_trainable_state_dict(ck["model"])
    n = merge_all(model.lm)
    print(f"merged {n} layers")

    # sanity: every merged linear is ternary
    bad = [k for k, v in model.lm.state_dict().items()
           if k.endswith("proj.weight") and torch.unique(v.abs()).numel() > 2]
    assert not bad, f"non-ternary weights after merge: {bad[:3]}"

    os.makedirs(args.out, exist_ok=True)
    model.lm.save_pretrained(os.path.join(args.out, "lm"))
    model.tokenizer.save_pretrained(os.path.join(args.out, "lm"))
    model.vision.save_pretrained(os.path.join(args.out, "vision"))
    torch.save({"state_dict": model.projector.state_dict(), "pixel_shuffle": model.cfg.pixel_shuffle,
                "num_image_tokens": model.num_image_tokens, "image_token": model.cfg.image_token},
               os.path.join(args.out, "projector.pt"))
    print("exported to", args.out)


if __name__ == "__main__":
    main()
