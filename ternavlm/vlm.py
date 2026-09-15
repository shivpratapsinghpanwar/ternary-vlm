"""
TernaVLM: SigLIP vision encoder -> pixel-shuffle -> MLP projector -> ternary BitNet LM.

Image tokens are spliced into the text embedding sequence at the position of a
placeholder token (`<image>`), LLaVA-style. Each image becomes `num_image_tokens`
embeddings (default 64 = (16x16 patches at 256px) / pixel_shuffle 2x2 = 8x8... adjusted per config).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

from .ternary import wrap_linears, lora_state_dict
from .ops import pixel_shuffle


def _fp32_mlp_forward(self, x):
    """BitNet MLP with the relu^2 * up product in fp32.
    In fp16, relu(g)^2 overflows for |g| > 256 (65504 max), giving NaN losses on T4. The sub-layer RMSNorm
    already computes in fp32 and rescales to unit range, so its output is safe to hand back to fp16."""
    g = self.act_fn(self.gate_proj(x).float())
    u = self.up_proj(x).float()
    return self.down_proj(self.ffn_sub_norm(g * u).to(x.dtype))


def make_mlp_fp16_safe(lm) -> int:
    import types
    n = 0
    for layer in getattr(getattr(lm, "model", lm), "layers", []):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "ffn_sub_norm") and hasattr(mlp, "act_fn"):
            mlp.forward = types.MethodType(_fp32_mlp_forward, mlp)
            n += 1
    return n


@dataclass
class TernaVLMConfig:
    lm_name: str = "microsoft/bitnet-b1.58-2B-4T-bf16"
    vision_name: str = "google/siglip2-base-patch16-256"
    pixel_shuffle: int = 2           # 2 -> tokens / 4, 3 -> tokens / 9
    projector_hidden: int = 2048
    lora_r: int = 0                  # 0 = frozen LM (stage 1); 16-32 for stage 2
    lora_alpha: float = 16.0
    lora_mode: str = "joint"         # "joint" (ours) | "plain" (peft-style fp delta) | "none" (r=0)
    quantize_act: bool = True
    quantize_lm: bool = True         # False = fp control backbone: no ternary quantizer, plain fp LoRA
    prequantize_frozen: bool = True  # r=0: store Q(W) once instead of re-quantizing 2.5B weights every forward
    fp32_residual: bool = True       # keep the LM residual stream in fp32: BitNet's hidden states reach |h| ~ 6e4,
                                     # which overflows fp16 (kernel v4 went non-finite at decoder layer 7)
    freeze_vision: bool = True
    image_token: str = "<image>"
    torch_dtype: torch.dtype = torch.float16   # T4 has no bf16; use fp16 + GradScaler


def _from_pretrained(cls, name, dtype, **kw):
    """transformers >=5 renamed torch_dtype -> dtype; support both."""
    try:
        return cls.from_pretrained(name, dtype=dtype, **kw)
    except TypeError:
        return cls.from_pretrained(name, torch_dtype=dtype, **kw)


class Projector(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TernaVLM(nn.Module):
    def __init__(self, cfg: TernaVLMConfig):
        super().__init__()
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.lm_name)
        if cfg.image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [cfg.image_token]})
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(cfg.image_token)

        # Language model. The BitNet checkpoint ships a quantization_config that makes transformers swap in
        # its own online-quantizing linear layers. We strip it so the LM loads with plain nn.Linear holding
        # the latent bf16 weights, and TernaryLoRALinear becomes the *only* quantizer (no double quantization).
        lm_config = AutoConfig.from_pretrained(cfg.lm_name)
        if hasattr(lm_config, "quantization_config"):
            print(f"[TernaVLM] stripping quantization_config {getattr(lm_config, 'quantization_config')}")
            del lm_config.quantization_config
        self.lm = _from_pretrained(AutoModelForCausalLM, cfg.lm_name, cfg.torch_dtype, config=lm_config)
        if self.lm.get_input_embeddings().num_embeddings < len(self.tokenizer):
            self.lm.resize_token_embeddings(len(self.tokenizer))
        for p in self.lm.parameters():
            p.requires_grad_(False)
        if cfg.quantize_lm:
            # Ternary regime: TernaryLoRALinear quantizes W (+ s*BA in "joint" mode) and the activations.
            n = wrap_linears(self.lm, r=cfg.lora_r, alpha=cfg.lora_alpha, quantize_act=cfg.quantize_act,
                             lora_mode=cfg.lora_mode, quantize_weight=True,
                             prequantize=cfg.prequantize_frozen and cfg.lora_r == 0)
        else:
            # Matched full-precision control (e.g. Qwen2.5-1.5B-Instruct / SmolLM2-1.7B-Instruct): the ternary
            # wrapping is NOT applied; the same module is used as an ordinary fp LoRA (unquantized W, fp delta).
            n = wrap_linears(self.lm, r=cfg.lora_r, alpha=cfg.lora_alpha, quantize_act=False,
                             lora_mode=cfg.lora_mode, quantize_weight=False)
        # Use the config of whichever LM actually loaded (BitNet, Qwen2.5, SmolLM2 all use q/k/v/o/gate/up/down).
        expected = 7 * self.lm.config.num_hidden_layers
        print(f"[TernaVLM] replaced {n} linear layers with TernaryLoRALinear (r={cfg.lora_r}, mode={cfg.lora_mode}, "
              f"quantize_lm={cfg.quantize_lm}); expected {expected}")
        assert n == expected, f"replaced {n} linears, expected {expected}: layer names differ from q/k/v/o/gate/up/down"
        # BitNet's relu^2 MLP overflows fp16 (kernels v1-v3 produced NaN on the first forward on T4); do that
        # product in fp32. No-op for LMs without ffn_sub_norm/act_fn (the fp control backbones).
        n_mlp = make_mlp_fp16_safe(self.lm)
        print(f"[TernaVLM] fp16-safe MLP forward on {n_mlp} layers")
        if cfg.fp32_residual and cfg.torch_dtype == torch.float16:
            # Embedding output in fp32 => every decoder layer's `residual + f(norm(residual))` stays fp32 while the
            # linears/attention still run in fp16 under autocast (RMSNorm already normalises in fp32). Covers both
            # the training path (_splice builds embeds through this module) and generate()'s decode steps.
            self.lm.get_input_embeddings().register_forward_hook(lambda mod, inp, out: out.float())
            print("[TernaVLM] fp32 residual stream (embedding output upcast)")

        # Vision encoder.
        self.vision = _from_pretrained(SiglipVisionModel, cfg.vision_name, cfg.torch_dtype)
        if cfg.freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad_(False)

        v_dim = self.vision.config.hidden_size * (cfg.pixel_shuffle ** 2)
        lm_dim = self.lm.get_input_embeddings().embedding_dim
        self.projector = Projector(v_dim, cfg.projector_hidden, lm_dim)  # fp32: it is trained under GradScaler

        patches = (self.vision.config.image_size // self.vision.config.patch_size) ** 2
        self.num_image_tokens = patches // (cfg.pixel_shuffle ** 2)
        print(f"[TernaVLM] {patches} patches -> {self.num_image_tokens} image tokens")

    # ---- image path -------------------------------------------------------------------
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not self.cfg.freeze_vision):
            feats = self.vision(pixel_values=pixel_values).last_hidden_state  # (B, N, C)
        feats = pixel_shuffle(feats, self.cfg.pixel_shuffle)
        return self.projector(feats.float())  # (B, T_img, D)

    def _splice(self, input_ids: torch.Tensor, image_embeds: torch.Tensor) -> torch.Tensor:
        """Replace each <image> token embedding with the projected image tokens.
        The tokenizer side must already have expanded <image> to num_image_tokens copies."""
        embeds = self.lm.get_input_embeddings()(input_ids)
        mask = input_ids == self.image_token_id  # (B, L)
        n_slots = int(mask.sum().item())
        n_img = image_embeds.shape[0] * image_embeds.shape[1]
        assert n_slots == n_img, f"{n_slots} <image> slots but {n_img} image embeddings"
        embeds = embeds.clone()
        embeds[mask] = image_embeds.reshape(-1, image_embeds.shape[-1]).to(embeds.dtype)
        return embeds

    def autocast(self):
        """fp16 weights on CUDA need autocast (fp32 residual stream + fp16 linears); the trainer's own autocast
        nests harmlessly, and inference scripts get the same numerics without knowing about it."""
        p = next(self.lm.parameters())
        if p.is_cuda and p.dtype == torch.float16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        with self.autocast():
            image_embeds = self.encode_images(pixel_values)
            inputs_embeds = self._splice(input_ids, image_embeds)
            return self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, use_cache=False)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, pixel_values, **gen_kwargs):
        with self.autocast():
            image_embeds = self.encode_images(pixel_values)
            inputs_embeds = self._splice(input_ids, image_embeds)
            return self.lm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_kwargs)

    # ---- checkpointing: only the small trainable parts ---------------------------------
    def trainable_state_dict(self) -> dict:
        sd = {f"projector.{k}": v for k, v in self.projector.state_dict().items()}
        sd.update({f"lm.{k}": v for k, v in lora_state_dict(self.lm).items()})
        if not self.cfg.freeze_vision:
            sd.update({f"vision.{k}": v for k, v in self.vision.state_dict().items()})
        # the resized embedding rows for <image> (tiny) so the placeholder is stable across resumes
        emb = self.lm.get_input_embeddings().weight
        sd["image_token_row"] = emb[self.image_token_id].detach().clone()
        return sd

    def load_trainable_state_dict(self, sd: dict):
        missing, unexpected = self.load_state_dict({k: v for k, v in sd.items() if k != "image_token_row"}, strict=False)
        if "image_token_row" in sd:
            with torch.no_grad():
                self.lm.get_input_embeddings().weight[self.image_token_id].copy_(sd["image_token_row"])
        unexpected = [u for u in unexpected]
        if unexpected:
            raise RuntimeError(f"unexpected keys in trainable checkpoint: {unexpected[:5]}")

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
