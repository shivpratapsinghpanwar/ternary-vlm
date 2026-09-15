"""
Ternary (1.58-bit) linear layers with LoRA that is quantized *jointly* with the base weight.

Why this exists
---------------
Plain LoRA on a BitNet model computes  y = x @ Q(W)^T + s * x @ (B A)^T.
The delta B A is full precision, so the deployed model is no longer ternary,
and merging then re-quantizing gives a train/test mismatch.

Here we compute  y = Qa(x) @ Q(W + s * B A)^T  with a straight-through estimator,
so after training you merge W' = W + s * B A, quantize once, and the exported
model is exactly what was trained.

Quantization follows BitNet b1.58:
  weights:     W_q = RoundClip(W / (mean|W| + eps), -1, 1) * mean|W|      (per-tensor)
  activations: x_q = RoundClip(x * 127 / max|x|, -128, 127) * max|x| / 127  (per-token)

LoRA modes (the paper's ablation, selected with `lora_mode`):
  "joint"  y = Qa(x) @ Q(W + s*BA)^T            STE; merge() is exact.                (default)
  "plain"  y = Qa(x) @ Q(W)^T + s*(x A^T) B^T   delta in full precision (what peft does);
           merge() does the post-hoc merge-and-requantize Q(W + s*BA) -> train/test mismatch.
  "none"   y = Qa(x) @ Q(W)^T                   r forced to 0, frozen ternary (projector-only).
With quantize_weight=False (and quantize_act=False) the same module is ordinary fp LoRA on an
unquantized backbone, used for the matched full-precision control.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def ternary_quant(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-tensor absmean ternary quantization (exact values, no gradient). Used for export."""
    # Do the statistics in fp32 so fp16 training on T4 doesn't under/overflow.
    w32 = w.float()
    scale = w32.abs().mean().clamp_min(eps)
    return ((w32 / scale).round().clamp_(-1, 1) * scale).to(w.dtype)


def ternary_quant_ste(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """ternary_quant with a straight-through gradient. Value may differ from ternary_quant by 1 ulp."""
    return w + (ternary_quant(w, eps) - w).detach()


def int8_act_quant_ste(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-token absmax int8 activation quantization with straight-through gradient."""
    x32 = x.float()
    scale = x32.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / 127.0
    xq = (x32 / scale).round().clamp_(-128, 127) * scale
    xq = xq.to(x.dtype)
    return x + (xq - x).detach()


LORA_MODES = ("joint", "plain", "none")


class TernaryLoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear (or a BitNet BitLinear) that:
      * keeps the frozen master weight W (latent full-precision weights of the ternary model)
      * adds trainable LoRA A/B
      * quantizes (W + s*BA) to ternary and the input to int8 in every forward pass ("joint" mode)

    lora_mode: "joint" (default), "plain" (fp delta on a frozen ternary base) or "none" (r=0).
    quantize_weight=False disables the ternary quantizer entirely (fp control backbone).
    Set r=0 to get a pure frozen ternary linear (used for stage 1 where only the projector trains).
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 0,
        alpha: float = 16.0,
        dropout: float = 0.0,
        quantize_act: bool = True,
        lora_mode: str = "joint",
        quantize_weight: bool = True,
    ):
        super().__init__()
        if lora_mode not in LORA_MODES:
            raise ValueError(f"lora_mode must be one of {LORA_MODES}, got {lora_mode!r}")
        if lora_mode == "none":
            r = 0
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.scaling = alpha / r if r > 0 else 0.0
        self.quantize_act = quantize_act
        self.quantize_weight = quantize_weight
        self.lora_mode = lora_mode

        # Master weight: frozen. We keep the original Parameter object so state_dict names line up.
        self.weight = base.weight
        self.weight.requires_grad_(False)
        self.bias = base.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)

        if r > 0:
            # LoRA params stay fp32 (GradScaler refuses fp16 grads); the fp16 master weight is promoted inside
            # effective_weight(), and autocast casts the quantized result back to fp16 for the matmul.
            self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=torch.float32, device=self.weight.device))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=torch.float32, device=self.weight.device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()

    def effective_weight(self) -> torch.Tensor:
        """Full-precision merged weight W + s*BA (what merge_fp exports; merge quantizes it)."""
        if self.r > 0:
            return self.weight.float() + self.scaling * (self.lora_B @ self.lora_A)
        return self.weight

    def _q(self, w: torch.Tensor) -> torch.Tensor:
        return ternary_quant_ste(w) if self.quantize_weight else w

    @torch.no_grad()
    def train_weight(self) -> torch.Tensor:
        """The weight the forward pass effectively uses (exact quantizer values, for the mismatch report).
        joint: Q(W + s*BA)   plain: Q(W) + s*BA   none / r=0: Q(W)."""
        q = ternary_quant if self.quantize_weight else (lambda w: w)
        if self.r > 0 and self.lora_mode == "plain":
            return q(self.weight.float()) + self.scaling * (self.lora_B @ self.lora_A)
        return q(self.effective_weight().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        if self.quantize_act:
            x = int8_act_quant_ste(x)
        if self.r > 0 and self.lora_mode == "plain":
            # Standard LoRA: frozen (ternary) base + full-precision delta path, dropout on the delta input.
            w = self._q(self.weight)
            a, b = self.lora_A, self.lora_B
            if not torch.is_autocast_enabled():
                if w.dtype != x.dtype:
                    w = w.to(x.dtype)
                if a.dtype != x.dtype:
                    a, b = a.to(x.dtype), b.to(x.dtype)
            delta = F.linear(F.linear(self.lora_dropout(x_in), a), b) * self.scaling
            return F.linear(x, w, self.bias) + delta
        # "joint" (and "none"/r=0): the delta is fused into the weight before quantization, so there is no
        # separate delta path and LoRA dropout is left as identity (kept only for API compatibility).
        w = self._q(self.effective_weight())
        if not torch.is_autocast_enabled() and w.dtype != x.dtype:
            w = w.to(x.dtype)
        return F.linear(x, w, self.bias)

    def _linear_with(self, w: torch.Tensor) -> nn.Linear:
        lin = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None).to(self.weight.dtype)
        lin.weight.copy_(w.to(self.weight.dtype))
        if self.bias is not None:
            lin.bias.copy_(self.bias)
        return lin

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Return a plain nn.Linear whose weight is the *quantized* merged weight Q(W + s*BA).
        joint: exact export (bit-for-bit what was trained). plain: post-hoc merge-and-requantize, i.e. the
        train/test mismatch under study. quantize_weight=False: nothing to quantize, same as merge_fp()."""
        w = self.effective_weight()
        return self._linear_with(ternary_quant(w) if self.quantize_weight else w)

    @torch.no_grad()
    def merge_fp(self) -> nn.Linear:
        """Return a plain nn.Linear with the *unquantized* merged weight W + s*BA (fp export, for measuring
        the mismatch against merge())."""
        return self._linear_with(self.effective_weight())

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, r={self.r}, mode={self.lora_mode}, "
                f"w_q={self.quantize_weight}, act_q={self.quantize_act}")


DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def wrap_linears(
    model: nn.Module,
    r: int,
    alpha: float = 16.0,
    targets: Iterable[str] = DEFAULT_TARGETS,
    quantize_act: bool = True,
    lora_mode: str = "joint",
    quantize_weight: bool = True,
    dropout: float = 0.0,
) -> int:
    """
    Replace every nn.Linear (or BitLinear-like module exposing .weight/.in_features/.out_features)
    whose attribute name is in `targets` with TernaryLoRALinear.

    IMPORTANT: if the loaded HF model already applies BitNet quantization inside its own
    linear layers, wrapping them here would double-quantize. We therefore replace the module
    entirely rather than wrapping its forward, and we read only its .weight / .bias.
    lora_mode / quantize_weight are passed through (quantize_weight=False -> plain fp LoRA control).
    Returns the number of replaced modules.
    """
    targets = set(targets)
    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and hasattr(child, "weight") and child.weight.dim() == 2:
                base = nn.Linear(child.weight.shape[1], child.weight.shape[0], bias=getattr(child, "bias", None) is not None)
                base.weight = child.weight
                if getattr(child, "bias", None) is not None:
                    base.bias = child.bias
                setattr(module, child_name, TernaryLoRALinear(
                    base, r=r, alpha=alpha, dropout=dropout, quantize_act=quantize_act,
                    lora_mode=lora_mode, quantize_weight=quantize_weight))
                replaced += 1
    return replaced


def _rel_l2(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """||a - b||_2 / ||b||_2, computed in fp32."""
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(eps)).item()


@torch.no_grad()
def mismatch_report(model: nn.Module) -> dict:
    """
    Cheap per-layer train/test-mismatch report for a wrapped model (used in the paper's ablation).

    For every TernaryLoRALinear it compares
      export_vs_fp    : ternary export Q(W + s*BA)  vs  fp merge W + s*BA        (merge() vs merge_fp())
      export_vs_train : ternary export Q(W + s*BA)  vs  the weight the forward pass actually used
                        (joint: identical -> 0; plain: Q(W) + s*BA -> the mismatch we want to measure)
    as relative L2 distances. Returns {"layers": {name: {...}}, "mean_export_vs_fp", "max_export_vs_fp",
    "mean_export_vs_train", "max_export_vs_train", "n_layers", "lora_modes"}.
    """
    layers: dict = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, TernaryLoRALinear):
            continue
        w_fp = mod.effective_weight().float()
        w_q = ternary_quant(w_fp) if mod.quantize_weight else w_fp
        layers[name] = {
            "export_vs_fp": _rel_l2(w_q, w_fp),
            "export_vs_train": _rel_l2(w_q, mod.train_weight()),
            "lora_mode": mod.lora_mode,
            "r": mod.r,
        }
    fp = [v["export_vs_fp"] for v in layers.values()]
    tr = [v["export_vs_train"] for v in layers.values()]
    return {
        "layers": layers,
        "n_layers": len(layers),
        "lora_modes": sorted({v["lora_mode"] for v in layers.values()}),
        "mean_export_vs_fp": sum(fp) / len(fp) if fp else 0.0,
        "max_export_vs_fp": max(fp) if fp else 0.0,
        "mean_export_vs_train": sum(tr) / len(tr) if tr else 0.0,
        "max_export_vs_train": max(tr) if tr else 0.0,
    }


def lora_state_dict(model: nn.Module) -> dict:
    return {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


@torch.no_grad()
def merge_all(model: nn.Module, fp: bool = False) -> int:
    """Replace every TernaryLoRALinear with its merged nn.Linear (for export).
    fp=False: quantized merge() (ternary export). fp=True: merge_fp() (unquantized, for the mismatch study)."""
    merged = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, TernaryLoRALinear):
                setattr(module, child_name, child.merge_fp() if fp else child.merge())
                merged += 1
    return merged


@torch.no_grad()
def flip_fraction(model: nn.Module, max_layers: int = 6) -> dict:
    """Fraction of ternary states that differ between Q(W + s*BA) and Q(W), on a spread of layers.
    Cheap diagnostic for the paper: with a per-tensor absmean scale, LoRA only changes the exported model
    once |delta| crosses ~0.5*mean|W| for some weights. 0.0 means the export still equals the base model."""
    layers = [m for m in model.modules() if isinstance(m, TernaryLoRALinear) and m.r > 0]
    if not layers:
        return {"flip_frac": 0.0, "n_layers": 0}
    step = max(1, len(layers) // max_layers)
    fracs, rel = [], []
    for m in layers[::step][:max_layers]:
        base_q = ternary_quant(m.weight.float())
        eff = m.effective_weight().float()
        new_q = ternary_quant(eff)
        fracs.append((base_q != new_q).float().mean().item())
        rel.append(((eff - m.weight.float()).norm() / m.weight.float().norm()).item())
    return {"flip_frac": sum(fracs) / len(fracs), "delta_rel": sum(rel) / len(rel), "n_layers": len(fracs)}
