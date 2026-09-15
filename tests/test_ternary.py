"""CPU unit tests for the ternary/LoRA layer. Run: python -m pytest tests -q  (or python tests/test_ternary.py)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, torch.nn as nn
from ternavlm.ternary import (ternary_quant, ternary_quant_ste, int8_act_quant_ste, TernaryLoRALinear, wrap_linears,
                              merge_all, mismatch_report)
from ternavlm.ops import pixel_shuffle

torch.manual_seed(0)


def test_ternary_values():
    w = torch.randn(64, 32)
    q = ternary_quant_ste(w)
    s = w.abs().mean()
    levels = torch.unique((q / s).round())
    assert set(levels.tolist()) <= {-1.0, 0.0, 1.0}
    assert q.shape == w.shape


def test_int8_act_values():
    x = torch.randn(4, 7, 32) * 3
    q = int8_act_quant_ste(x)
    scale = x.abs().amax(-1, keepdim=True) / 127
    ints = (q / scale).round()
    assert ints.abs().max() <= 127
    assert torch.allclose(q, ints * scale, atol=1e-5)


def test_ste_gradient_reaches_lora_not_master():
    base = nn.Linear(32, 16)
    layer = TernaryLoRALinear(base, r=4, alpha=8)
    with torch.no_grad():
        layer.lora_B.normal_()   # at init B=0 so dL/dA=0 by construction; perturb to test the A path
    x = torch.randn(5, 32, requires_grad=True)
    layer(x).pow(2).sum().backward()
    assert layer.lora_A.grad is not None and layer.lora_A.grad.abs().sum() > 0
    assert layer.lora_B.grad is not None and layer.lora_B.grad.abs().sum() > 0
    assert layer.weight.grad is None            # master frozen
    assert x.grad is not None                    # STE passes gradient to inputs


def test_r0_is_pure_ternary_forward():
    base = nn.Linear(32, 16)
    layer = TernaryLoRALinear(base, r=0, quantize_act=False)
    x = torch.randn(3, 32)
    ref = torch.nn.functional.linear(x, ternary_quant_ste(base.weight), base.bias)
    assert torch.allclose(layer(x), ref, atol=1e-6)


def test_merge_is_exact():
    base = nn.Linear(32, 16)
    layer = TernaryLoRALinear(base, r=4, alpha=8, quantize_act=False)
    with torch.no_grad():
        layer.lora_B.normal_()   # make the delta non-trivial
    x = torch.randn(3, 32)
    merged = layer.merge()
    assert torch.allclose(layer(x), merged(x), atol=1e-5)
    # a ternary tensor has at most 3 distinct values: -a, 0, +a
    assert torch.unique(merged.weight.abs()).numel() <= 2


def test_wrap_and_merge_all():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8); self.k_proj = nn.Linear(8, 8); self.other = nn.Linear(8, 8)
    m = nn.Sequential(Block(), Block())
    for p in m.parameters():
        p.requires_grad_(False)   # mirrors TernaVLM: freeze LM, then wrap
    n = wrap_linears(m, r=2)
    assert n == 4
    assert isinstance(m[0].q_proj, TernaryLoRALinear) and isinstance(m[0].other, nn.Linear)
    trainable = [k for k, p in m.named_parameters() if p.requires_grad]
    assert all("lora" in k for k in trainable) and len(trainable) == 8
    assert merge_all(m) == 4 and isinstance(m[0].q_proj, nn.Linear)


def test_pixel_shuffle_shape():
    x = torch.randn(2, 256, 768)
    y = pixel_shuffle(x, 2)
    assert y.shape == (2, 64, 768 * 4)
    y3 = pixel_shuffle(torch.randn(1, 576, 8), 3)
    assert y3.shape == (1, 64, 72)


def test_fp16_stats_stable():
    w = (torch.randn(64, 64) * 1e-3).half()      # tiny weights: absmean in fp16 could underflow
    q = ternary_quant_ste(w)
    assert torch.isfinite(q).all() and q.abs().sum() > 0


# ---- lora_mode ablation -----------------------------------------------------------------

def _lora_delta(layer, x):
    return layer.scaling * (x @ layer.lora_A.t()) @ layer.lora_B.t()


def _wrapped_pair_model(lora_mode, r=2, quantize_weight=True, quantize_act=True):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8); self.v_proj = nn.Linear(8, 8); self.other = nn.Linear(8, 8)
    m = nn.Sequential(Block(), Block())
    for p in m.parameters():
        p.requires_grad_(False)
    n = wrap_linears(m, r=r, lora_mode=lora_mode, quantize_weight=quantize_weight, quantize_act=quantize_act)
    assert n == 4
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, TernaryLoRALinear) and mod.r > 0:
                mod.lora_B.normal_()
    return m


def test_plain_mode_is_frozen_ternary_plus_fp_delta():
    base = nn.Linear(32, 16)
    layer = TernaryLoRALinear(base, r=4, alpha=8, lora_mode="plain")
    with torch.no_grad():
        layer.lora_B.normal_()
    x = torch.randn(5, 32)
    frozen = TernaryLoRALinear(nn.Linear(32, 16), r=0)
    frozen.weight, frozen.bias = base.weight, base.bias
    ref = frozen(x) + _lora_delta(layer, x)                    # Qa(x)·Q(W)ᵀ + s·(x·Aᵀ)·Bᵀ, delta unquantized
    assert torch.allclose(layer(x), ref, atol=1e-5)
    joint = TernaryLoRALinear(base, r=4, alpha=8, lora_mode="joint")
    with torch.no_grad():
        joint.lora_A.copy_(layer.lora_A); joint.lora_B.copy_(layer.lora_B)
    assert not torch.allclose(joint(x), layer(x), atol=1e-3)   # the two modes are genuinely different
    # gradients reach A/B (and x), never the frozen master
    xg = x.clone().requires_grad_(True)
    layer(xg).pow(2).sum().backward()
    assert layer.lora_A.grad.abs().sum() > 0 and layer.lora_B.grad.abs().sum() > 0
    assert layer.weight.grad is None and xg.grad is not None


def test_plain_merge_requantizes_and_mismatch_report():
    base = nn.Linear(32, 16)
    layer = TernaryLoRALinear(base, r=4, alpha=8, lora_mode="plain", quantize_act=False)
    with torch.no_grad():
        layer.lora_B.normal_()
    x = torch.randn(3, 32)
    tern, fp = layer.merge(), layer.merge_fp()
    assert torch.unique(tern.weight.abs()).numel() <= 2                       # ternary export
    assert torch.unique(fp.weight.abs()).numel() > 2                          # fp export is not ternary
    assert torch.allclose(fp.weight, layer.effective_weight())                # merge_fp = W + s·BA exactly
    assert torch.allclose(tern.weight, ternary_quant(layer.effective_weight()))  # merge = Q(W + s·BA)
    assert not torch.allclose(tern.weight, fp.weight)
    # post-hoc requantization is the train/test mismatch: the merged ternary layer does not reproduce training
    assert not torch.allclose(layer(x), tern(x), atol=1e-3)
    # what training actually used is Q(W) + s·BA: neither export equals it (the fp merge W + s·BA differs too)
    assert torch.allclose(layer(x), torch.nn.functional.linear(x, layer.train_weight(), layer.bias), atol=1e-5)
    assert not torch.allclose(layer(x), fp(x), atol=1e-3)

    m = _wrapped_pair_model("plain")
    rep = mismatch_report(m)
    assert rep["n_layers"] == 4 and rep["lora_modes"] == ["plain"]
    vals = [rep[k] for k in ("mean_export_vs_fp", "max_export_vs_fp", "mean_export_vs_train", "max_export_vs_train")]
    assert all(torch.isfinite(torch.tensor(v)) for v in vals) and all(v > 0 for v in vals)
    for v in rep["layers"].values():
        assert 0 < v["export_vs_fp"] < 10 and 0 < v["export_vs_train"] < 10
    # merge_all(fp=True) exports the unquantized weights
    assert merge_all(m, fp=True) == 4 and torch.unique(m[0].q_proj.weight.abs()).numel() > 2


def test_joint_mode_unchanged():
    base = nn.Linear(32, 16)
    default = TernaryLoRALinear(base, r=4, alpha=8, quantize_act=False)
    explicit = TernaryLoRALinear(base, r=4, alpha=8, quantize_act=False, lora_mode="joint")
    assert default.lora_mode == "joint"
    with torch.no_grad():
        default.lora_B.normal_(); explicit.lora_A.copy_(default.lora_A); explicit.lora_B.copy_(default.lora_B)
    x = torch.randn(3, 32)
    assert torch.equal(default(x), explicit(x))
    assert torch.allclose(explicit(x), explicit.merge()(x), atol=1e-5)      # merge still exact
    m = _wrapped_pair_model("joint")
    rep = mismatch_report(m)
    assert rep["n_layers"] == 4 and rep["max_export_vs_train"] == 0.0     # joint: export == what was trained
    assert rep["mean_export_vs_fp"] > 0


def test_none_mode_is_frozen_ternary():
    base = nn.Linear(32, 16)
    layer = TernaryLoRALinear(base, r=32, alpha=64, lora_mode="none", quantize_act=False)
    assert layer.r == 0 and layer.lora_A is None and layer.lora_B is None
    x = torch.randn(3, 32)
    assert torch.allclose(layer(x), torch.nn.functional.linear(x, ternary_quant_ste(base.weight), base.bias), atol=1e-6)
    m = _wrapped_pair_model("none", r=32)
    assert not any(p.requires_grad for p in m.parameters())
    assert mismatch_report(m)["max_export_vs_train"] == 0.0


def test_fp_control_equals_linear_plus_lora():
    base = nn.Linear(32, 16)
    x = torch.randn(5, 32)
    for mode in ("plain", "joint"):
        layer = TernaryLoRALinear(base, r=4, alpha=8, lora_mode=mode, quantize_weight=False, quantize_act=False)
        with torch.no_grad():
            layer.lora_B.normal_()
        ref = base(x) + _lora_delta(layer, x)
        assert torch.allclose(layer(x), ref, atol=1e-5), mode
        assert torch.allclose(layer.merge().weight, layer.merge_fp().weight), mode   # nothing to quantize
        assert torch.allclose(layer.merge()(x), ref, atol=1e-5), mode
    layer = TernaryLoRALinear(base, r=0, quantize_weight=False, quantize_act=False)
    assert torch.allclose(layer(x), base(x), atol=1e-6)                        # r=0 fp control == nn.Linear
    m = _wrapped_pair_model("plain", quantize_weight=False, quantize_act=False)
    assert mismatch_report(m)["max_export_vs_fp"] == 0.0


def test_lora_mode_validation():
    try:
        TernaryLoRALinear(nn.Linear(4, 4), r=2, lora_mode="bogus")
    except ValueError:
        return
    raise AssertionError("bad lora_mode accepted")


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for f in fns:
        f(); print("ok ", f.__name__)
