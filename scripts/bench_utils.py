"""
Helpers for scripts/bench_cpu.py: machine info, peak RSS, synthetic image, timing statistics,
parameter/byte accounting, and the int8 ternary matmul proxy layer.

Nothing here downloads anything. `python scripts/bench_utils.py` runs a tiny self-test on random tensors.
"""

from __future__ import annotations

import math
import os
import platform
import statistics
import sys
import time
from typing import Iterable

import torch
import torch.nn as nn
from PIL import Image


# ----------------------------------------------------------------------------------------------
# machine / memory
# ----------------------------------------------------------------------------------------------
def machine_info() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    try:
        import transformers  # noqa: F401
        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = None
    cpu = None
    try:
        import cpuinfo  # py-cpuinfo, optional
        cpu = cpuinfo.get_cpu_info().get("brand_raw")
    except Exception:
        pass
    info["cpu"] = cpu or platform.processor() or "unknown"
    try:
        import psutil
        info["cores_physical"] = psutil.cpu_count(logical=False)
        info["cores_logical"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        info["cores_physical"] = None
        info["cores_logical"] = os.cpu_count()
        info["ram_gb"] = None
    try:
        info["cpu_capability"] = torch.backends.cpu.get_cpu_capability()  # e.g. AVX2 / AVX512
    except Exception:
        info["cpu_capability"] = None
    info["mkldnn"] = bool(getattr(torch.backends, "mkldnn", None) and torch.backends.mkldnn.is_available())
    info["torch_threads"] = torch.get_num_threads()
    info["omp_num_threads_env"] = os.environ.get("OMP_NUM_THREADS")
    return info


def current_rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def peak_rss_bytes() -> tuple[int | None, str]:
    """Peak resident set size of this process, with the method used. None if nothing works."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        if hasattr(mi, "peak_wset"):  # Windows
            return int(mi.peak_wset), "psutil.peak_wset"
    except Exception:
        pass
    try:
        import resource  # Linux / macOS
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(ru if sys.platform == "darwin" else ru * 1024), "resource.ru_maxrss"
    except Exception:
        pass
    rss = current_rss_bytes()
    if rss is not None:
        return rss, "psutil.rss (current, not peak)"
    return None, "unavailable"


def mb(nbytes: int | float | None) -> float | None:
    return None if nbytes is None else round(nbytes / 2**20, 1)


# ----------------------------------------------------------------------------------------------
# deterministic synthetic image (no download)
# ----------------------------------------------------------------------------------------------
def synthetic_image(size: int = 256, seed: int = 0) -> Image.Image:
    """RGB gradient + seeded noise + a dark disc and a bright square, so the encoder sees some structure."""
    g = torch.Generator().manual_seed(seed)
    ys, xs = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    img = torch.stack([xs, ys, 1.0 - 0.5 * (xs + ys)], dim=-1) * 200.0
    img = img + torch.rand(size, size, 3, generator=g) * 40.0
    disc = ((xs - 0.35) ** 2 + (ys - 0.4) ** 2) < 0.03
    img[disc] = img[disc] * 0.25
    sq = (xs > 0.6) & (xs < 0.85) & (ys > 0.55) & (ys < 0.8)
    img[sq] = 255.0 - img[sq] * 0.3
    return Image.fromarray(img.clamp(0, 255).to(torch.uint8).numpy()).convert("RGB")


# ----------------------------------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------------------------------
class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t0) * 1000.0


def stats(values: Iterable[float]) -> dict | None:
    v = [float(x) for x in values if x is not None]
    if not v:
        return None
    return {
        "median": round(statistics.median(v), 2),
        "mean": round(statistics.fmean(v), 2),
        "min": round(min(v), 2),
        "max": round(max(v), 2),
        "std": round(statistics.pstdev(v), 2) if len(v) > 1 else 0.0,
        "runs": v,
    }


# ----------------------------------------------------------------------------------------------
# parameter / byte accounting
# ----------------------------------------------------------------------------------------------
TERNARY_BITS = 2  # 1.58-bit weights are stored at 2 bits/weight in practice (TQ2_0, bitnet.cpp I2_S)


def param_report(model: nn.Module, ternary_weight_ids: set[int] | None = None) -> dict:
    """
    Counts parameters (shared tensors counted once) and reports:
      bytes_as_loaded            actual bytes of the tensors in memory (dtype-dependent)
      bytes_packed_ternary       the real ternary layers at 2 bits/weight (+4 B per-tensor scale), everything
                                 else at its loaded dtype, LoRA A/B excluded (they are merged at export)
      bytes_all_linear_2bit      hypothetical: *every* nn.Linear weight (incl. lm_head) at 2 bits; rest as loaded
    """
    ternary_weight_ids = ternary_weight_ids or set()
    linear_weight_ids = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)} | ternary_weight_ids
    total = ternary_n = linear_n = 0
    b_loaded = b_packed = b_hyp = 0.0
    seen = set()
    for name, p in model.named_parameters():
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        n, b = p.numel(), p.numel() * p.element_size()
        total += n
        b_loaded += b
        is_lora = name.endswith("lora_A") or name.endswith("lora_B")
        if id(p) in ternary_weight_ids:
            ternary_n += n
            b_packed += n * TERNARY_BITS / 8 + 4
        elif not is_lora:
            b_packed += b
        if id(p) in linear_weight_ids:
            linear_n += n
            b_hyp += n * TERNARY_BITS / 8 + 4
        elif not is_lora:
            b_hyp += b
    dtypes = sorted({str(p.dtype).replace("torch.", "") for p in model.parameters()})
    return {
        "params_total": total,
        "params_ternary": ternary_n,
        "params_linear": linear_n,
        "param_dtypes": dtypes,
        "bytes_as_loaded": int(b_loaded),
        "bytes_packed_ternary": int(b_packed) if ternary_n else None,
        "bytes_all_linear_2bit": int(b_hyp),
        "ternary_bits_per_weight": TERNARY_BITS,
    }


# ----------------------------------------------------------------------------------------------
# int8 ternary matmul proxy ("pytorch-proxy, not llama.cpp")
# ----------------------------------------------------------------------------------------------
_INT_MM_CACHE: dict[tuple[int, int], bool] = {}


def int_mm_available(k: int = 64, n: int = 64) -> bool:
    """Runtime check that torch._int_mm exists and gives exact int8 x int8 -> int32 results on this device/shape."""
    key = (k, n)
    if key in _INT_MM_CACHE:
        return _INT_MM_CACHE[key]
    ok = False
    if hasattr(torch, "_int_mm"):
        try:
            for m in (1, 37):
                a = torch.randint(-128, 127, (m, k), dtype=torch.int8)
                b = torch.randint(-1, 2, (k, n), dtype=torch.int8)
                r = torch._int_mm(a, b)
                if r.dtype != torch.int32 or not torch.equal(r, a.int() @ b.int()):
                    raise RuntimeError("mismatch")
            ok = True
        except Exception:
            ok = False
    _INT_MM_CACHE[key] = ok
    return ok


class Int8TernaryLinear(nn.Module):
    """
    y = (Q_int8(x) @ W_int8^T) * (x_scale * w_scale) + b,  W_int8 in {-1,0,1}.

    Rough CPU proxy for a ternary kernel: weights are 1 byte each (not 2 bits), activations are quantized
    per token exactly like ternavlm.ternary.int8_act_quant_ste, and the integer GEMM is torch._int_mm when
    it verifies at runtime, else a float matmul of the int8 tensors. This is NOT a llama.cpp / bitnet.cpp
    number: it has no packed-weight kernels and no fused dequant.
    """

    proxy_label = "pytorch-proxy, not llama.cpp"

    def __init__(self, lin: nn.Linear, use_int_mm: bool | None = None):
        super().__init__()
        w = lin.weight.detach().float()
        scale = float(w.abs().max().item())  # merged weights are exactly {-a, 0, +a}
        if scale == 0.0:
            scale = 1.0
        w_int8 = (w / scale).round_().clamp_(-1, 1).to(torch.int8)
        self.in_features, self.out_features = lin.in_features, lin.out_features
        # stored transposed (in, out) so the GEMM is x_q @ W with no per-call transpose
        self.weight = nn.Parameter(w_int8.t().contiguous(), requires_grad=False)
        self.register_buffer("w_scale", torch.tensor(scale, dtype=torch.float32))
        self.bias = nn.Parameter(lin.bias.detach().float().clone(), requires_grad=False) if lin.bias is not None else None
        self.use_int_mm = int_mm_available(self.in_features, self.out_features) if use_int_mm is None else use_int_mm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.in_features).float()
        x_scale = x2.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) / 127.0
        xq = (x2 / x_scale).round_().clamp_(-128, 127).to(torch.int8)
        if self.use_int_mm:
            y = torch._int_mm(xq, self.weight).float()
        else:
            y = xq.float() @ self.weight.float()
        y = y * (x_scale * self.w_scale)
        if self.bias is not None:
            y = y + self.bias
        return y.to(x.dtype).reshape(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, int_mm={self.use_int_mm} ({self.proxy_label})"


@torch.no_grad()
def replace_ternary_layers(root: nn.Module, mode: str) -> tuple[int, set[int], bool]:
    """
    mode="merged":   TernaryLoRALinear -> merge() (plain nn.Linear holding the exact ternary values in float)
    mode="int8-sim": TernaryLoRALinear -> merge() -> Int8TernaryLinear
    Returns (count, ids of the ternary weight Parameters, whether torch._int_mm is used by every layer).
    """
    from ternavlm.ternary import TernaryLoRALinear  # lazy: hf baselines never import ternavlm

    count, ids, all_int_mm = 0, set(), True
    for _, module in list(root.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, TernaryLoRALinear):
                lin = child.merge()
                if mode == "int8-sim":
                    lin = Int8TernaryLinear(lin)
                    all_int_mm &= lin.use_int_mm
                elif mode != "merged":
                    raise ValueError(mode)
                setattr(module, child_name, lin)
                ids.add(id(lin.weight))
                count += 1
    return count, ids, all_int_mm


def ternary_weight_ids_qat(root: nn.Module) -> set[int]:
    from ternavlm.ternary import TernaryLoRALinear

    return {id(m.weight) for m in root.modules() if isinstance(m, TernaryLoRALinear)}


# ----------------------------------------------------------------------------------------------
# self-test (random tensors only)
# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from ternavlm.ternary import TernaryLoRALinear, ternary_quant

    torch.manual_seed(0)
    lin = nn.Linear(96, 48)
    lin.weight.data.copy_(ternary_quant(lin.weight.data))
    q = Int8TernaryLinear(lin)
    x = torch.randn(5, 96)
    # reference: int8 per-token act quant, exact ternary weights, float matmul
    xs = x.abs().amax(-1, keepdim=True) / 127.0
    ref = ((x / xs).round().clamp(-128, 127) * xs) @ lin.weight.T + lin.bias
    err = (q(x) - ref).abs().max().item()
    print(f"int_mm={q.use_int_mm} max|err| vs float reference = {err:.2e}")
    assert err < 1e-3

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = TernaryLoRALinear(nn.Linear(96, 48), r=4)
            self.other = nn.Linear(48, 8)

    t = Tiny()
    t.q_proj.lora_B.data.normal_()
    ref_out = t.q_proj(x)
    n, ids, ok = replace_ternary_layers(t, "int8-sim")
    assert n == 1 and isinstance(t.q_proj, Int8TernaryLinear)
    err2 = (t.q_proj(x) - ref_out).abs().max().item()
    print(f"replaced {n}, int_mm_all={ok}, max|err| vs QAT forward = {err2:.2e}")
    assert err2 < 1e-2
    rep = param_report(t, ids)
    print(rep)
    assert rep["params_ternary"] == 96 * 48 and rep["bytes_packed_ternary"] == int(96 * 48 * 2 / 8 + 4 + (48 + 48 * 8 + 8) * 4)
    print("peak rss:", mb(peak_rss_bytes()[0]), "MB via", peak_rss_bytes()[1])
    print("image:", synthetic_image().size, synthetic_image().mode)
    print("machine:", machine_info())
    print("selftest ok")
