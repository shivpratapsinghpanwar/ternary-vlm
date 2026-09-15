"""
Reproducible CPU latency / memory benchmark for the paper table.

    # TernaVLM from a train.py checkpoint (PyTorch path)
    python scripts/bench_cpu.py --model ternavlm --ckpt ckpt/stage2/latest.pt --threads 8
    python scripts/bench_cpu.py --model ternavlm --ckpt ckpt/stage2/latest.pt --threads 8 --merge
    python scripts/bench_cpu.py --model ternavlm --ckpt ckpt/stage2/latest.pt --threads 8 --simulate-ternary-int8

    # HF baselines
    python scripts/bench_cpu.py --model hf --hf-name HuggingFaceTB/SmolVLM-256M-Instruct --threads 8
    python scripts/bench_cpu.py --model hf --hf-name HuggingFaceTB/SmolVLM-500M-Instruct --threads 8
    python scripts/bench_cpu.py --model hf --hf-name llava-hf/llava-onevision-qwen2-0.5b-ov-hf --threads 8

    # ternary decode ceiling: the BitNet LM alone, no image
    python scripts/bench_cpu.py --model bitnet-text-only --threads 8

What is measured (torch.inference_mode, greedy, fixed synthetic 256x256 image, fixed 32-token question,
1 warmup + K timed repeats, medians reported):
    vision ms     image -> vision tower (TernaVLM: SigLIP + pixel-shuffle + projector; HF: vision tower only)
    prefill ms    LM forward over [image tokens + question]; TernaVLM/BitNet measured directly,
                  HF baselines derived as (full multimodal forward) - (vision ms)
    TTFT ms       time to first token = vision + prefill (HF: measured as one forward)
    decode tok/s  32 single-token KV-cached forwards after prefill (EOS ignored; fixed length)
    peak RSS      psutil peak_wset (Windows) / resource ru_maxrss (Linux, macOS) / fallback current RSS
    params/bytes  parameter count, bytes as loaded, bytes with ternary layers packed at 2 bit/weight,
                  and the hypothetical "every nn.Linear at 2 bit" figure for the baselines

Weight paths for TernaVLM:
    default                  as trained: TernaryLoRALinear re-quantizes W (+LoRA) on every forward (slowest, exact)
    --merge                  merge() once: plain float nn.Linear holding the exact ternary values
    --simulate-ternary-int8  merge() then int8 weights + int8 activations via torch._int_mm when it verifies
                             at runtime, else float matmul of the int8 tensors. Labelled
                             "pytorch-proxy, not llama.cpp": no packed-bit kernels, only a rough proxy.

Writes results/bench_<tag>_<threads>t.json and prints one markdown table row.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time


def _preset_thread_env() -> None:
    """OMP/MKL thread counts must be set before torch is imported to take effect everywhere."""
    n = None
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            n = argv[i + 1]
        elif a.startswith("--threads="):
            n = a.split("=", 1)[1]
    if n and n.isdigit():
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, n)


_preset_thread_env()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from scripts.bench_utils import (  # noqa: E402
    Int8TernaryLinear,
    Timer,
    current_rss_bytes,
    machine_info,
    mb,
    param_report,
    peak_rss_bytes,
    replace_ternary_layers,
    stats,
    synthetic_image,
    ternary_weight_ids_qat,
)

QUESTION_BASE = (
    "Describe this image in detail. What objects, colours and shapes can you see, where are they placed, "
    "and what might the scene represent? Answer in one short paragraph please."
)
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def from_pretrained_dtype(cls, name, dtype, **kw):
    """transformers >=5 uses dtype=, 4.x uses torch_dtype=."""
    try:
        return cls.from_pretrained(name, dtype=dtype, **kw)
    except TypeError:
        return cls.from_pretrained(name, torch_dtype=dtype, **kw)


def fit_question(tokenizer, n_tokens: int) -> tuple[list[int], str]:
    """Question ids of exactly n_tokens under this tokenizer, plus the decoded text (for text-only processors)."""
    text = QUESTION_BASE
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < n_tokens:
        text = text + " " + QUESTION_BASE
        ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:n_tokens]
    return ids, tokenizer.decode(ids)


def last_logits(out) -> torch.Tensor:
    return out.logits[:, -1, :]


# ----------------------------------------------------------------------------------------------
# backends
# ----------------------------------------------------------------------------------------------
class Backend:
    """Three stages: vision() (may be None), prefill() -> (logits_last, past, attn), decode_step()."""

    name = "?"
    has_vision = True
    separable_prefill = True  # prefill() excludes the vision stage (TernaVLM / text-only)
    weights_path = "as loaded"
    image_tokens = 0
    prompt_tokens = 0
    question_tokens = 0
    ternary_weight_ids: set[int] = set()
    notes: list[str] = []
    model: nn.Module
    tokenizer = None

    def vision(self):
        return None

    def prefill(self):
        raise NotImplementedError

    def decode_step(self, tok, past, attn):
        raise NotImplementedError

    def generate_fallback(self, n_tokens: int) -> float:
        """Seconds for a full generate() of n_tokens (used only if the manual KV loop fails)."""
        raise NotImplementedError

    def lm_decode_step(self, lm, tok, past, attn):
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=1)
        out = lm(input_ids=tok, attention_mask=attn, past_key_values=past, use_cache=True)
        return last_logits(out), out.past_key_values, attn


class TernaVLMBackend(Backend):
    name = "ternavlm"

    def __init__(self, a, image):
        from scripts.infer import load_model
        from ternavlm.data import build_image_processor

        self.notes = []
        self.model = load_model(a.ckpt, "cpu")  # float32 on CPU, projector fp32
        dtype = DTYPES[a.dtype]
        if dtype != torch.float32:
            self.model.lm.to(dtype)
            self.model.vision.to(dtype)
        self.tokenizer = self.model.tokenizer
        self.lm = self.model.lm
        self._apply_weight_path(a)

        proc = build_image_processor(self.model.cfg.vision_name)
        vdtype = next(self.model.vision.parameters()).dtype
        self.pix = proc(images=image, return_tensors="pt")["pixel_values"].to(vdtype)
        self.image_tokens = self.model.num_image_tokens

        tok = self.tokenizer
        q_ids, _ = fit_question(tok, a.question_tokens)
        self.question_tokens = len(q_ids)
        img_block = self.model.cfg.image_token * self.model.num_image_tokens
        ids = [tok.bos_token_id] if tok.bos_token_id is not None else []
        ids += tok.encode(f"User: {img_block}\n", add_special_tokens=False)
        ids += q_ids
        ids += tok.encode("\nAssistant:", add_special_tokens=False)  # training format (ternavlm/data.py)
        self.ids = torch.tensor([ids])
        n_slots = int((self.ids == self.model.image_token_id).sum())
        assert n_slots == self.image_tokens, f"{n_slots} <image> slots, expected {self.image_tokens}"
        self.prompt_tokens = self.ids.shape[1]
        self.image_embeds = None

    def _apply_weight_path(self, a):
        if a.simulate_ternary_int8:
            n, ids, all_int_mm = replace_ternary_layers(self.lm, "int8-sim")
            self.ternary_weight_ids = ids
            self.weights_path = f"int8-sim ({Int8TernaryLinear.proxy_label}; torch._int_mm={'yes' if all_int_mm else 'no, float fallback'})"
            self.notes.append(f"replaced {n} TernaryLoRALinear by merge()+Int8TernaryLinear; "
                              f"int8 weights are 1 byte each in memory, not 2 bits")
        elif a.merge:
            n, ids, _ = replace_ternary_layers(self.lm, "merged")
            self.ternary_weight_ids = ids
            self.weights_path = "merged-float (exact ternary values in float nn.Linear)"
            self.notes.append(f"merged {n} TernaryLoRALinear into plain nn.Linear")
        else:
            self.ternary_weight_ids = ternary_weight_ids_qat(self.lm)
            self.weights_path = "qat-online (TernaryLoRALinear re-quantizes each forward)"
            self.notes.append("default path re-quantizes W(+LoRA) on every forward; use --merge for the export-equivalent float path")

    def vision(self):
        self.image_embeds = self.model.encode_images(self.pix)

    def prefill(self):
        if self.image_embeds is None:
            self.vision()
        embeds = self.model._splice(self.ids, self.image_embeds)
        attn = torch.ones_like(self.ids)
        out = self.lm(inputs_embeds=embeds, attention_mask=attn, use_cache=True)
        return last_logits(out), out.past_key_values, attn

    def decode_step(self, tok, past, attn):
        return self.lm_decode_step(self.lm, tok, past, attn)

    def generate_fallback(self, n_tokens):
        t0 = time.perf_counter()
        self.model.generate(self.ids, torch.ones_like(self.ids), self.pix, max_new_tokens=n_tokens,
                            min_new_tokens=n_tokens, do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id)
        return time.perf_counter() - t0


class BitNetTextBackend(Backend):
    """The ternary LM alone, loaded exactly as TernaVLM loads it (quantization_config stripped, wrap_linears r=0)."""

    name = "bitnet-text-only"
    has_vision = False

    def __init__(self, a, image=None):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from ternavlm.ternary import wrap_linears
        from ternavlm.vlm import _from_pretrained

        self.notes = []
        dtype = DTYPES[a.dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(a.lm_name)
        cfg = AutoConfig.from_pretrained(a.lm_name)
        if hasattr(cfg, "quantization_config"):
            del cfg.quantization_config
        self.lm = _from_pretrained(AutoModelForCausalLM, a.lm_name, dtype, config=cfg).eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        n = wrap_linears(self.lm, r=0, alpha=16.0, quantize_act=not a.no_act_quant)
        self.notes.append(f"wrapped {n} linears with TernaryLoRALinear(r=0)")
        self.model = self.lm
        self._apply_weight_path(a)

        tok = self.tokenizer
        q_ids, _ = fit_question(tok, a.question_tokens)
        self.question_tokens = len(q_ids)
        ids = [tok.bos_token_id] if tok.bos_token_id is not None else []
        ids += tok.encode("User: ", add_special_tokens=False) + q_ids
        ids += tok.encode("\nAssistant:", add_special_tokens=False)  # training format (ternavlm/data.py)
        self.ids = torch.tensor([ids])
        self.prompt_tokens = self.ids.shape[1]
        self.image_tokens = 0

    _apply_weight_path = TernaVLMBackend._apply_weight_path

    def prefill(self):
        attn = torch.ones_like(self.ids)
        out = self.lm(input_ids=self.ids, attention_mask=attn, use_cache=True)
        return last_logits(out), out.past_key_values, attn

    def decode_step(self, tok, past, attn):
        return self.lm_decode_step(self.lm, tok, past, attn)

    def generate_fallback(self, n_tokens):
        t0 = time.perf_counter()
        self.lm.generate(input_ids=self.ids, attention_mask=torch.ones_like(self.ids), max_new_tokens=n_tokens,
                         min_new_tokens=n_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return time.perf_counter() - t0


class HFBackend(Backend):
    """SmolVLM (Idefics3), LLaVA-OneVision, ... via AutoProcessor + AutoModelForImageTextToText."""

    name = "hf"
    separable_prefill = False

    def __init__(self, a, image):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.notes = []
        dtype = DTYPES[a.dtype]
        self.processor = AutoProcessor.from_pretrained(a.hf_name)
        self.model = from_pretrained_dtype(AutoModelForImageTextToText, a.hf_name, dtype).eval()
        self.tokenizer = getattr(self.processor, "tokenizer", None) or self.processor
        self.weights_path = f"hf as loaded ({a.dtype})"

        _, q_text = fit_question(self.tokenizer, a.question_tokens)
        self.question_tokens = len(self.tokenizer.encode(q_text, add_special_tokens=False))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q_text}]}]
        try:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception as e:  # processors without a chat template
            self.notes.append(f"chat template unavailable ({type(e).__name__}); using '<image>\\n{{question}}'")
            prompt = f"<image>\n{q_text}"
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        self.inputs = dict(inputs)
        self.prompt_tokens = int(self.inputs["input_ids"].shape[1])
        self.image_token_id = self._find_image_token_id()
        self.image_tokens = int((self.inputs["input_ids"] == self.image_token_id).sum()) if self.image_token_id is not None else None
        if self.image_token_id is None:
            self.notes.append("could not determine image token id; image_tokens=None")
        self.vision_tower = self._find_vision_tower()
        self.has_vision = self.vision_tower is not None
        if self.vision_tower is None:
            self.notes.append("no vision tower found; vision ms not measured")
        else:
            self.notes.append("vision ms = vision tower forward only (connector/projector excluded)")
        self.notes.append("prefill ms derived as (full multimodal forward) - (vision ms)")

    def _find_image_token_id(self):
        cfg = self.model.config
        for attr in ("image_token_id", "image_token_index"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int):
                return v
        tok = self.tokenizer
        for cand in (getattr(self.processor, "image_token", None), "<image>", "<|image|>", "<image_pad>"):
            if cand:
                i = tok.convert_tokens_to_ids(str(cand))
                if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
                    return i
        return None

    def _find_vision_tower(self):
        cands = [(n, m) for n, m in self.model.named_modules()
                 if n and n.split(".")[-1] in ("vision_model", "vision_tower", "vision_encoder", "visual")]
        if not cands:
            return None
        cands.sort(key=lambda nm: len(nm[0]))
        return cands[0][1]

    def _vision_pixels(self):
        pv = self.inputs["pixel_values"]
        if pv.dim() == 5:
            pv = pv.flatten(0, 1)
        # Idefics3 drops all-zero padding images before the vision tower; do the same
        keep = (pv == 0).flatten(1).all(dim=1).logical_not()
        return pv[keep] if keep.any() else pv

    def vision(self):
        self.vision_tower(pixel_values=self._vision_pixels())

    def prefill(self):
        out = self.model(**self.inputs, use_cache=True)
        return last_logits(out), out.past_key_values, self.inputs["attention_mask"]

    def decode_step(self, tok, past, attn):
        return self.lm_decode_step(self.model, tok, past, attn)

    def generate_fallback(self, n_tokens):
        t0 = time.perf_counter()
        self.model.generate(**self.inputs, max_new_tokens=n_tokens, min_new_tokens=n_tokens, do_sample=False)
        return time.perf_counter() - t0


# ----------------------------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------------------------
def run_once(backend: Backend, n_new: int, use_generate: bool) -> dict:
    r: dict = {}
    with torch.inference_mode():
        if backend.has_vision:
            with Timer() as tv:
                backend.vision()
            r["vision_ms"] = tv.ms
        else:
            r["vision_ms"] = None

        with Timer() as tp:
            logits, past, attn = backend.prefill()
        if backend.separable_prefill:
            r["prefill_ms"] = tp.ms
            r["ttft_ms"] = tp.ms + (r["vision_ms"] or 0.0)
        else:
            r["ttft_ms"] = tp.ms
            r["prefill_ms"] = tp.ms - (r["vision_ms"] or 0.0)

        if use_generate:
            total = backend.generate_fallback(n_new)
            r["decode_ms"] = total * 1000.0 - r["ttft_ms"]
            r["tokens"] = []
        else:
            tok = logits.argmax(-1, keepdim=True)
            toks = [int(tok)]
            with Timer() as td:
                for _ in range(n_new):
                    logits, past, attn = backend.decode_step(tok, past, attn)
                    tok = logits.argmax(-1, keepdim=True)
                    toks.append(int(tok))
            r["decode_ms"] = td.ms
            r["tokens"] = toks
        r["decode_tok_per_s"] = n_new / max(r["decode_ms"] / 1000.0, 1e-9)
    if isinstance(backend, TernaVLMBackend):
        backend.image_embeds = None  # force a fresh vision pass next repeat
    return r


def build_backend(a, image):
    if a.model == "ternavlm":
        if not a.ckpt:
            sys.exit("--ckpt is required for --model ternavlm")
        return TernaVLMBackend(a, image)
    if a.model == "bitnet-text-only":
        return BitNetTextBackend(a, image)
    if a.model == "hf":
        if not a.hf_name:
            sys.exit("--hf-name is required for --model hf")
        return HFBackend(a, image)
    raise ValueError(a.model)


def result_tag(a) -> str:
    if a.tag:
        return a.tag
    if a.model == "hf":
        return a.hf_name.rstrip("/").split("/")[-1]
    tag = a.model
    if a.simulate_ternary_int8:
        tag += "-int8sim"
    elif a.merge:
        tag += "-merged"
    return tag


def md_header() -> str:
    cols = ["model", "weights", "params (M)", "as-loaded MB", "packed 1.58-bit MB", "img tok", "prompt tok",
            "vision ms", "prefill ms", "TTFT ms", "decode tok/s", "peak RSS MB", "threads"]
    return "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|"


def md_row(res: dict) -> str:
    p, t = res["params"], res["timing_ms"]
    f = lambda s: "-" if s is None else f"{s['median']:.0f}"  # noqa: E731
    packed = mb(p["bytes_packed_ternary"])
    packed_s = f"{packed:.0f}" if packed is not None else f"({mb(p['bytes_all_linear_2bit']):.0f} hyp.)"
    cells = [
        res["tag"], res["weights_path"], f"{p['params_total'] / 1e6:.0f}", f"{mb(p['bytes_as_loaded']):.0f}", packed_s,
        str(res["tokens"]["image_tokens"]), str(res["tokens"]["prompt_tokens"]),
        f(t["vision"]), f(t["prefill"]), f(t["ttft"]), f"{res['decode_tok_per_s']['median']:.2f}",
        f"{res['memory']['peak_rss_mb']:.0f}" if res["memory"]["peak_rss_mb"] is not None else "-",
        str(res["threads"]),
    ]
    return "| " + " | ".join(cells) + " |"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["ternavlm", "hf", "bitnet-text-only"], required=True)
    ap.add_argument("--ckpt", help="train.py checkpoint (.pt) for --model ternavlm")
    ap.add_argument("--hf-name", help="HF repo for --model hf, e.g. HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--lm-name", default="microsoft/bitnet-b1.58-2B-4T-bf16", help="LM for --model bitnet-text-only")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (default: torch's default)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--new-tokens", type=int, default=32, help="decode steps timed after prefill")
    ap.add_argument("--question-tokens", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=list(DTYPES), default="float32", help="weights dtype as loaded (CPU default fp32)")
    ap.add_argument("--merge", action="store_true", help="ternavlm/bitnet: merge() ternary layers into float nn.Linear")
    ap.add_argument("--simulate-ternary-int8", action="store_true",
                    help="ternavlm/bitnet: merge() then int8 weights/activations via torch._int_mm (pytorch-proxy, not llama.cpp)")
    ap.add_argument("--no-act-quant", action="store_true", help="bitnet-text-only: disable int8 activation quantization")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--tag", default=None, help="override the <model> part of the output filename / table row")
    ap.add_argument("--no-header", action="store_true", help="print only the markdown row")
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    threads = torch.get_num_threads()
    torch.manual_seed(a.seed)

    image = synthetic_image(a.image_size, a.seed)
    t_load = time.perf_counter()
    backend = build_backend(a, image)
    backend.model.eval()
    load_s = time.perf_counter() - t_load
    rss_after_load = current_rss_bytes()
    params = param_report(backend.model, backend.ternary_weight_ids)
    print(f"[bench] loaded {backend.name} in {load_s:.1f}s: {params['params_total'] / 1e6:.1f}M params, "
          f"{mb(params['bytes_as_loaded'])} MB as loaded, weights: {backend.weights_path}", flush=True)

    # warmup (also decides whether the manual KV loop works for this model)
    use_generate = False
    for i in range(a.warmup):
        try:
            run_once(backend, a.new_tokens, use_generate)
        except Exception as e:
            if use_generate:
                raise
            print(f"[bench] manual KV-cache decode loop failed ({type(e).__name__}: {e}); "
                  f"falling back to generate()-derived decode timing", flush=True)
            use_generate = True
            run_once(backend, a.new_tokens, use_generate)
    runs = []
    for i in range(a.repeat):
        r = run_once(backend, a.new_tokens, use_generate)
        runs.append(r)
        print(f"[bench] run {i + 1}/{a.repeat}: vision {r['vision_ms'] and round(r['vision_ms'], 1)} ms, "
              f"prefill {r['prefill_ms']:.1f} ms, ttft {r['ttft_ms']:.1f} ms, decode {r['decode_tok_per_s']:.2f} tok/s", flush=True)

    peak, peak_method = peak_rss_bytes()
    sample = ""
    if runs[-1]["tokens"] and backend.tokenizer is not None:
        try:
            sample = backend.tokenizer.decode(runs[-1]["tokens"], skip_special_tokens=True)
        except Exception:
            sample = ""

    res = {
        "tag": result_tag(a),
        "backend": backend.name,
        "model_id": a.ckpt if a.model == "ternavlm" else (a.hf_name if a.model == "hf" else a.lm_name),
        "weights_path": backend.weights_path,
        "dtype": a.dtype,
        "threads": threads,
        "warmup": a.warmup,
        "repeat": a.repeat,
        "new_tokens": a.new_tokens,
        "image_size": a.image_size,
        "decode_method": "generate-derived" if use_generate else "manual-kv-loop",
        "prefill_method": "measured" if backend.separable_prefill else "derived (multimodal forward - vision)",
        "machine": machine_info(),
        "params": params,
        "tokens": {"image_tokens": backend.image_tokens, "prompt_tokens": backend.prompt_tokens,
                   "question_tokens": backend.question_tokens},
        "timing_ms": {
            "vision": stats(r["vision_ms"] for r in runs),
            "prefill": stats(r["prefill_ms"] for r in runs),
            "ttft": stats(r["ttft_ms"] for r in runs),
            "decode_total": stats(r["decode_ms"] for r in runs),
        },
        "decode_tok_per_s": stats(r["decode_tok_per_s"] for r in runs),
        "memory": {"peak_rss_mb": mb(peak), "peak_rss_method": peak_method, "rss_after_load_mb": mb(rss_after_load)},
        "load_seconds": round(load_s, 1),
        "sample_output": sample,
        "notes": list(backend.notes),
        "argv": sys.argv[1:],
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, f"bench_{res['tag']}_{threads}t.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    if not a.no_header:
        print(md_header())
    print(md_row(res))
    if a.simulate_ternary_int8:
        print(f"note: '{res['tag']}' int8 numbers are a {Int8TernaryLinear.proxy_label} (no packed 1.58-bit kernels)")
    print(f"[bench] sample output: {sample!r}")
    print(f"[bench] wrote {out_path}")


if __name__ == "__main__":
    main()
