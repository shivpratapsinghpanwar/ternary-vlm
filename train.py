"""
Resumable trainer for TernaVLM on Kaggle (fp16 + GradScaler, single GPU or torchrun DDP).

    python train.py --config configs/stage1.yaml
    python train.py --config configs/stage2.yaml --init ckpt/stage1/latest.pt

Checkpoints hold only projector + LoRA + optimizer + step + data cursor, so they are small
and any 12-hour Kaggle session can resume exactly where the previous one stopped.
The trainer also stops itself cleanly before Kaggle kills the session (--time-budget-min).
"""

from __future__ import annotations

import argparse
import functools
import gc
import math
import os
import queue
import threading
import time

import torch
import yaml
from torch.utils.data import DataLoader

from ternavlm.data import DataConfig, LlavaStream, build_image_processor, collate
from ternavlm.vlm import TernaVLM, TernaVLMConfig
from ternavlm.ternary import flip_fraction, transition_report


def ddp_setup():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.distributed.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return rank, int(os.environ["WORLD_SIZE"])
    return 0, 1


def lr_at(step, total, warmup, peak, floor=0.0):
    if step < warmup:
        return peak * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(1.0, p)))


def host_mem_gb() -> str:
    try:
        rss = [l for l in open("/proc/self/status") if l.startswith("VmRSS")][0].split()[1]
        avail = [l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]
        return f"rss {int(rss)/2**20:.1f}G avail {int(avail)/2**20:.1f}G"
    except Exception:
        return "rss n/a"


def save(path, model, opt, scaler, step, seen, data_state, cfg, elapsed_min=None):
    raw = model.module if hasattr(model, "module") else model
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"model": raw.trainable_state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "seen": seen, "data": data_state, "cfg": cfg}, tmp)
    os.replace(tmp, path)
    # ternary-transition diagnostics for the paper (LoRA layers only; a no-op in projector-only stages)
    rep = transition_report(raw.lm)
    if rep.get("n_layers"):
        import json
        with open(os.path.join(os.path.dirname(path), "diag.jsonl"), "a") as f:
            f.write(json.dumps({"step": step, "seen": seen, "elapsed_min": elapsed_min, **rep}) + "
")
        print(f"[diag] step {step} flip {rep['flip_frac']:.3e} to0 {rep['to_zero']:.2e} from0 {rep['from_zero']:.2e} "
              f"sign {rep['sign_flip']:.1e} drel {rep['delta_rel']:.2e} scale {rep['scale_ratio']:.4f} "
              f"boundary {rep['boundary_mass']:.3f} zero {rep['zero_frac0']:.3f}->{rep['zero_frac1']:.3f}", flush=True)


class Prefetch:
    """Iterate a DataLoader in a background thread so image decoding overlaps the GPU step (the stream itself
    must stay in-process for exact cursor checkpointing, so DataLoader workers are not used)."""

    def __init__(self, dl, depth: int = 3):
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.t = threading.Thread(target=self._run, args=(dl,), daemon=True, name="batch-prefetch")
        self.t.start()

    def _run(self, dl):
        try:
            for b in dl:
                self.q.put(b)
            self.q.put(None)
        except BaseException as e:  # surface in the main thread
            self.q.put(e)

    def __iter__(self):
        while True:
            b = self.q.get()
            if b is None:
                return
            if isinstance(b, BaseException):
                raise b
            yield b


@torch.no_grad()
def locate_nonfinite(model, batch, device) -> str:
    """Re-run the forward with hooks and report the first module whose output is non-finite (once, for the log)."""
    raw = model.module if hasattr(model, "module") else model
    found = []

    def hook(name):
        def f(mod, inp, out):
            if found:
                return
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
                xin = inp[0] if inp and torch.is_tensor(inp[0]) else None
                xmax = f"{xin.float().abs().max().item():.3g}" if xin is not None else "?"
                found.append(f"{name} ({type(mod).__name__}) out={tuple(t.shape)} dtype={t.dtype} max|in|={xmax}")
        return f

    hs = [m.register_forward_hook(hook(n)) for n, m in raw.named_modules() if n]
    try:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            raw(**{k: v for k, v in batch.items() if k != "labels"})
    finally:
        for h in hs:
            h.remove()
    return found[0] if found else "no non-finite module output found (loss itself overflowed?)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="checkpoint to continue (same stage)")
    ap.add_argument("--init", default=None, help="trainable weights to start from (previous stage)")
    ap.add_argument("--time-budget-min", type=float, default=680, help="stop cleanly after this many minutes")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    t0 = time.time()

    rank, world = ddp_setup()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        # fp16 GEMMs must accumulate in fp32 (split-K reductions in fp16 can overflow on T4)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    m = cfg["model"]
    # Build the model one rank at a time: two ranks materialising a 2.5B model in host RAM simultaneously
    # exceeded Kaggle's ~29 GB.
    for r in range(world):
        if world > 1:
            torch.distributed.barrier()
        if r != rank:
            continue
        model = TernaVLM(TernaVLMConfig(
            lm_name=m["lm_name"], vision_name=m["vision_name"], pixel_shuffle=m.get("pixel_shuffle", 2),
            projector_hidden=m.get("projector_hidden", 2048), lora_r=m.get("lora_r", 0),
            lora_alpha=m.get("lora_alpha", 16.0), lora_mode=m.get("lora_mode", "joint"),
            quantize_act=m.get("quantize_act", True), quantize_lm=m.get("quantize_lm", True),
            prequantize_frozen=m.get("prequantize_frozen", True), fp32_residual=m.get("fp32_residual", True),
            freeze_vision=m.get("freeze_vision", True),
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,  # fp16 matmuls on CPU are very slow
        ))
        if cfg["train"].get("grad_checkpointing", True):
            model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.lm.enable_input_require_grads()
        model.to(device)
        gc.collect()
    if world > 1:
        torch.distributed.barrier()

    step, seen, data_state = 0, 0, None
    tr = cfg["train"]
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=tr["lr"], betas=(0.9, 0.95), weight_decay=tr.get("wd", 0.0))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if args.init:
        ck = torch.load(args.init, map_location="cpu")
        model.load_trainable_state_dict(ck["model"])
        print(f"[init] loaded trainable weights from {args.init} (step {ck.get('step')})")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        model.load_trainable_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, seen, data_state = ck["step"], ck["seen"], ck.get("data")
        if data_state is None:  # checkpoint from the old streaming reader: restart the data from the top
            print("[resume] checkpoint has no data cursor; data restarts from the beginning")
        print(f"[resume] step {step}, seen {seen} samples, data cursor {data_state}")

    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable {n_train/1e6:.1f}M / total {n_total/1e6:.0f}M  {host_mem_gb()}", flush=True)

    d = cfg["data"]
    proc = build_image_processor(m["vision_name"])
    if data_state is not None and world > 1:
        # rank 0 saved its cursor; start every rank on a fresh stripe (global row multiple of world)
        g = -(-data_state["global"] // world) * world
        data_state = {**data_state, "row": data_state["row"] + (g - data_state["global"]), "global": g}
    ds = LlavaStream(DataConfig(name=d["name"], subset=d.get("subset"), split=d.get("split", "train"),
                                max_samples=d.get("max_samples"), max_len=d.get("max_len", 512), seed=d.get("seed", 0),
                                shuffle_buffer=d.get("shuffle_buffer", 2000), shard_dir=d.get("shard_dir"),
                                keep_shards=d.get("keep_shards", 2)),
                     model.tokenizer, proc, model.cfg.image_token, model.num_image_tokens,
                     rank=rank, world_size=world, state=data_state)
    if rank == 0:
        print(f"[data] {len(ds.files)} shards of {d['name']} ({d.get('subset') or 'default'}), cursor {ds.state}", flush=True)
    pad = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
    if tr.get("workers", 0):
        print("[data] workers>0 ignored: the shard stream runs in-process (a prefetch thread overlaps decoding)")
    dl = DataLoader(ds, batch_size=tr["batch_size"], num_workers=0, collate_fn=functools.partial(collate, pad_id=pad))

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])

    accum = tr.get("grad_accum", 1)
    total = tr["total_steps"]
    ckpt_path = os.path.join(tr["out_dir"], "latest.pt")
    budget = args.time_budget_min * 60
    model.train()
    opt.zero_grad(set_to_none=True)
    running, tick = 0.0, time.time()
    bad_losses = 0
    last_state = ds.state_dict()
    i = -1

    for i, batch in enumerate(Prefetch(dl)):
        last_state = batch.pop("data_state", last_state)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(**batch)
            loss = out.loss / accum
        # every rank must take the same branch (backward is a collective under DDP)
        bad = torch.tensor([0.0 if torch.isfinite(loss) else 1.0], device=device)
        if world > 1:
            torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
        if bad.item() > 0:
            bad_losses += 1
            where = locate_nonfinite(model, batch, device) if bad_losses == 1 and not torch.isfinite(loss) else ""
            print(f"[warn] rank {rank} non-finite loss at micro-step {i} (#{bad_losses}); skipping batch {where}", flush=True)
            del out, loss
            if bad_losses >= 20:
                raise RuntimeError("20 non-finite losses: forward is overflowing, aborting")
            continue
        scaler.scale(loss).backward()
        running += loss.item()
        seen += batch["input_ids"].shape[0] * world
        if (i + 1) % accum != 0:
            continue

        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, tr.get("warmup", 100), tr["lr"])
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, tr.get("clip", 1.0))
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if rank == 0 and step % tr.get("log_every", 20) == 0:
            dt = time.time() - tick
            extra = ""
            if step % (tr.get("log_every", 20) * 5) == 0:
                raw = model.module if hasattr(model, "module") else model
                ff = flip_fraction(raw.lm)
                extra = f"  flip {ff['flip_frac']:.2e} drel {ff.get('delta_rel', 0):.2e}  {host_mem_gb()}"
            gpu = f"  gpu {torch.cuda.max_memory_allocated()/2**30:.1f}G" if device.type == "cuda" else ""
            print(f"step {step}/{total} loss {running/tr.get('log_every',20):.4f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{tr.get('log_every',20)*tr['batch_size']*accum*world/dt:.1f} samp/s  elapsed {(time.time()-t0)/60:.0f}m"
                  f"{gpu}{extra}", flush=True)
            running, tick = 0.0, time.time()
        if rank == 0 and step % tr.get("save_every", 500) == 0:
            save(ckpt_path, model, opt, scaler, step, seen, last_state, cfg, (time.time() - t0) / 60)
        if step >= total or (time.time() - t0) > budget:
            break
    else:
        if rank == 0:
            print(f"[data] stream exhausted after micro-step {i} (max_samples reached)")

    if rank == 0:
        save(ckpt_path, model, opt, scaler, step, seen, last_state, cfg, (time.time() - t0) / 60)
        print(f"[done] step {step} saved to {ckpt_path}. "
              f"{'finished' if step >= total else 'time budget hit; rerun with --resume'}")
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
