"""
Resumable trainer for TernaVLM on Kaggle (fp16 + GradScaler, single GPU or torchrun DDP).

    python train.py --config configs/stage1.yaml
    python train.py --config configs/stage2.yaml --init ckpt/stage1/latest.pt

Checkpoints hold only projector + LoRA + optimizer + step + data offset, so they are small
and any 12-hour Kaggle session can resume exactly where the previous one stopped.
The trainer also stops itself cleanly before Kaggle kills the session (--time-budget-min).
"""

from __future__ import annotations

import argparse
import functools
import gc
import math
import os
import time

import torch
import yaml
from torch.utils.data import DataLoader

from ternavlm.data import DataConfig, LlavaStream, build_image_processor, collate
from ternavlm.vlm import TernaVLM, TernaVLMConfig
from ternavlm.ternary import flip_fraction


def ddp_setup():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.distributed.init_process_group("nccl")
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


def save(path, model, opt, scaler, step, seen, cfg):
    raw = model.module if hasattr(model, "module") else model
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"model": raw.trainable_state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "seen": seen, "cfg": cfg}, tmp)
    os.replace(tmp, path)


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

    m = cfg["model"]
    # Build the model one rank at a time: two ranks materialising a 2.5B model in host RAM simultaneously,
    # plus DataLoader workers, exceeded Kaggle's ~29 GB and the OOM killer took out a worker.
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

    step, seen = 0, 0
    tr = cfg["train"]
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=tr["lr"], betas=(0.9, 0.95), weight_decay=tr.get("wd", 0.0))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    if args.init:
        ck = torch.load(args.init, map_location="cpu")
        model.load_trainable_state_dict(ck["model"])
        print(f"[init] loaded trainable weights from {args.init} (step {ck.get('step')})")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        model.load_trainable_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, seen = ck["step"], ck["seen"]
        print(f"[resume] step {step}, seen {seen} samples")

    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable {n_train/1e6:.1f}M / total {n_total/1e6:.0f}M  {host_mem_gb()}", flush=True)

    d = cfg["data"]
    proc = build_image_processor(m["vision_name"])
    ds = LlavaStream(DataConfig(name=d["name"], subset=d.get("subset"), split=d.get("split", "train"),
                                max_samples=d.get("max_samples"), max_len=d.get("max_len", 512), seed=d.get("seed", 0),
                                shuffle_buffer=d.get("shuffle_buffer", 2000)),
                     model.tokenizer, proc, model.cfg.image_token, model.num_image_tokens,
                     skip=seen // (world * max(1, tr.get("workers", 2))), rank=rank, world_size=world)
    pad = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
    dl = DataLoader(ds, batch_size=tr["batch_size"], num_workers=tr.get("workers", 2),
                    collate_fn=functools.partial(collate, pad_id=pad))

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

    for i, batch in enumerate(dl):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(**batch)
            loss = out.loss / accum
        if not torch.isfinite(loss):
            bad_losses += 1
            print(f"[warn] non-finite loss at micro-step {i} (#{bad_losses}); skipping batch", flush=True)
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
            print(f"step {step}/{total} loss {running/tr.get('log_every',20):.4f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{tr.get('log_every',20)*tr['batch_size']*accum*world/dt:.1f} samp/s  elapsed {(time.time()-t0)/60:.0f}m"
                  f"  gpu {torch.cuda.max_memory_allocated()/2**30:.1f}G{extra}" if device.type == "cuda" else
                  f"step {step}/{total} loss {running/tr.get('log_every',20):.4f}{extra}", flush=True)
            running, tick = 0.0, time.time()
        if rank == 0 and step % tr.get("save_every", 500) == 0:
            save(ckpt_path, model, opt, scaler, step, seen, cfg)
        if step >= total or (time.time() - t0) > budget:
            break

    if rank == 0:
        save(ckpt_path, model, opt, scaler, step, seen, cfg)
        print(f"[done] step {step} saved to {ckpt_path}. "
              f"{'finished' if step >= total else 'time budget hit; rerun with --resume'}")
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
