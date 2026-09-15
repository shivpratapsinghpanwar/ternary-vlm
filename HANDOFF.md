# TernaVLM Handoff

**Repo:** https://github.com/shivpratapsinghpanwar/ternary-vlm (private)  
**Kaggle:** shivpratap0007/ternavlm-runner (kernel v5 = smoke passed; v6 = stage 1 session 1)  
**Status:** see "Current state" at the bottom (updated per session).

## What's Built

- **ternavlm/**: Core model (ternary.py with joint/plain/none LoRA + r=0 prequantization, vlm.py SigLIP2+BitNet
  with fp16-safe MLP and fp32 residual stream, data.py shard-local resumable loader)
- **train.py**: fp16 trainer, exact data-cursor resume, batch prefetch thread, DDP-consistent non-finite skip,
  NaN locator (prints the first module whose output is non-finite), flip-fraction metric
- **scripts/**: infer.py, eval_pope.py, eval_suite.py (5 benchmarks), export.py, bench_cpu.py
- **tests/**: 15 ternary + 6 data (offline synthetic parquet) + 16 eval-metric tests, all passing (`python -m pytest tests`)
- **configs/**: smoke, stage1, stage2 + ablations (plain_lora, projector_only, fp16_qwen1.5b), local_dryrun*
- **docs/**: PAPER.md (skeleton), EXPORT.md (llama.cpp), RELATED_WORK.md (BitVLA baseline)
- **kaggle/push.py**: Orchestrator (embed repo, push, poll, fetch checkpoint, version dataset)

## What kernels v1-v4 taught us (2026-09-15)

1. **Host OOM ~50 s into every run (v1-v3).** `datasets` streaming reads remote parquet through fsspec/pyarrow
   with up to 10 shards open in parallel; on Kaggle's network it buffered GBs and the OOM killer took out the
   streaming process (a DataLoader worker with workers=2, the main rank with workers=0). Fixed in data.py: one
   shard downloaded to `/kaggle/working/shards` at a time via `hf_hub_download`, next shard prefetched in a
   thread, `keep_shards` bounds disk, cursor `(epoch, pos, row, global)` stored in the checkpoint. v4 confirmed:
   memory flat, 1.5 s per micro-step.
2. **NaN on the first forward (v1-v4).** Two fp16 overflows: BitNet's relu^2 MLP (patch existed but was never
   called; now wired), and the residual stream itself: v4's locator reported decoder layer 7 non-finite with
   max|input| = 5.7e4 (fp16 max 65504). Fixed by upcasting the embedding output to fp32 (forward hook), so all
   residual adds are fp32 while linears/attention run fp16 under autocast. `TernaVLM.forward/generate` now
   apply autocast themselves so inference matches training numerics. Verified on CPU with a bf16 proxy.
3. Prompt format in scripts/infer.py and bench_cpu.py did not match training (`<eos>Assistant: ` vs
   `\nAssistant:`); aligned.

## Next Steps

```bash
cd /c/Users/hp/Innovation/ternary-vlm
PYTHONUTF8=1 python kaggle/push.py --status                    # kernel state
PYTHONUTF8=1 python kaggle/push.py --fetch runs/<run_dir>      # pull log + checkpoint, version ckpt dataset
PYTHONUTF8=1 python kaggle/push.py stage1                      # once smoke shows a finite, decreasing loss
```

Read a kernel log (it is a JSON array of stream chunks):
```bash
python -c "import json;[print(e['data'].rstrip()) for e in json.load(open('runs/<run_dir>/output/ternavlm-runner.log',encoding='utf-8')) if 'step ' in e['data'] or '[warn]' in e['data'] or '[data]' in e['data']]"
```

Watch for in the smoke log: `[warn] ... non-finite` (should be absent), `step N/100 loss ...` decreasing from ~14,
`samp/s` (decides the stage-1 sample budget: stage1.yaml assumes >= 7 samp/s for 300k samples in ~12 h;
lower it or accept more sessions if v5 is slower), `gpu xxG` < 14 G.

## Kaggle Setup

- Kernel: shivpratap0007/ternavlm-runner; dataset: shivpratap0007/ternavlm-ckpt (auto-versioned on success)
- Shards land in /kaggle/working/shards and are deleted before the kernel ends (never uploaded)
- Quota: ~30 GPU-h/week; each smoke run costs ~25 min of T4x2

## Local machine caveats

- 4 GB RAM: the 2B BitNet cannot be loaded locally. Use `configs/local_dryrun.yaml` (SmolLM2-135M stand-in);
  set `TERNAVLM_SHARD_DIR` to a scratch dir (one ReCap shard is ~500 MB).
- `HF_HUB_DISABLE_XET=1`; Python needs `C:/...` paths (Git-Bash `/c/...` paths fail inside Python).

## Current state

- 2026-09-15: kernel v5 smoke PASSED: 100 steps, no non-finite loss, loss 1.86 -> ~1.3-1.5, gpu 7.1 G,
  rss flat at 4.4 G, 1.7 samp/s (r=8 joint LoRA, batch 4 x 2 GPUs), flip fraction 0.35-0.43 at lr 5e-4
  (worth a look for the paper: tiny deltas, drel ~1e-4, already flip a third of the ternary states).
  Checkpoint versioned to shivpratap0007/ternavlm-ckpt as smoke/latest.pt (run dir runs/20260915T085432Z_smoke).
- 2026-09-15: kernel v6 = stage 1, session 1, launched with `push.py stage1 --no-wait` (680 min budget).
  When it completes: `push.py --fetch runs/20260915T091912Z_stage1` versions stage1/latest.pt, then rerun
  `push.py stage1` to resume until the log says "finished". FIRST THING TO CHECK in its log: `samp/s` at
  step 20-100. stage1.yaml (9000 steps x 32 = 288k samples) needs ~7 samp/s to finish in one 11 h session;
  at the smoke's 1.7 samp/s it would take ~4 sessions. If it is below ~4 samp/s, cut `max_samples`/`total_steps`
  (e.g. 150k / 4500) or reduce `max_len` before session 2 (the cursor + LR schedule resume cleanly either way).
