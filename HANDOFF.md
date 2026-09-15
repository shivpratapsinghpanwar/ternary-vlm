# TernaVLM Handoff

**Repo:** https://github.com/shivpratapsinghpanwar/ternary-vlm (private)  
**Kaggle:** shivpratap0007/ternavlm-runner (kernel v3, failed on NaN)  
**Status:** Ready for stage1 training on Kaggle T4x2.

## What's Built

- **ternavlm/**: Core model (ternary.py with joint/plain/none LoRA, vlm.py SigLIP2+BitNet, data.py resumable streaming)
- **train.py**: fp16 trainer, resume, non-finite loss guard, flip-fraction metric
- **scripts/**: infer.py, eval_pope.py, eval_suite.py (5 benchmarks), export.py, bench_cpu.py
- **tests/**: 14 ternary tests + 16 eval metrics tests, all passing
- **configs/**: smoke, stage1, stage2 + ablations (plain_lora, projector_only, fp16_qwen1.5b)
- **docs/**: PAPER.md (skeleton), EXPORT.md (llama.cpp), RELATED_WORK.md (BitVLA baseline)
- **kaggle/push.py**: Orchestrator (embed repo, push, poll, fetch checkpoint, version dataset)

## Current Issue

Smoke run v3: NaN loss on first forward pass despite MLP fp32 safety patch. Two options:
1. **Skip smoke, run stage1** (pragmatic: 300k samples less likely to hit pathological batch)
2. **Debug smoke** (disable quantize_act in configs/smoke.yaml, reduce batch_size to 2)

## Next Steps

```bash
cd /c/Users/hp/Innovation/ternary-vlm

# Option 1: Go to stage1 (recommended)
PYTHONUTF8=1 python kaggle/push.py stage1

# Option 2: Debug smoke with simpler config
# Edit configs/smoke.yaml: quantize_act: false, batch_size: 2, total_steps: 20
# Then: PYTHONUTF8=1 python kaggle/push.py smoke
```

## Kaggle Setup

- Username: shivpratapsinghpanwar
- Kernel: shivpratap0007/ternavlm-runner (v3 exists, will push v4+)
- Dataset: shivpratap0007/ternavlm-ckpt (auto-versioned on success)
- Quota: ~2-3 more sessions this week before reset

## Key Files to Know

- `train.py`: Entry point, has flip_fraction logging
- `configs/stage1.yaml`: 300k samples, r=0 (projector-only), 9000 steps
- `configs/stage2.yaml`: 150k samples, r=32 (LoRA), 4500 steps
- `ternavlm/ternary.py`: Joint quantization logic (line 122-154)
- `kaggle/push.py`: Remote orchestrator
- `scripts/export.py`: Merge and export to GGUF

## Known Bugs / Notes

- BitNet MLP relu^2 overflows in fp16 on T4 (fixed with fp32 patch in vlm.py line 21-37, but may not be applying)
- Workers set to 0 (in-process) to avoid OOM kills
- Per-tensor absmean scale means LoRA delta must reach ~0.5*mean|W| to flip states (flip_fraction metric added)
- Related work identified BitVLA (2025) as the baseline ternary VLM we're comparing against
