# TernaVLM: a 1.58-bit vision-language model that runs on a CPU

> **Ongoing research project. Paper in preparation.** This repository is public so the work can be followed as
> it happens: the training pipeline is verified (finite, decreasing loss on Kaggle T4x2) and stage 1 is running,
> but there are **no released weights, no benchmark numbers, and no claims yet**. Code, configs and docs will
> change without notice until the paper is out. Code is licensed **Apache-2.0** (see [LICENSE](LICENSE)); released
> weights, when they exist, will carry their own license. For collaboration, open an issue.

Every small VLM today (SmolVLM, Moondream, FastVLM) ships in fp16 or int4/int8. BitNet-style
**ternary** language models (weights in {-1, 0, +1}) already run fast on CPUs with integer-only
kernels. Ternary VLMs exist (LLaVaOLMoBitnet1B, BitVLA, Ternary Bonsai; see docs/RELATED_WORK.md),
but all of them update the full set of latent weights with QAT. This repo attaches a vision encoder to a
pre-trained ternary LM and adapts it **parameter-efficiently, without leaving the ternary regime**: only a
projector and LoRA adapters are trained, and the exported model is bit-for-bit the ternary model that was trained.

## Why this is not "just LoRA on BitNet"

Plain LoRA computes `y = x·Q(W)ᵀ + s·x·(BA)ᵀ`. The delta `BA` is full precision, so the deployed
model is no longer ternary, and merging then re-quantizing gives a train/test mismatch.

`ternavlm/ternary.py` instead computes

```
y = Q_int8(x) · Q_ternary(W + s·BA)ᵀ        (straight-through estimator for the gradient)
```

so `merge()` is exact: the exported weights are precisely the ternary tensors the model was
trained with. `tests/test_ternary.py` checks this bit-for-bit.

## Architecture

```
image ─► SigLIP2-base (256px, frozen) ─► 256 patches ─► pixel-shuffle 2x2 ─► 64 tokens
      ─► MLP projector (trained) ─► spliced into <image> slots ─► BitNet b1.58 2B (ternary, LoRA-QAT)
```

* LM: `microsoft/bitnet-b1.58-2B-4T-bf16` (master weights, made for fine-tuning).
  Alternative: `tiiuae/Falcon-E-1B-Instruct` for an even smaller model.
* 64 image tokens keeps both T4 training and CPU inference cheap.

## Training on Kaggle (2x T4, 12 h sessions, ~30 GPU-h/week)

| Stage | Trains | Data | GPU-h | Sessions |
|---|---|---|---|---|
| smoke | projector + tiny LoRA | 2k samples | 0.5 | 1 |
| stage1 | projector only | LLaVA-ReCap-558K (300k) | 10-12 | 1-2 |
| stage2 | projector + ternary-LoRA r=32 | LLaVA-OneVision subsets (150k) | 20-25 | 2-3 |
| export | merge, GGUF, CPU benchmark | - | CPU | 1 |

Checkpoints contain only projector + LoRA + optimizer state (a few hundred MB) and are versioned into a
private Kaggle dataset after every session. `train.py --resume` continues from the exact data cursor
(epoch, shard, row). The trainer stops itself before the 12 h limit so Kaggle never kills it mid-write.

```bash
python -m pytest tests -q                      # 38 tests, CPU only, no downloads

# Kaggle (private kernel; needs Kaggle CLI credentials), see kaggle/push.py for the design
python kaggle/push.py smoke                    # ~25 min: proves the pipeline
python kaggle/push.py stage1                   # rerun each session until the log says "finished"
python kaggle/push.py stage2

# a single local GPU (>= 10 GB), same checkpoints
python train.py --config configs/stage1.yaml --resume ckpt/stage1/latest.pt

# export
python scripts/export.py --ckpt ckpt/stage2/latest.pt --out export/ternavlm
```

## Engineering notes that cost GPU hours to learn

- The HF BitNet class quantizes inside its own linear layers; we strip `quantization_config` so the model loads
  with plain `nn.Linear` holding the latent bf16 weights and `TernaryLoRALinear` is the only quantizer.
- BitNet's hidden states reach |h| ~ 6e4 and its relu² MLP overflows fp16 on T4: the residual stream is kept in
  fp32 (embedding output upcast) and the MLP product is computed in fp32; linears run fp16 under autocast.
- `datasets` streaming of remote parquet got OOM-killed on Kaggle; `ternavlm/data.py` downloads one shard at a
  time instead (rank 0 only) with a checkpointed cursor.
- Smoke-run numbers (2x T4, batch 4, r=8): 7.1 GB GPU, 1.7 samples/s.

## Roadmap

- [x] smoke run on Kaggle (finite, decreasing loss)
- [ ] stage 1 (running) + stage 2
- [ ] ablations: plain fp16 LoRA (merge-and-requantize), projector-only, fp16 Qwen2.5-1.5B control
- [ ] export to GGUF (TQ2_0) and run with llama.cpp `mtmd` on a laptop CPU and a Raspberry Pi 5
- [ ] benchmark table: tokens/s and memory vs SmolVLM-256M / Moondream on the same CPU
- [ ] eval: TextVQA, GQA, POPE subset
- [ ] ternary Falcon-E-1B variant

## Layout

```
ternavlm/ternary.py   ternary + int8 quantizers, TernaryLoRALinear, wrap/merge helpers
ternavlm/vlm.py       vision encoder + projector + LM glue, small trainable checkpoints
ternavlm/data.py      shard-local LLaVA-format data (one parquet shard on disk at a time), exact resume, label masking
train.py              fp16 trainer, resume, time budget, torchrun DDP
kaggle/push.py        local orchestrator: embed repo in a private Kaggle kernel, poll, fetch, version checkpoints
scripts/              infer.py, eval_pope.py, eval_suite.py, export.py (exact merge -> GGUF), bench_cpu.py
configs/*.yaml        smoke / stage1 / stage2 / ablations / local CPU dry runs
docs/                 PAPER.md (draft skeleton), RELATED_WORK.md (novelty assessment), EXPORT.md
HANDOFF.md            current status and per-machine instructions
```

## Citation

A paper is in preparation. Until then, please cite this repository:

```
@misc{ternavlm2026,
  title  = {TernaVLM: parameter-efficient vision adaptation of a natively ternary language model with exact ternary export},
  author = {Shiv Pratap Singh Panwar},
  year   = {2026},
  url    = {https://github.com/shivpratapsinghpanwar/ternary-vlm},
  note   = {Work in progress}
}
```
