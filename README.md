# TernaVLM: a 1.58-bit vision-language model that runs on a CPU

> **Ongoing research project. Paper in preparation.** This repository is public so the work can be followed as
> it happens: the training pipeline is verified on Kaggle T4x2 and stage 1 (projector) is complete and stage 2 (ternary LoRA) is running,
> but there are **no released weights, no benchmark numbers, and no claims yet**. Code, configs and docs will
> change without notice until the paper is out. Code is licensed **Apache-2.0** (see [LICENSE](LICENSE)); released
> weights, when they exist, will carry their own license. For collaboration, open an issue.

Every small VLM today (SmolVLM, Moondream, FastVLM) ships in fp16 or int4/int8. BitNet-style
**ternary** language models (weights in {-1, 0, +1}) already run fast on CPUs with integer-only
kernels. Ternary VLMs exist (LLaVaOLMoBitnet1B, BitVLA, Ternary Bonsai; see docs/RELATED_WORK.md),
and the published ones adapt the language model by training all of its latent weights (full fine-tuning or full QAT).
This repo attaches a vision encoder to a
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
- [x] stage 1 (9000 steps, loss 1.78 -> 1.02)
- [ ] stage 2 (running)
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

## Built on / please also cite

This project fine-tunes and evaluates other people's models and data; if you use it, cite them too.
`docs/RELATED_WORK.md` has the full survey with links.

- **BitNet b1.58 2B4T** (the ternary LM): Ma et al., *BitNet b1.58 2B4T Technical Report*, [arXiv:2504.12285](https://arxiv.org/abs/2504.12285);
  weights `microsoft/bitnet-b1.58-2B-4T-bf16` (MIT). The 1.58-bit scheme: Ma et al., *The Era of 1-bit LLMs*, [arXiv:2402.17764](https://arxiv.org/abs/2402.17764).
- **SigLIP 2** (vision encoder): Tschannen et al., [arXiv:2502.14786](https://arxiv.org/abs/2502.14786); weights `google/siglip2-base-patch16-256` (Apache-2.0).
- **LLaVA** recipe, projector + `<image>` splicing, and data: Liu et al., *Visual Instruction Tuning*, [arXiv:2304.08485](https://arxiv.org/abs/2304.08485);
  stage-1 data `lmms-lab/LLaVA-ReCap-558K` and stage-2 data `lmms-lab/LLaVA-OneVision-Data`, Li et al., *LLaVA-OneVision*, [arXiv:2408.03326](https://arxiv.org/abs/2408.03326).
- **Pixel shuffle** token reduction: Chen et al., *InternVL 1.5*, [arXiv:2404.16821](https://arxiv.org/abs/2404.16821).
- **LoRA**: Hu et al., [arXiv:2106.09685](https://arxiv.org/abs/2106.09685). Quantization-aware LoRA with the adapter *inside* the quantizer, which
  `TernaryLoRALinear` follows for the ternary case: **L4Q**, Jeon et al., [arXiv:2402.04902](https://arxiv.org/abs/2402.04902) and
  **LR-QAT**, Bondarenko et al., [arXiv:2406.06385](https://arxiv.org/abs/2406.06385). Straight-through estimator: Bengio et al., [arXiv:1308.3432](https://arxiv.org/abs/1308.3432).
- **Prior ternary VLMs** we compare against: **BitVLA**, Wang et al., [arXiv:2506.07530](https://arxiv.org/abs/2506.07530);
  **LLaVaOLMoBitnet1B**, Sundaram & Iyer, [arXiv:2408.13402](https://arxiv.org/abs/2408.13402).
- **Evaluation data**: POPE ([arXiv:2305.10355](https://arxiv.org/abs/2305.10355)), TextVQA ([arXiv:1904.08920](https://arxiv.org/abs/1904.08920)),
  GQA ([arXiv:1902.09506](https://arxiv.org/abs/1902.09506)), ScienceQA ([arXiv:2209.09513](https://arxiv.org/abs/2209.09513)), MME ([arXiv:2306.13394](https://arxiv.org/abs/2306.13394)), via the `lmms-lab` mirrors.
- **Inference**: [llama.cpp](https://github.com/ggml-org/llama.cpp) (GGUF export, `mtmd`), [Hugging Face transformers](https://github.com/huggingface/transformers).
- **Compute**: Kaggle's free T4x2 kernels.

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
