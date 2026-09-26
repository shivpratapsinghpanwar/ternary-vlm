# TernaVLM: a 1.58-bit vision-language model that runs on a CPU

> Status: training. The pipeline is verified on Kaggle T4x2 (smoke run: finite, decreasing loss); stage 1 is running.

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

Checkpoints contain only projector + LoRA + optimizer state (a few hundred MB) and are pushed to a
private Hugging Face model repo after every session. `train.py --resume` continues from the exact
data cursor (epoch, shard, row). The trainer stops itself at 11h20m so Kaggle never kills it mid-write.

```bash
# local unit tests (CPU, no transformers needed)
python tests/test_ternary.py

# on Kaggle: edit REPO / HF_REPO in kaggle/run_stage.py, add HF_TOKEN secret, then
STAGE=smoke  python kaggle/run_stage.py
STAGE=stage1 python kaggle/run_stage.py     # rerun each session until "finished"
STAGE=stage2 python kaggle/run_stage.py

# export
python scripts/export.py --ckpt ckpt/stage2/latest.pt --out export/ternavlm
```

## First-session checklist (things that must be verified before burning GPU hours)

1. `AutoModelForCausalLM.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")` loads, and the
   number of layers replaced by `wrap_linears` is `7 x num_layers`. If the HF class already
   quantizes inside its own linear layers, the replacement must remove that path (no double quantization).
2. Check the model's chat template and switch `data.py` from the `User:/Assistant:` format to
   `tokenizer.apply_chat_template` if one exists.
3. Confirm fp16 does not overflow: watch for `nan` loss in the smoke run. If it does, keep
   the projector in fp32 and autocast only the LM.
4. Throughput: expect ~8-12 samples/s on 2x T4 for stage 1. If far below, drop `max_len` or batch size.

## Roadmap

- [ ] smoke run on Kaggle
- [ ] stage 1 + stage 2
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
kaggle/run_stage.py   one-file Kaggle session driver with Hub checkpoint sync
scripts/export.py     merge LoRA exactly, save HF checkpoints for GGUF conversion
configs/*.yaml        smoke / stage1 / stage2
```
