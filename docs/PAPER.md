# TernaVLM — workshop paper skeleton

> Working draft. Every number in square brackets (`[XX.X]`, `[N]`, `[verify]`) is a placeholder to be
> filled from `runs/` after the corresponding experiment has completed. Claims marked *expected* are
> hypotheses, not results, and must be rewritten or removed once data exists. Target venue: a
> NeurIPS/ICLR workshop on efficient ML or multimodal models (4 pages + references + appendix), or arXiv.

---

## Title candidates

1. **TernaVLM: A Vision-Language Model on a 1.58-bit Language Model via Ternary-Consistent LoRA**
2. **Fine-Tuning Without Leaving the Ternary Regime: Vision-Language Adaptation of BitNet b1.58**
3. **Quantize Jointly, Merge Exactly: Ternary-Consistent Low-Rank Adaptation for CPU-Deployable Vision-Language Models**

---

## Abstract (draft, ~150 words)

Small vision-language models (VLMs) are shipped in 16-bit or 4–8-bit weights, while ternary
language models such as BitNet b1.58 already run with integer-only kernels on commodity CPUs. We
present TernaVLM, to our knowledge the first VLM built on a pre-trained ternary (1.58-bit) LLM. A
frozen SigLIP2 encoder produces 64 image tokens (2×2 pixel shuffle) that an MLP projector maps into
BitNet b1.58 2B-4T. The language model is adapted with *ternary-consistent LoRA*: the low-rank
update is added to the latent weight and the sum is re-quantized in every forward pass,
$Q(W + sBA)$, trained with a straight-through estimator. The merged model is therefore exactly
ternary and identical to the trained model, whereas merging a conventional LoRA delta into a ternary
base discards most of the update. Trained in [XX] GPU-hours on free-tier Kaggle T4s, TernaVLM reaches
[XX.X] POPE F1, [XX.X] GQA and [XX.X] ScienceQA-IMG, within [X.X] points of an fp16 control of similar
size, with a [X.X]× smaller LM footprint and [X.X]× faster CPU decoding than [baseline]. We also show
that likelihood scoring of candidate answers ("no-decode" inference) recovers [XX.X]% of generative
accuracy at [X.X]× lower CPU latency.

---

## 1 Introduction

Deploying a VLM on a laptop, phone, or single-board computer is bounded by weight memory bandwidth
during autoregressive decoding. Current "small" VLMs (SmolVLM-256M/500M, Moondream, FastVLM,
LLaVA-OneVision-0.5B) address this by shrinking parameter counts and image-token counts while
keeping fp16/bf16 or post-training int4/int8 weights. Ternary LLMs offer a complementary axis:
BitNet b1.58 2B-4T stores weights in $\{-1, 0, +1\}$ with one scale per tensor, and inference
libraries (bitnet.cpp, llama.cpp `TQ2_0`) execute them with integer-only kernels at a memory cost of
roughly 2 bits per weight. No ternary VLM has been released; the obstacle is not architecture but
adaptation: the standard LLaVA recipe fine-tunes the LLM with full-precision LoRA, whose delta cannot
be represented on the ternary grid, so the deployed model is either no longer ternary or no longer
the model that was trained.

We make the following contributions.

1. **Ternary-consistent LoRA.** We fine-tune a ternary LLM by quantizing the *sum* of the latent
   base weight and the LoRA update in every forward pass, $Q_t(W + sBA)$, with a straight-through
   estimator. We give the forward and gradient expressions, prove that merging is exact (the exported
   ternary tensors are bit-identical to those used during training), and characterize why merging a
   conventional LoRA delta into a ternary base is inconsistent: with a per-tensor absmean grid, any
   delta entry smaller than half the grid step is discarded (Section 2.2).
2. **TernaVLM**, the first (to our knowledge) vision-language model whose language backbone is a
   pre-trained 1.58-bit LLM (BitNet b1.58 2B-4T), with a SigLIP2-base encoder, 2×2 pixel shuffle to
   64 image tokens, and an MLP projector. The full two-stage recipe (projector alignment on ~300k
   captions, then LoRA r=32 on ~150k instruction samples) fits in ~[35] GPU-hours on Kaggle's free
   2×T4 tier; we report benchmarks on POPE, TextVQA, GQA, ScienceQA-IMG and MME against
   SmolVLM-256M/500M, LLaVA-OneVision-0.5B and an fp16 control backbone (Qwen2.5-1.5B) trained with
   the identical recipe (Section 3).
3. **A no-decode inference mode and CPU measurements.** For closed-set tasks we replace
   autoregressive generation with batched likelihood scoring of candidate answers, which turns a
   memory-bound sequential decode into a single compute-bound forward pass; we quantify the
   accuracy/latency trade-off, and report memory and tokens/s on a laptop CPU and a Raspberry Pi 5
   with the same runtime for all models (Sections 2.3, 3.4).

---

## 2 Method

### 2.1 Architecture

TernaVLM follows the LLaVA layout (Figure 1): a vision encoder, a projector, and a decoder-only LM
into whose input embedding sequence the projected image tokens are spliced at the positions of a
placeholder token `<image>`.

- **Vision encoder.** `google/siglip2-base-patch16-256`, frozen. A 256×256 image yields a
  $16 \times 16$ grid of 256 patch embeddings of width $C = 768$.
- **Pixel shuffle.** A $2 \times 2$ space-to-depth rearrangement (as in InternVL-1.5 / Idefics3)
  reduces 256 tokens to 64 tokens of width $4C = 3072$. The encoder's last hidden state is used
  without the pooling head.
- **Projector.** LayerNorm → Linear($3072 \to 2048$) → GELU → Linear($2048 \to d_{\mathrm{LM}}$),
  where $d_{\mathrm{LM}} = 2560$ for BitNet b1.58 2B-4T; $\approx$ [11.5]M parameters, kept in fp32.
- **Language model.** `microsoft/bitnet-b1.58-2B-4T-bf16` (the latent-weight checkpoint intended for
  further training), [30] layers, [2.4]B parameters [verify]. The checkpoint's on-line quantizing
  linear layers are removed and replaced by our `TernaryLoRALinear` on the seven projections of every
  block (q, k, v, o, gate, up, down; $7 \times [30] = [210]$ modules), so that exactly one quantizer is
  applied per linear layer. Embeddings, the LM head, and normalization layers keep their original
  precision, as in the base model.
- **Sequence.** One image = 64 tokens. With `max_len` 256 (stage 1) / 512 (stage 2), the image occupies
  25% / 12.5% of the context.

### 2.2 Ternary-consistent LoRA

**Quantizers.** Following BitNet b1.58, weights use per-tensor absmean ternary quantization and
activations per-token absmax int8 quantization:

$$
Q_t(W) = \gamma \, \mathrm{clip}\!\left(\mathrm{round}(W/\gamma), -1, 1\right), \qquad
\gamma = \mathrm{mean}\,|W| + \epsilon,
$$

$$
Q_a(x) = \eta \, \mathrm{clip}\!\left(\mathrm{round}(x/\eta), -128, 127\right), \qquad
\eta = \max_j |x_j| / 127 \quad \text{(per token)}.
$$

The rounding threshold of $Q_t$ is $\gamma/2$: an entry of $W$ maps to $0$ if $|W_{ij}| < \gamma/2$ and
to $\pm\gamma$ otherwise. Statistics are computed in fp32 even under fp16 autocast.

**Forward.** Let $W \in \mathbb{R}^{d_o \times d_i}$ be the frozen latent weight,
$A \in \mathbb{R}^{r \times d_i}$, $B \in \mathbb{R}^{d_o \times r}$ trainable, $s = \alpha / r$, and
$W' = W + sBA$. A TernaryLoRALinear computes

$$
y = Q_a(x)\, Q_t(W')^{\top} + b, \qquad \hat W := Q_t(W') = Q_t(W + sBA).
$$

$B$ is zero-initialized, so at step 0 $\hat W = Q_t(W)$ and the model is exactly the released ternary
base model. The LoRA path is *not* a separate branch: the delta is fused into the weight before
quantization and the layer performs a single ternary matmul, exactly as at deployment.

**Straight-through gradient.** We use the identity STE for both quantizers,
$\partial Q_t(W')/\partial W' := I$ and $\partial Q_a(x)/\partial x := I$ (implemented as
$W' + \mathrm{sg}[Q_t(W') - W']$). Let $\delta = \partial \mathcal{L}/\partial y \in \mathbb{R}^{N \times d_o}$
over the $N$ tokens of a batch and $\hat x = Q_a(x) \in \mathbb{R}^{N \times d_i}$. Then

$$
G := \frac{\partial \mathcal{L}}{\partial \hat W} = \delta^{\top} \hat x \in \mathbb{R}^{d_o \times d_i},
\qquad
\frac{\partial \mathcal{L}}{\partial W'} \overset{\text{STE}}{=} G,
$$

$$
\frac{\partial \mathcal{L}}{\partial B} = s\, G A^{\top} \in \mathbb{R}^{d_o \times r},
\qquad
\frac{\partial \mathcal{L}}{\partial A} = s\, B^{\top} G \in \mathbb{R}^{r \times d_i}.
$$

The gradient with respect to $A, B$ is the same as for an unquantized layer, but the *loss* is
evaluated on the ternary grid. $A$ and $B$ therefore move continuously in the latent space and change
the deployed function only when an entry of $W'/\gamma$ crosses a rounding threshold $\pm 1/2$
(or when $\gamma$ itself shifts). Note that $G$ is never materialized as a dense $d_o \times d_i$
tensor in the LoRA branch alone; autograd forms $GA^\top$ and $B^\top G$ through the fused weight, so
peak memory is one fp32 copy of $W'$ per layer per micro-step (released under gradient
checkpointing).

**Proposition 1 (exact merge).** Let $\hat W = Q_t(W + sBA)$ be computed once after training with the
same $W, A, B, s, \epsilon$ and the same quantizer implementation. Define
$f_{\mathrm{train}}(x) = Q_a(x)\,\big(W' + \mathrm{sg}[Q_t(W') - W']\big)^{\top} + b$ and
$f_{\mathrm{export}}(x) = Q_a(x)\,\hat W^{\top} + b$. Then $f_{\mathrm{train}}(x) = f_{\mathrm{export}}(x)$
for all $x$, and $\hat W = \gamma T$ with $T \in \{-1, 0, +1\}^{d_o \times d_i}$ and a single scalar
$\gamma$, i.e. the exported layer is in the same 2-bit-plus-scale format as the base model.

*Proof.* The forward value of $W' + \mathrm{sg}[Q_t(W') - W']$ is $Q_t(W')$ in exact arithmetic; the
stop-gradient affects only the backward pass. Hence the two maps coincide pointwise. The second
claim is the definition of $Q_t$. In IEEE arithmetic the training-time expression $w + (q - w)$ can
differ from $q$ by one ulp of the master dtype; the export path (`merge()`) stores $q$ directly, and
`tests/test_ternary.py` asserts equality between the exported tensor and $Q_t(W')$ bit-for-bit. $\square$

*Remark 1 (rank of the realized update).* Although $sBA$ has rank $\le r$, the realized change on the
ternary grid, $\hat W - Q_t(W) \in \{-2\gamma', -\gamma', 0, \gamma', 2\gamma'\}$-valued (up to the scale
change $\gamma \to \gamma'$), is a sparse sign-flip pattern with no rank constraint. The low-rank
parameterization is a low-dimensional *search direction* in latent space, not a constraint on the
deployed weights. We report the fraction of flipped entries per layer in Table 2 / Figure 2.

**Why merging a conventional LoRA is inconsistent.** Plain LoRA on a ternary base computes

$$
g_{\mathrm{train}}(x) = Q_a(x)\, Q_t(W)^{\top} + s\, x\,(BA)^{\top} + b,
$$

with a full-precision delta on a separate branch. Three export options exist, none of which
preserves $g_{\mathrm{train}}$ inside a ternary model:

- (i) *Keep the delta in fp16.* Deployable but not ternary: each layer gains $2r(d_i + d_o)$ fp16
  weights and a second, floating-point matmul, breaking integer-only kernels.
- (ii) *Merge into the quantized base and re-quantize*, $\hat W_{\mathrm{ii}} = Q_t(Q_t(W) + sBA)$.
- (iii) *Merge into the latent base and re-quantize*, $\hat W_{\mathrm{iii}} = Q_t(W + sBA)$.

For (ii) the deployed-vs-trained discrepancy is

$$
E(x) = Q_a(x)\big[Q_t(Q_t(W) + sBA) - Q_t(W)\big]^{\top} - s\, x\,(BA)^{\top}.
$$

Because the ternary grid has spacing $\gamma$ and threshold $\gamma/2$, every entry with
$|(sBA)_{ij}| < \gamma/2$ is rounded away in the first term while contributing fully to the second.
LoRA updates under standard learning rates are small relative to $\gamma$ (we measure
$\|sBA\|_{\max}/\gamma = [X.XX]$ and the fraction of entries with $|sBA_{ij}| \ge \gamma/2$ as
$[X.X]\%$), so the merged model discards nearly all of the fine-tuning and behaves like the
projector-only model, while the reported accuracy of the *trained* model was obtained with the
unquantized branch. Option (iii) is not even a merge of the trained function: the training forward
never evaluated $Q_t(W + sBA)$. QA-LoRA's remedy—absorbing the delta into per-group zero-points—is
unavailable here because absmean ternary quantization has one scale per tensor and no zero-point.
Ternary-consistent LoRA removes the issue by construction: $\hat W$ is the trained weight.

### 2.3 No-decode (likelihood-scoring) inference

For tasks with a closed candidate set $\mathcal{C} = \{c_1, \dots, c_K\}$ (POPE and MME: {yes, no};
ScienceQA: the option letters; GQA: the top-[N] answer strings of the training distribution
[verify]), we avoid autoregressive decoding entirely. Given image $v$ and prompt $q$, we score

$$
S(c_k) = \sum_{t=1}^{|c_k|} \log p\big(c_{k,t} \mid v, q, c_{k,<t}\big), \qquad
\hat c = \arg\max_k \; \tfrac{1}{|c_k|^{\beta}} S(c_k), \quad \beta \in \{0, 1\},
$$

where $\beta = 1$ is length normalization (used only when candidates differ in token length). The
shared prefix $(v, q)$ is pre-filled once; the $K$ candidates are appended as a batch and scored in a
single forward pass. Cost: one prefill of $64 + |q|$ tokens plus $\sum_k |c_k|$ token positions
evaluated *in parallel*, instead of $|{\hat c}| + [\text{EOS}]$ sequential memory-bound decode
steps. On CPU with ternary kernels the batched pass is compute-bound and benefits from the
integer matmul; sequential decoding is bandwidth-bound and does not. An optional contextual
calibration (subtracting the score of each candidate under a content-free image) is reported as an
ablation. TextVQA is open-vocabulary and is evaluated only generatively.

### 2.4 Training recipe and compute budget

| | Stage 1 (alignment) | Stage 2 (instruction tuning) |
|---|---|---|
| Trainable | projector | projector + ternary-consistent LoRA |
| LoRA | r = 0 (frozen ternary LM) | r = 32, α = 64 ($s = 2$), 7 projections/block |
| Data | LLaVA-ReCap-558K, 300k samples | LLaVA-OneVision-Data, `sharegpt4v(coco)` subset [+ mixture, TBD], 150k samples |
| `max_len` | 256 | 512 |
| Batch | 8 × accum 2 × 2 GPUs = 32 | 8 × accum 4 × 2 GPUs = 64 |
| Steps / epochs | 9000 (≈1 epoch) | 4500 (≈[1.9] epochs of 150k [verify after mixture is fixed]) |
| LR / schedule | 1e-3, 200 warmup, cosine [verify] | 2e-4, 100 warmup, cosine [verify] |
| Weight decay / clip | 0 / 1.0 | 0 / 1.0 |
| Precision | fp16 autocast + GradScaler; projector and LoRA in fp32 | same |
| Gradient checkpointing | on | on |
| GPU-hours (2×T4) | [10–12] | [20–25] |

The vision encoder is frozen throughout. Activation quantization $Q_a$ is active in both stages so
that the frozen LM in stage 1 already sees the deployed numerics. Checkpoints contain only projector,
LoRA, and optimizer state (a few hundred MB) and are synchronized to a Hugging Face repository at the
end of each Kaggle session; training resumes from the exact data cursor (epoch, shard, row).

**Compute budget as a design constraint.** All training was performed on Kaggle's free tier: two
NVIDIA T4 GPUs (16 GB each, no bf16, no flash attention), sessions capped at 12 h and a quota of
[30] GPU-hours per week. The total budget is ≈[35] GPU-hours ([X] sessions over [Y] weeks), i.e. the
recipe is reproducible by anyone without institutional compute. The constraints shaped the design:
64 image tokens (not 256–729), a frozen encoder, `max_len` ≤ 512, and fp32 adapters under fp16
autocast. We state this as a scope rather than a limitation: the goal is a deployable ternary VLM
and a controlled study of the adaptation scheme, not state-of-the-art accuracy.

Trainable parameter counts: projector [11.5]M; LoRA $r \sum_{\ell} (d_i + d_o) =$ [XX.X]M across [210]
modules [compute from config: q/o $2560{+}2560$, k/v $2560{+}[640]$, gate/up $2560{+}[6912]$, down
$[6912]{+}2560$, per block, × [30] blocks, × 32; verify GQA head counts]; total trainable [XX]M
([X.X]% of the LM).

---

## 3 Experiments

**Setup.** All accuracy numbers are produced with `lmms-eval` [version], greedy decoding,
`max_new_tokens` = [16] for short-answer tasks, the same prompt templates for all models (Appendix
A.2). Baselines are evaluated with their released checkpoints and default image resolutions
(SmolVLM-256M/500M: [512]px, [64] tokens/tile [verify]; LLaVA-OneVision-0.5B: AnyRes up to
[729×N] tokens [verify]). "TernaVLM" always denotes the *merged, exported ternary* model unless a row
says "as trained".

**Benchmarks.** POPE (random/popular/adversarial; we report the mean F1 and accuracy), TextVQA (val,
VQA accuracy, without OCR tokens in the prompt [and with, in Appendix]), GQA (testdev-balanced,
accuracy), ScienceQA-IMG (test, accuracy), MME (perception score /2000 and cognition score /800).

### 3.1 Main results

**Table 1. Comparison with small VLMs.** LM weight memory is the size of the LM weights in the
deployed format (ternary: `TQ2_0`/`I2_S`; baselines: fp16 and the int8/int4 GGUF variants used in
Table 4). "Gen" = generative, "ND" = no-decode scoring (Section 2.3). Baseline rows are re-evaluated
under our protocol; published numbers, where they exist, go in parentheses.

| Model | LM params | LM weight bits | LM mem (MB) | Img tokens | POPE F1 | TextVQA | GQA | SQA-IMG | MME-P | MME-C |
|---|---|---|---|---|---|---|---|---|---|---|
| SmolVLM-256M | 135M | 16 | [XXX] | [64/tile] | [XX.X] | [XX.X] ([pub]) | [XX.X] | [XX.X] | [XXXX] | [XXX] |
| SmolVLM-500M | 360M | 16 | [XXX] | [64/tile] | [XX.X] | [XX.X] ([pub]) | [XX.X] | [XX.X] | [XXXX] | [XXX] |
| LLaVA-OneVision-0.5B | 0.5B | 16 | [XXXX] | [729×N] | [XX.X] | [XX.X] ([pub]) | [XX.X] | [XX.X] | [XXXX] | [XXX] |
| Moondream2 (optional) | 1.9B | 16 | [XXXX] | [XXX] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XXXX] | [XXX] |
| Qwen2.5-1.5B control (our recipe, fp16, plain LoRA) | 1.5B | 16 | [XXXX] | 64 | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XXXX] | [XXX] |
| Qwen2.5-1.5B control, PTQ to ternary after tuning | 1.5B | 1.58 | [XXX] | 64 | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XXXX] | [XXX] |
| **TernaVLM** (Gen) | 2.4B | 1.58 | [XXX] | 64 | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XXXX] | [XXX] |
| **TernaVLM** (ND) | 2.4B | 1.58 | [XXX] | 64 | [XX.X] | — | [XX.X] | [XX.X] | [XXXX] | [XXX] |

*Expected outcome.* TernaVLM should trail the fp16 Qwen2.5-1.5B control by a few points on every
benchmark (the ternary base is weaker per parameter and 64 tokens limit TextVQA), and should be
competitive with SmolVLM-500M and LLaVA-OneVision-0.5B on POPE, GQA and ScienceQA while being
clearly behind on TextVQA, where baselines use higher-resolution tiling. The PTQ-to-ternary control
row is expected to collapse (near-chance GQA/POPE), which isolates *pre-trained* ternary weights, not
the fine-tuning recipe, as the reason the approach works. If TernaVLM is not within [5] points of
the fp16 control on POPE/GQA, the framing of the paper must change from "comparable" to "trade-off".

### 3.2 Ablation: adaptation scheme

**Table 2. Adaptation schemes on the same base, data, and schedule.** "Deployable ternary" means the
exported LM consists only of ternary tensors plus per-tensor scales. "Δ entries" is the fraction of
LM weight entries whose ternary value differs between the deployed model and the stage-1 model
(zero for rows without LM updates). Avg is the mean of POPE F1, GQA, SQA-IMG, and TextVQA.

| # | Scheme | Trained forward | Exported weight | Deployable ternary | Eval mode | Δ entries (%) | POPE F1 | GQA | SQA-IMG | TextVQA | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | Projector-only (stage 1 only) | $Q_a(x)Q_t(W)^\top$ | $Q_t(W)$ | yes | — | 0.0 | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] |
| B | Plain LoRA, fp16 delta kept | $Q_a(x)Q_t(W)^\top + s\,x(BA)^\top$ | $Q_t(W)$, $BA$ fp16 | no | as trained | — | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] |
| C | Plain LoRA, merge (ii) | same as B | $Q_t(Q_t(W)+sBA)$ | yes | merged | [X.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] |
| C′ | Plain LoRA, merge (iii) | same as B | $Q_t(W+sBA)$ | yes | merged | [X.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] |
| D | Ternary-consistent LoRA (ours) | $Q_a(x)Q_t(W+sBA)^\top$ | $Q_t(W+sBA)$ | yes | as trained | [X.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] |
| D′ | Ours, merged and exported | same as D | same as D | yes | merged | = D | [= D] | [= D] | [= D] | [= D] | [= D] |

Also report for D′ the maximum absolute logit difference against D over [1000] evaluation samples
(expected 0 up to fp16 accumulation-order noise, [X.Xe-X]).

*Expected outcome.* B is an upper reference that is not deployable as a ternary model. C should fall
back to roughly A (the delta is rounded away; Δ entries ≪ 1%), and C′ may be *worse* than A because
it deploys a function that was never trained. D should approach B (within [1–2] points) and D′ must
equal D exactly; any nonzero gap between D and D′ is a bug, not a result. The interesting scientific
question is the gap B − D: how much does constraining the update to the ternary grid cost.

**Table 2b. Secondary ablations on scheme D** (one change at a time; report Avg from Table 2).

| Variant | Avg | Note |
|---|---|---|
| r = 16 (α = 32) | [XX.X] | capacity |
| r = 32 (default) | [XX.X] | — |
| $Q_a$ disabled during training, enabled at deploy | [XX.X] | activation train/test mismatch |
| $Q_a$ disabled during training and deploy (fp16 activations) | [XX.X] | not integer-only |
| Pixel shuffle 4×4 → 16 image tokens | [XX.X] | token budget |
| Stage 2 without stage 1 (random projector) | [XX.X] | necessity of alignment |
| Base LM = Falcon-E-1B-Instruct (if time permits) | [XX.X] | second ternary backbone |

### 3.3 No-decode vs generative inference

**Table 3. Closed-set answering.** Tokens/answer counts generated tokens (Gen) or scored candidate
tokens (ND). Latency is per question on the laptop CPU of Table 4, excluding image encoding
(identical for both modes), median over [200] questions.

| Benchmark | Candidates K | Gen acc | ND acc (β=0) | ND acc (β=1) | ND + calibration | Gen tokens/answer | ND tokens/answer | Gen latency (ms) | ND latency (ms) | Speed-up |
|---|---|---|---|---|---|---|---|---|---|---|
| POPE (F1) | 2 | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [X.X] | 2 | [XXX] | [XXX] | [X.X]× |
| MME-P (score) | 2 | [XXXX] | [XXXX] | [XXXX] | [XXXX] | [X.X] | 2 | [XXX] | [XXX] | [X.X]× |
| ScienceQA-IMG | 2–5 | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [X.X] | [X] | [XXX] | [XXX] | [X.X]× |
| GQA (top-[N] answers) | [N] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [X.X] | [XXX] | [XXX] | [XXX] | [X.X]× |

*Expected outcome.* On yes/no tasks ND should match or slightly exceed Gen (it removes formatting
failures) at 2–4× lower latency. On ScienceQA, ND with letter candidates should be within [1] point.
On GQA the candidate set truncates the label space, so ND accuracy is bounded by the coverage of the
top-[N] answers ([XX.X]% of testdev labels); report that bound alongside the result. Calibration is
expected to help POPE-adversarial (yes-bias) and be neutral elsewhere.

### 3.4 CPU latency and memory

**Table 4. On-device inference.** All models run in the same runtime (llama.cpp `mtmd`, commit
[hash]; ternary weights via `TQ2_0`, cross-checked against bitnet.cpp `I2_S` for the LM-only path)
with [N] threads, batch 1, fp16 KV cache, image encoding included in prefill. Memory is peak RSS.
Prompt: one 256px image + [30]-token question; decode: 32 tokens. Median of [10] runs after one
warm-up.

*Hardware A: laptop CPU* ([vendor model, cores/threads, RAM, OS]).

| Model | LM format | Total mem (MB) | LM weights (MB) | Prefill (tok/s) | TTFT (ms) | Decode (tok/s) | POPE F1 (Table 1) |
|---|---|---|---|---|---|---|---|
| SmolVLM-256M | F16 | [XXX] | [XXX] | [XXXX] | [XXX] | [XX.X] | [XX.X] |
| SmolVLM-256M | Q8_0 | [XXX] | [XXX] | [XXXX] | [XXX] | [XX.X] | [XX.X] |
| SmolVLM-500M | F16 | [XXX] | [XXX] | [XXXX] | [XXX] | [XX.X] | [XX.X] |
| SmolVLM-500M | Q8_0 | [XXX] | [XXX] | [XXXX] | [XXX] | [XX.X] | [XX.X] |
| LLaVA-OneVision-0.5B | F16 | [XXXX] | [XXXX] | [XXXX] | [XXX] | [XX.X] | [XX.X] |
| LLaVA-OneVision-0.5B | Q4_K_M | [XXX] | [XXX] | [XXXX] | [XXX] | [XX.X] | [XX.X] |
| Qwen2.5-1.5B control | Q4_K_M | [XXXX] | [XXX] | [XXXX] | [XXX] | [XX.X] | [XX.X] |
| **TernaVLM** | TQ2_0 | [XXX] | [XXX] | [XXXX] | [XXX] | [XX.X] | [XX.X] |
| **TernaVLM** | I2_S (bitnet.cpp, LM only) | [XXX] | [XXX] | [XXXX] | — | [XX.X] | — |

*Hardware B: Raspberry Pi 5 (8 GB)* — same columns, same rows (drop rows that do not fit in memory
and say so).

| Model | LM format | Total mem (MB) | LM weights (MB) | Prefill (tok/s) | TTFT (ms) | Decode (tok/s) | POPE F1 |
|---|---|---|---|---|---|---|---|
| … | … | … | … | … | … | … | … |

*Expected outcome.* TernaVLM's LM has ~[4–5]× more parameters than SmolVLM-500M's but stores them in
≈[1/8] the bytes per weight, so LM weight memory should be comparable to SmolVLM-500M-F16 and
roughly [1.5–2]× LLaVA-OneVision-0.5B-Q4; decode tok/s is expected between SmolVLM-500M-Q8 and
LLaVA-OV-0.5B-Q4 (ternary kernels are bandwidth-efficient but 2.4B parameters is not 0.5B).
SmolVLM-256M will be faster in absolute terms; the claim is accuracy per byte and per millisecond
(Figure 3), not absolute speed. Prefill is where 64 image tokens and integer kernels should give
the largest relative advantage over AnyRes baselines. If TQ2_0 lacks an optimized path on the test
CPU, report it and use bitnet.cpp numbers for the LM.

---

## 4 Related work

**Ternary and 1-bit language models.** BitNet [Wang et al., 2023] and BitNet b1.58
[Ma et al., 2024] train transformers with binary/ternary weights and int8 activations from scratch;
BitNet b1.58 2B-4T [Ma et al., 2025] scales this to a 2B model on 4T tokens and releases a
bf16 latent-weight checkpoint for fine-tuning, which is our backbone. Falcon-E [TII, 2025] and
Spectra/TriLM [Kaushal et al., 2024] provide further ternary LMs; bitnet.cpp [Wang et al., 2025]
and llama.cpp's `TQ1_0/TQ2_0` types supply CPU kernels. None of these models has a vision interface.
[1–2 sentences on ternary vision models, e.g. 1.58-bit FLUX, if cited.]

**Parameter-efficient fine-tuning of quantized models.** QLoRA [Dettmers et al., 2023] fine-tunes
fp LoRA adapters on an NF4-quantized frozen base; the deployed model keeps fp adapters or is merged
into a de-quantized base. QA-LoRA [Xu et al., 2023] makes the merge quantization-aware by folding the
adapter into per-group zero-points, which requires a zero-point and group structure that absmean
ternary quantization does not have. LoftQ [Li et al., 2023] initializes the adapter to compensate
quantization error of the base; PEQA [Kim et al., 2023] trains only quantization scales. Our scheme
differs in that the quantizer is applied to the *sum* of base and adapter in the forward pass
(quantization-aware training restricted to a low-rank subspace), so no separate merge step exists
and the STE gradient is the standard one from QAT [Bengio et al., 2013; Jacob et al., 2018].

**Small vision-language models.** LLaVA / LLaVA-1.5 [Liu et al., 2023a,b] established the
encoder–projector–LLM recipe and the two-stage schedule we follow; LLaVA-OneVision [Li et al., 2024]
provides the 0.5B baseline and the instruction data. SmolVLM [Marafioti et al., 2025], Moondream
[Korrapati, 2024], and FastVLM [Vasu et al., 2025] reduce cost by shrinking the LM, the encoder, or
the number of visual tokens, all in 16-bit or PTQ-int weights. Pixel-shuffle token reduction was
introduced to VLMs by InternVL-1.5 [Chen et al., 2024] and adopted by Idefics3 [Laurençon et al.,
2024] and SmolVLM. SigLIP2 [Tschannen et al., 2025] is our encoder. MM1 [McKinzie et al., 2024]
studies the image-token/resolution trade-off; Matryoshka Multimodal Models [Cai et al., 2024] and the
Matryoshka Query Transformer [Hu et al., 2024] make the token count adjustable at inference, which
is complementary to our fixed 64-token budget.

**Likelihood-based answering.** Scoring candidates by log-likelihood is standard in LM evaluation
harnesses [Gao et al., 2023] and was used with calibration for few-shot classification
[Zhao et al., 2021]; we apply it to VLM benchmarks as a deployment mode and measure its CPU cost.

---

## 5 Limitations

[Placeholder — to be written from results. Expected points:]
- Fixed 256px input and 64 tokens limit text-heavy tasks (TextVQA, DocVQA); no tiling/AnyRes.
- The vision encoder and projector remain fp16/fp32; the model is "ternary LM + fp encoder", and the
  encoder's [93]M parameters are a fixed fraction of memory and prefill time.
- Ternary-consistent LoRA relies on the STE; we observe [describe: dead-zone behaviour, sensitivity
  to lr, need for larger α] and do not study alternative estimators.
- Single ternary backbone; results may not transfer to other ternary models (Falcon-E row pending).
- Training data is a subset of public LLaVA data; no safety tuning; [note license constraints].
- Baselines are re-evaluated under one protocol; discrepancies with published numbers are reported
  but not reconciled.

## 6 Conclusion

[Placeholder.] We showed that a pre-trained ternary LLM can be turned into a vision-language model
without leaving the ternary regime, by quantizing the LoRA update jointly with the base weight so
that the merged model is exactly the trained model. On a free-tier compute budget, TernaVLM reaches
[summary numbers] with an LM footprint of [XXX] MB and [XX] tok/s on a laptop CPU. The
ternary-consistent adapter is not specific to vision and applies to any fine-tuning of ternary LMs;
future work includes [ternary projector/encoder, AnyRes tiling, larger instruction mixtures].

---

## References (expected; verify all identifiers before submission)

- Bengio, Léonard, Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv:1308.3432, 2013.
- Cai et al. Matryoshka Multimodal Models. arXiv:2405.17430, 2024.
- Chen et al. How far are we to GPT-4V? Closing the gap … (InternVL-1.5). arXiv:2404.16821, 2024.
- Chen et al. ShareGPT4V: Improving large multi-modal models with better captions. arXiv:2311.12793, 2023.
- Dettmers, Pagnoni, Holtzman, Zettlemoyer. QLoRA: Efficient finetuning of quantized LLMs. NeurIPS 2023, arXiv:2305.14314.
- Fu et al. MME: A comprehensive evaluation benchmark for multimodal LLMs. arXiv:2306.13394, 2023.
- Gao et al. A framework for few-shot language model evaluation (lm-evaluation-harness). Zenodo, 2023.
- Gerganov et al. llama.cpp. GitHub, 2023–.
- Hu et al. LoRA: Low-rank adaptation of large language models. ICLR 2022, arXiv:2106.09685.
- Hu et al. Matryoshka Query Transformer for large vision-language models. arXiv:2405.19315, 2024.
- Hudson, Manning. GQA: A new dataset for real-world visual reasoning. CVPR 2019.
- Jacob et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference. CVPR 2018.
- Kaushal et al. Spectra: A comprehensive study of ternary, quantized, and FP16 language models. arXiv:2407.12327, 2024.
- Kim et al. Memory-efficient fine-tuning of compressed large language models via sub-4-bit integer quantization (PEQA). NeurIPS 2023, arXiv:2305.14152.
- Korrapati. Moondream. GitHub, 2024. [no paper; cite repository]
- Laurençon, Marafioti, Sanh, Tronchon. Building and better understanding vision-language models: insights and future directions (Idefics3). arXiv:2408.12637, 2024.
- Li et al. LLaVA-OneVision: Easy visual task transfer. arXiv:2408.03326, 2024.
- Li et al. LoftQ: LoRA-fine-tuning-aware quantization for large language models. ICLR 2024, arXiv:2310.08659.
- Li et al. Evaluating object hallucination in large vision-language models (POPE). EMNLP 2023, arXiv:2305.10355.
- Liu, Li, Wu, Lee. Visual instruction tuning (LLaVA). NeurIPS 2023, arXiv:2304.08485.
- Liu, Li, Li, Lee. Improved baselines with visual instruction tuning (LLaVA-1.5). CVPR 2024, arXiv:2310.03744.
- Lu et al. Learn to explain: Multimodal reasoning via thought chains for science question answering (ScienceQA). NeurIPS 2022.
- Ma et al. The era of 1-bit LLMs: All large language models are in 1.58 bits. arXiv:2402.17764, 2024.
- Ma et al. BitNet b1.58 2B4T technical report. arXiv:2504.12285, 2025.
- Marafioti et al. SmolVLM: Redefining small and efficient multimodal models. arXiv:2504.05299, 2025.
- McKinzie et al. MM1: Methods, analysis and insights from multimodal LLM pre-training. arXiv:2403.09611, 2024.
- Singh et al. Towards VQA models that can read (TextVQA). CVPR 2019.
- TII. Falcon-E / Falcon-Edge: a series of powerful, universal, fine-tunable 1.58-bit language models. Technical report / blog, 2025. [verify citation form]
- Tschannen et al. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv:2502.14786, 2025.
- Vasu et al. FastVLM: Efficient vision encoding for vision language models. CVPR 2025, arXiv:2412.13303.
- Wang et al. BitNet: Scaling 1-bit transformers for large language models. arXiv:2310.11453, 2023.
- Wang et al. Bitnet.cpp: Efficient edge inference for ternary LLMs. arXiv:2502.11880, 2025.
- Xu et al. QA-LoRA: Quantization-aware low-rank adaptation of large language models. ICLR 2024, arXiv:2309.14717.
- Zhai et al. Sigmoid loss for language image pre-training (SigLIP). ICCV 2023.
- Zhang et al. LMMs-Eval: Reality check on the evaluation of large multimodal models. arXiv:2407.12772, 2024.
- Zhao et al. Calibrate before use: Improving few-shot performance of language models. ICML 2021, arXiv:2102.09690.

---

## Appendix

### A.1 Reproducibility checklist

- [ ] Code: `ternavlm/ternary.py` (quantizers, `TernaryLoRALinear`, `merge_all`), `ternavlm/vlm.py`,
      `ternavlm/data.py`, `train.py`, `scripts/export.py`, `kaggle/run_stage.py`; commit hash [XXXXXXX].
- [ ] Configs: `configs/stage1.yaml`, `configs/stage2.yaml` reproduced verbatim in A.3; `smoke.yaml`
      for a 0.5 GPU-h sanity run.
- [ ] Base checkpoints and revisions: `microsoft/bitnet-b1.58-2B-4T-bf16` [revision],
      `google/siglip2-base-patch16-256` [revision]; `transformers` [version], `torch` [version].
- [ ] Data: `lmms-lab/LLaVA-ReCap-558K` (first 300k rows of the seed-0 shard order, buffer-shuffled);
      `lmms-lab/LLaVA-OneVision-Data` subsets [list], 150k samples, seed 1; exact sample-order
      reproducibility via the resumable loader.
- [ ] Prompt format: [chat template of the base model / `User:`–`Assistant:` fallback], label masking
      on the assistant span only; `<image>` expanded to 64 copies.
- [ ] Unit test: `python tests/test_ternary.py` verifies (a) `merge()` equals $Q_t(W + sBA)$
      bit-for-bit, (b) forward equality between `TernaryLoRALinear` and the merged `nn.Linear`,
      (c) the number of wrapped modules is $7 \times$ layers and no double quantization occurs.
- [ ] Training logs: loss curves for both stages, fraction of flipped ternary entries per layer vs.
      step, $\|sBA\|_{\max}/\gamma$ per layer at the end of stage 2.
- [ ] Evaluation: `lmms-eval` version and task configs; `max_new_tokens`; candidate lists for
      no-decode mode (Table 3) published as JSON.
- [ ] CPU benchmark: runtime commit, build flags, thread count, GGUF conversion commands for every
      model in Table 4; raw timing logs.
- [ ] Compute: GPU-hours per stage and number of Kaggle sessions; wall-clock per session; peak GPU
      memory.
- [ ] Released artifacts: merged ternary HF checkpoint, GGUF (`TQ2_0`) files, projector/LoRA
      checkpoints, evaluation predictions.
- [ ] Random seeds and the number of runs per number ([1] run per configuration; state this
      explicitly and report the smoke-run seed variance if more runs are unaffordable).

### A.2 Evaluation prompts

[Exact prompt strings per benchmark, identical across models, and the answer-extraction rules.]

### A.3 Full configurations

[Verbatim `configs/stage1.yaml` and `configs/stage2.yaml`; the stage-2 data mixture once fixed.]

### A.4 Additional tables

- Per-split POPE (random / popular / adversarial) for all rows of Table 1.
- TextVQA with OCR tokens in the prompt.
- MME per-subtask scores.
- Table 2 with standard error from the smoke-run seed study, if available.
- Per-layer statistics: $\gamma$ before/after stage 2, fraction of $0 \leftrightarrow \pm1$ flips.

### A.5 Figure list

- **Figure 1 — Architecture.** Image → SigLIP2-base (frozen, 256 patches) → 2×2 pixel shuffle (64
  tokens × 3072) → MLP projector (fp32, trained) → spliced into `<image>` slots of the token
  embedding sequence → BitNet b1.58 2B-4T with `TernaryLoRALinear` on all seven projections per
  block. Annotate tensor shapes and which blocks are trained in stage 1 vs. stage 2.
- **Figure 2 — Quantization consistency.** Left: plain LoRA training graph (ternary base branch +
  fp delta branch) and its three export options (i)–(iii), with the mismatch term $E(x)$. Right:
  ternary-consistent LoRA (single fused branch $Q_t(W + sBA)$) and its export, which is the same
  graph. Inset: a 1-D illustration of the ternary grid with threshold $\gamma/2$, showing a latent
  entry moved by $sBA$ below the threshold (discarded under merge) versus across it (flipped).
  Optional panel: histogram of $|sBA_{ij}|/\gamma$ at the end of stage 2 with the $1/2$ threshold
  marked, and the resulting flipped fraction.
- **Figure 3 — Latency vs. accuracy.** Scatter of POPE F1 (y) against decode tok/s on the laptop CPU
  (x, log scale), marker area proportional to peak memory, one point per row of Table 4; TernaVLM in
  both Gen and ND modes. A second panel or inset with the Raspberry Pi 5 numbers.
- (Supplementary) Training curves for both stages; flipped-entry fraction vs. step.
