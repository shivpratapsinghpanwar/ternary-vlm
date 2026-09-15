# TernaVLM: related work and novelty assessment

Survey date: 2026-09-15. Scope: everything found on arXiv, Hugging Face, GitHub and vendor blogs that
bears on (1) ternary / 1-bit vision-language models, (2) LoRA merged into a quantized base,
(3) fine-tuning recipes for BitNet b1.58 2B4T and Falcon-E, (4) small/edge VLM baselines and their
numbers, (5) likelihood scoring instead of generation. Every claim carries a URL. Items I could not
verify from a primary source are marked **[unverified]** and listed again in the last section.

## TL;DR

* **Ternary VLMs already exist. Three of them.** Intel's LLaVaOLMoBitnet1B (Aug 2024) was the first,
  BitVLA (Jun 2025) attaches SigLIP-L to *exactly our backbone* (BitNet b1.58 2B4T) and releases the
  vision-language checkpoints, and PrismML's Ternary Bonsai 27B (Jul 2026) is a ternary Qwen3.6 with a
  4-bit vision tower that runs on a phone. The README sentence "nobody has released a ternary VLM" is
  false and must be removed.
* **Joint quantization Q(W + s·BA) with STE is not new either.** L4Q (Feb 2024) and LR-QAT (Jun 2024)
  both put the LoRA product inside the quantizer so the merged model is a single integer tensor. They
  do it for 3/4-bit uniform quantizers on LLaMA; nobody has published it for BitNet's per-tensor
  absmean ternary quantizer, and nobody has used it to build a VLM.
* **Likelihood scoring of candidate answers is standard practice** (Flamingo/OpenFlamingo,
  SEED-Bench, lm-eval-harness `multiple_choice`, lmms-eval `seedbench_ppl`). It is an evaluation
  protocol, not a contribution, and its numbers are not comparable with generation-based numbers.
* **What is left that is defensible:** parameter-efficient (no master-weight update) adaptation of a
  native ternary LLM to vision with an exactly-ternary export, a controlled comparison against
  mixed-precision LoRA and full QAT, and a CPU/edge deployment study with a ternary VLM. That is a
  workshop-paper-sized contribution if the experiments are clean; it is not an arXiv "first".

---

## 1. Closest prior work

| Work | Venue / date | What they did | How TernaVLM differs |
|---|---|---|---|
| **LLaVaOLMoBitnet1B** (Intel Labs) — [arXiv 2408.13402](https://arxiv.org/abs/2408.13402), [HF model](https://huggingface.co/IntelLabs/LlavaOLMoBitnet1B) | arXiv, Aug 2024 | "First Ternary Multimodal LLM." CLIP ViT-L/14 + 2-layer MLP + NousResearch OLMo-Bitnet-1B (16 layers, BitLinear158, only 60B Dolma tokens). LLaVA-1.5 recipe: 595K CC (projector only), then LLaVA-Instruct-150K with **full LLM fine-tuning** ("We adopt full LLM finetuning instead of any low-rank approaches"). Reports POPE 66.92, VQAv2 68.41, TextVQA 26.3. Vision encoder and projector stay full precision. | Stronger backbone (2B4T vs an undertrained 1B PoC), SigLIP2 + pixel-shuffle to 64 tokens, LoRA-only adaptation with joint ternary quantization, CPU deployment. Their paper's own "future work" asks for exactly our direction: "effective ways to Post-Training Quantize/Quantization-Aware Finetune open weight pre-trained models to ternary domain." |
| **BitVLA** (ICT/CAS; Hongyu Wang et al., BitNet authors) — [arXiv 2506.07530](https://arxiv.org/abs/2506.07530), [GitHub](https://github.com/ustcwhy/BitVLA) | arXiv, Jun 2025 (v2 2026) | Vision-language-action model built on **BitNet b1.58 2B4T + SigLIP-L (224 px)**. Stage I: connector only on LLaVA-1.5-558k. Stage II: **LLM + connector, full-parameter QAT** on a 10M-sample MammoTH-VL subset ("quantization is performed on-the-fly during the forward pass … we adopt the straight-through estimator", "gradients and optimizer states are maintained in full precision"). Stage III "Quantize-then-Distill" ternarizes the vision encoder (W1.58-A8) with a full-precision teacher. Table III VLM numbers (bf16 ViT / ternary ViT): MMMU 37.4/35.4, SeedBench 70.6/69.3, SeedBench2+ 45.0/43.7, MMStar 43.6/41.5, AI2D 68.6/67.6. VL-pretrained checkpoints `bitvla-siglipL-224px-bf16` and `bitvla-bitsiglipL-224px-bf16` are listed in the README (HF card returned HTTP 401 to my fetch — **[unverified]** whether public). | Same backbone. They update all 2B latent weights with 10M samples on GPUs; we train only projector + LoRA on ~150k samples and export the identical ternary tensors. They do not study parameter-efficient adaptation, do not report CPU inference, and use the model as a VLA policy. BitVLA is the mandatory baseline and the biggest threat to any "first ternary VLM on BitNet" claim. |
| **Ternary Bonsai 27B** (PrismML) — [news](https://prismml.com/news/prismml-releases-bonsai-27b), [HF](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf), [demo repo](https://github.com/PrismML-Eng/Bonsai-demo), [llama.cpp discussion](https://github.com/ggml-org/llama.cpp/discussions/22019) | Product release, Jul 2026 | Ternary (group-128 FP16 scales, 1.71 bpw) and binary builds of Qwen3.6-27B with a **4-bit HQQ vision tower**; the 27B is their first vision-language model. Trained on TPU v5 (method not disclosed in the announcement; whitepaper in repo). 5.9 GB, phone-class inference, 95% of FP16 on 15 text benchmarks. Smaller Bonsai (1.7B/4B/8B) are text-only. | Not BitNet-native (converted Qwen3.6), group-wise not per-tensor scales, vision tower not ternary, 27B not 2B, no VLM benchmark numbers found. Relevant mainly as evidence that "ternary + vision on device" is already a shipping product. |
| **L4Q** — [arXiv 2402.04902](https://arxiv.org/abs/2402.04902) | arXiv Feb 2024 (v5 Dec 2024; listed as ACL 2025 by [papernotes](https://en.papernotes.org/ACL2025/model_compression/l4q_parameter_efficient_quantization_aware_finetuning/)) | `W_comb = W0 + α·BA`, then `W_q = round(clamp((W_comb − b)/s)) · s + b` with learnable scale/bias and STE. Trains A, B, s, b. 3/4-bit, OpenLLaMA-3B, LLaMA-1/2, Mistral-7B; CSQA/MMLU. Produces a fully quantized model. | Structurally the same trick as `TernaryLoRALinear.effective_weight()`. Differences: our quantizer is BitNet absmean ternary with a *data-dependent, non-learned* per-tensor scale, plus int8 per-token activation quantization; ours is applied to a natively-ternary pretrained model (no PTQ error to recover) and to a multimodal task. |
| **LR-QAT** (Qualcomm) — [arXiv 2406.06385](https://arxiv.org/abs/2406.06385), [code](https://github.com/Qualcomm-AI-research/lr-qat), [ICML 2024 workshop](https://mlanthology.org/icmlw/2024/bondarenko2024icmlw-low/) | Jun 2024 | Eq. (4)/(5) of the paper: `Ŵ = s · clip(round(W0/s0 + (α/r)·AB), −2^{b−1}, 2^{b−1}−1)`; the low-rank term sits *inside* the rounding, in integer-grid units; STE for A, B, s; W0/s0 downcast to INT-b or fixed point; after training the result "can be represented as a regular fixed point tensor … without any extra overhead". b ∈ {3,4}, LLaMA-1/2/3, Mistral. | Same goal (adapter folds into one integer tensor). Their adapter lives in quantization-step units, which sidesteps the "delta too small to flip a state" problem (see risks); ours adds in weight units under an absmean scale that itself moves with BA. Not ternary, not multimodal, not on a native low-bit model. |
| **LoTA-QAF** — [arXiv 2505.18724](https://arxiv.org/abs/2505.18724) | arXiv May 2025 (v2 Sep 2025) | *Ternary adapter* `Ŵ ∈ {−1,0,1}` added to a GPTQ integer weight (`W'_int = W_int + Ŵ`) so merging is lossless; t-SignSGD optimizer. Base is 2/3/4-bit GPTQ Llama-3.1 / Qwen-2.5, **not** a ternary model. | Ternary refers to the adapter, not the base. Complementary idea; worth citing as the other way to keep a merge exact. |
| **BitLoRA** (Song & Lee, Gachon U.) — [Expert Systems with Applications 2026](https://www.sciencedirect.com/science/article/abs/pii/S0957417426003106) | Journal, 2026 | "Quantization-compatible adapter tuning for 1.58-bit LLM" for federated on-device agents; abstract says all linear layers "function within a BitLinear framework quantized to 1.58-bit" and that BitNet's inference efficiency "is sustained during inference"; 45.42% avg on medical QA; >85% GPU memory reduction. Full text paywalled — **[unverified]** whether adapters are quantized jointly with W or separately. | If BitLoRA quantizes W and BA *jointly*, our LoRA mechanism is not new even in the ternary setting. Must be read before submission. Either way it is text-only and federated-learning-motivated. |
| **QVAC / Tether, "LoRA fine-tuning BitNet b1.58 on edge GPUs"** — [HF blog](https://huggingface.co/blog/qvac/fabric-llm-finetune-bitnet), [mirror](https://qvac.tether.io/blog/lora-fine-tuning-bitnet-b1-58-llms-on-heterogeneous-edge-gpus-via-qvac-fabric) | Blog, Mar 2026 | Standard PEFT LoRA (r=8, α=16) on BitNet b1.58 with **FP16 adapters alongside ternary base** — the mixed-precision case our README argues against. | This is the natural "plain LoRA" baseline for our ablation: same adapters, no joint quantization, deployed model not ternary. |
| **Falcon-E / Falcon-Edge** (TII) — [HF blog](https://huggingface.co/blog/tiiuae/falcon-edge), [onebitllms](https://github.com/tiiuae/onebitllms), [Axolotl docs](https://docs.axolotl.ai/docs/1_58bit_finetuning.html) | May 2025 | 1B/3B BitNet-architecture LMs with bf16, "prequantized" and ternary checkpoints; `onebitllms` swaps `nn.Linear` for `BitnetLinear` for **full fine-tuning of latent weights**, then `quantize_to_1bit`. Blog: PEFT "remains unsupported … an exciting and impactful open question", and lists "first multi-modal Bitnet VLM" as future work. | Confirms the gap we fill (PEFT on ternary) — but note BitVLA closed the "multimodal BitNet VLM" gap a month after this blog. |
| **BitNet b1.58 2B4T** — [arXiv 2504.12285](https://arxiv.org/abs/2504.12285), [bf16 card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16), [GGUF](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf) | Apr 2025 | 2B native ternary LM, 4T tokens, W1.58-A8, absmean per-tensor weights, absmax per-token int8 activations, ReLU², subln, no biases. bf16 repo "contains the master weights … intended only for training or fine-tuning". | Our backbone. Our `ternary_quant` / `int8_act_quant_ste` replicate their scheme, which is what makes exact export possible. |
| **BitNet a4.8** — [arXiv 2411.04965](https://arxiv.org/abs/2411.04965) | Nov 2024 | 4-bit activations for attention/FFN inputs via hybrid quantization + sparsification, 3-bit KV cache, 55% active parameters. | Orthogonal; possible future activation setting for us. |
| **BitNet Distillation (BitDistill)** — [arXiv 2510.13998](https://arxiv.org/abs/2510.13998) | Oct 2025 | Converts full-precision LLMs (Qwen) to 1.58-bit for downstream tasks via SubLN + attention distillation + continual pretraining; 10× memory, 2.65× CPU speed. | Opposite direction (fp → ternary). Not PEFT, not multimodal. |
| **TernaryCLIP** — [arXiv 2510.21879](https://arxiv.org/abs/2510.21879); **ViT-1.58b** — [arXiv 2406.18051](https://arxiv.org/pdf/2406.18051); **BitMedViT** — [arXiv 2510.13760](https://arxiv.org/pdf/2510.13760) | 2024–2025 | Ternary CLIP / ViT encoders (QAT + distillation). | We keep SigLIP2 frozen in full precision (as LLaVaOLMoBitnet1B did). BitVLA ternarizes the encoder; if we claim "fully ternary" we must do the same. |
| **PTQ of VLMs to ultra-low bit**: LUQ [2509.23729](https://arxiv.org/html/2509.23729v1), Q-VLM [2404.14047](https://arxiv.org/html/2404.14047v2), SAB-LVLM binarization [2607.01876](https://arxiv.org/pdf/2607.01876), HBVLA 1-bit PTQ for VLA [2602.13710](https://arxiv.org/html/2602.13710) | 2024–2026 | Post-training quantization / binarization of existing full-precision VLMs. | Different regime (PTQ of fp model vs adaptation of a natively ternary model). Cite to delimit. |

### Quantization-aware LoRA family: precise differences from Q(W + s·BA)

| Method | Where the adapter lives | Deployed model | Difference from ours |
|---|---|---|---|
| **QLoRA** — [arXiv 2305.14314](https://arxiv.org/abs/2305.14314) (NeurIPS 2023) | `y = Q_nf4(W)x + BAx`, adapters bf16 | Mixed precision (4-bit + bf16 adapters); merging requires re-quantizing and changes the function | Adapter outside quantizer; train/deploy mismatch after merge is exactly what we avoid |
| **QA-LoRA** — [arXiv 2309.14717](https://arxiv.org/abs/2309.14717) ([ICLR 2024](https://openreview.net/pdf?id=WvFoJccpo8)) | Group-wise INT4; LoRA A's input dim tied to the number of quantization groups so BA is a per-group offset that merges into zero-points losslessly | Pure INT4 | Exact merge is achieved by a structural constraint on A (one value per group); the base is PTQ'd. Ours has no rank/group constraint and the quantizer sees the full W+BA. |
| **LoftQ** — [arXiv 2310.08659](https://arxiv.org/abs/2310.08659) (ICLR 2024) | Alternating quantization + SVD to initialize A,B so `Q(W)+BA ≈ W`; then trains like QLoRA | Mixed precision | Initialization method; adapter still outside quantizer |
| **LQ-LoRA** — [arXiv 2311.12023](https://arxiv.org/abs/2311.12023) (ICLR 2024) | Decompose `W ≈ Q + L1L2` (mixed-precision Q), train L1L2 | Mixed precision | Same as above with a data-aware, mixed-bit Q |
| **IR-QLoRA** — [arXiv 2402.05445](https://arxiv.org/pdf/2402.05445) | Information-retention calibration of Q(W) + information-elastic connector for LoRA | Mixed precision | Adapter outside quantizer |
| **PEQA** — [arXiv 2305.14152](https://arxiv.org/abs/2305.14152) (NeurIPS 2023) | No low-rank adapter; trains only per-channel quantization scales of a sub-4-bit integer W | Fully quantized | Cannot change the integer pattern; BitNet's absmean scale is per-tensor and not a free parameter, so PEQA does not transfer |
| **QST** — [arXiv 2401.07159](https://arxiv.org/abs/2401.07159) (ACL 2024) | Separate side network next to the frozen 4-bit model; no backprop through the base | Mixed (4-bit base + fp side net) | Different architecture; not a merge |
| **L4Q** — [arXiv 2402.04902](https://arxiv.org/abs/2402.04902) | `Q(W0 + αBA)` with learned scale/bias, STE | Fully quantized (3/4-bit) | **Same mechanism as ours**; uniform quantizer with learned s,b instead of absmean ternary |
| **LR-QAT** — [arXiv 2406.06385](https://arxiv.org/abs/2406.06385) | `s·clip(round(W0/s0 + (α/r)AB))`, adapter in integer-grid units, STE | Fully quantized (3/4-bit) | **Same goal as ours**; integer-domain parametrization, downcast storage |
| **LoTA-QAF** — [arXiv 2505.18724](https://arxiv.org/abs/2505.18724) | Ternary adapter added to integer W (2–4-bit GPTQ) | Fully quantized | Ternary adapter, not ternary base |
| **EfficientQAT** — [arXiv 2407.11062](https://arxiv.org/abs/2407.11062) | Block-wise all-parameter QAT then end-to-end scale training | Fully quantized | Not low-rank |
| **LoRAQuant** — [arXiv 2510.26690](https://arxiv.org/abs/2510.26690) | Quantizes the *adapter* (SVD-based mixed precision) on an fp base | fp base + low-bit adapter | Unrelated to ternary base |
| **BitLoRA** — [ESWA 2026](https://www.sciencedirect.com/science/article/abs/pii/S0957417426003106) | LoRA inside a BitLinear framework on a 1.58-bit LLM **[unverified details]** | Claims BitNet efficiency retained | Possibly the same as ours in the ternary regime — read the full text |

Summary of the distinction to state in the paper: QLoRA/LoftQ/LQ-LoRA/IR-QLoRA keep the adapter
**outside** the quantizer (mixed-precision deployment or lossy merge); QA-LoRA and LoTA-QAF make the
merge exact by **constraining the adapter's structure**; PEQA trains only scales; L4Q and LR-QAT put
the adapter **inside** the quantizer with STE, which is what `TernaryLoRALinear` does. Our
differences are (i) the quantizer is BitNet's per-tensor absmean ternary with a data-dependent scale
(no learned s/b, no zero-point), (ii) activations are quantized to int8 per token in the same pass so
training matches the W1.58-A8 kernels used at inference, (iii) the base is natively ternary, so there is
no PTQ error to "recover", and (iv) the application is multimodal adaptation of a frozen ternary LM.

## 2. Fine-tuning recipes for BitNet b1.58 2B4T and Falcon-E

* **Official:** the [bf16 master-weight repo](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16) is "intended only for training or fine-tuning"; Microsoft publishes no fine-tuning script in [microsoft/BitNet](https://github.com/microsoft/BitNet) beyond BitDistill (fp→ternary). Users report difficulty: [issue #295](https://github.com/microsoft/BitNet/issues/295) (SFTTrainer on bf16 weights, loss stuck at 3.3–3.6), [issue #352](https://github.com/microsoft/BitNet/issues/352). Unsupported in [Unsloth #2390](https://github.com/unslothai/unsloth/issues/2390) and [LLaMA-Factory #7775](https://github.com/hiyouga/LlamaFactory/issues/7775) at the time of those issues. Expect the HF `BitNetForCausalLM` class to quantize inside its own linear layers — the README's "no double quantization" check is correct.
* **Full-parameter QAT on latent weights** is the only recipe with published results: BitVLA (above; fp32 grads/optimizer, STE), Falcon-E's `onebitllms` + trl/Axolotl ([blog](https://huggingface.co/blog/tiiuae/falcon-edge), [Axolotl blog](https://huggingface.co/blog/axolotl-ai-co/finetuning-ternary-llms-tii-axolotl): Falcon-E-1B-Base SFT took ~8.5 h on 8×A10G), and the pre-BitNet-2B4T studies "When are 1.58 bits enough?" ([arXiv 2411.05882](https://arxiv.org/pdf/2411.05882)) and "Continual QAT pre-training" ([arXiv 2502.11895](https://arxiv.org/pdf/2502.11895)).
* **LoRA variants:** only mixed-precision LoRA (QVAC/Tether, FP16 adapters, [blog](https://huggingface.co/blog/qvac/fabric-llm-finetune-bitnet)) and BitLoRA (details unverified). TII explicitly says PEFT is unsupported/open. No public recipe fine-tunes BitNet 2B4T with an adapter that merges back to exact ternary.

## 3. Baseline numbers for small/edge VLMs

All numbers are as reported by the cited source; evaluation harness and prompt differ between
sources, so re-run everything you compare against through one harness (lmms-eval) before the paper.

| Model (params) | POPE | TextVQA | GQA | MME | Other | Source |
|---|---|---|---|---|---|---|
| **LLaVaOLMoBitnet1B** (1.1B, ternary) | 66.92 | 26.3 | – | – | VQAv2 68.41 | [arXiv 2408.13402](https://arxiv.org/abs/2408.13402) Table 2 |
| **BitVLA VL-stage** (BitNet 2B4T + SigLIP-L; bf16 ViT / ternary ViT) | – | – | – | – | MMMU 37.4/35.4, SeedBench 70.6/69.3, MMStar 43.6/41.5, AI2D 68.6/67.6, SeedBench2+ 45.0/43.7 | [arXiv 2506.07530](https://arxiv.org/html/2506.07530) Table III |
| **SmolVLM-256M** | not reported | 50.2 | 66.0* | – | MMMU 29.0, DocVQA 58.3, OCRBench 52.6, AI2D 46.4, ChartQA 55.6, ScienceQA 73.8; VQAv2 80.1*; 0.8 GB VRAM | [arXiv 2504.05299](https://arxiv.org/html/2504.05299v1); *lmms-eval re-run in [arXiv 2603.16987](https://arxiv.org/html/2603.16987v1) Table 3 |
| **SmolVLM-500M** | not reported | 60.2 | 70.4* | – | MMMU 33.7, DocVQA 70.5, OCRBench 61.0, AI2D 59.2, ChartQA 62.8, ScienceQA 80.0; VQAv2 85.9*; 1.2 GB VRAM | same |
| **Moondream2 2B (2024-08-26)** | 89.6 / 88.8 / 87.2 (rand/pop/adv) | 65.2 | 64.3 | – | VQAv2 80.3, DocVQA 70.5, TallyQA 82.6/77.6 | [HF README @2024-08-26](https://huggingface.co/vikhyatk/moondream2/blob/2024-08-26/README.md) |
| **Moondream2 2B (2025-04-14)** | – | 76.3 | – | – | DocVQA 79.3, ChartQA 77.5, CountBenchQA 86.4, OCRBench 61.2 | [Moondream blog](https://moondream.ai/blog/moondream-2025-04-14-release) |
| **Moondream 0.5B** | none published | none | none | none | 375 MiB download / 816 MiB runtime (int4); "out-of-the-box accuracy is limited" | [moondream.ai/models](https://moondream.ai/models) |
| **FastVLM-0.5B** (Qwen2-0.5B, 256 tokens @1024px, R4 row) | 86.6 | 62.9 | 63.1 | – | SQA 81.5, DocVQA 70.4, SeedBench 69.2, MMMU 32.9 **[column alignment unverified — check Table 6]**; R5 (more data): POPE 86.2, TextVQA 65.8, GQA 62.7, DocVQA 79.1 | [arXiv 2412.13303](https://arxiv.org/html/2412.13303v2) Table 6, [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Vasu_FastVLM_Efficient_Vision_Encoding_for_Vision_Language_Models_CVPR_2025_paper.pdf) |
| **LLaVA-OneVision-0.5B** | 87.17† | 49.54† | 57.95† | – | paper: ScienceQA 67.2, MMVet 29.1, DocVQA 70.0, MMMU 31.4, SeedBench 65.5; †AMD lmms-eval re-run: SEED 65.43, MMMU 30.9, MMBench 44.6 | [arXiv 2408.03326](https://arxiv.org/html/2408.03326v1); †[AMD Instella-VL blog](https://rocm.blogs.amd.com/artificial-intelligence/Instella-BL-1B-VLM/README.html) Table 3 |
| **InternVL2-1B** (0.9B) | 87.3 | 70.5 | 59.8 | 1794.4 (1346.2 P + 448.2 C) | MMMU 36.7, DocVQA 81.7, ChartQA 72.9, AI2D 64.1, MMBench-EN 65.4; AMD re-run: POPE 87.40, TextVQA 69.6, GQA 55.06 | [HF card](https://huggingface.co/OpenGVLab/InternVL2-1B), [InternVL eval docs](https://internvl.readthedocs.io/en/latest/internvl2.0/evaluation.html), AMD blog above |
| **Florence-2-base-ft** (0.23B) / **-large-ft** (0.77B) | – | 63.6 / 73.5 | – | – | VQAv2 79.7 / 81.7 (task-specific fine-tuned, not a chat VLM) | [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao_Florence-2_Advancing_a_Unified_Representation_for_a_Variety_of_Vision_CVPR_2024_paper.pdf) Table 4 |
| **Ternary Bonsai 27B** | no VLM numbers found | – | – | – | 80.49 avg on 15 text benchmarks (95% of FP16); 5.9 GB | [HF card](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf) |

Note on POPE: SmolVLM's paper does not report it; the Sony re-run lists a "POPE" column with values
like 0.056/0.184 that are clearly not accuracy/F1 and should not be quoted.

A realistic target for a 2B ternary LM with 64 image tokens and ~150k SFT samples is the
LLaVA-1.5-7B-era band (POPE ~85, GQA ~60, TextVQA ~45–55 without OCR data), i.e. below FastVLM-0.5B
and InternVL2-1B on TextVQA but far above LLaVaOLMoBitnet1B. Present that honestly against
memory/CPU-speed, not against accuracy alone.

## 4. Likelihood scoring instead of generation ("no-decode" mode)

Prior art is extensive; this is a protocol, not a contribution.

* **Flamingo / OpenFlamingo** score fixed candidates by log-likelihood for classification: "For HatefulMemes, we compute the log-likelihood of completions 'yes' and 'no' and answer with the most likely completion" — [OpenFlamingo, arXiv 2308.01390](https://arxiv.org/html/2308.01390v2); Flamingo [arXiv 2204.14198](https://arxiv.org/abs/2204.14198).
* **SEED-Bench** "answer ranking": "For each choice of a question, we compute the likelihood that an MLLM generates the content of this choice given the question … the choice with the highest likelihood is selected" — [arXiv 2307.16125](https://arxiv.org/pdf/2307.16125); reused in SEED-Bench-2 [arXiv 2311.17092](https://arxiv.org/pdf/2311.17092).
* **lm-evaluation-harness** `output_type: multiple_choice` / `loglikelihood` with `acc` and byte-length-normalised `acc_norm` — [task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md). **lmms-eval** inherits this and ships `seedbench_ppl` (multiple_choice, PPL-based) — [task guide](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/task_guide.md).
* **MMBench** goes the other way: free-form generation + ChatGPT choice matching + CircularEval (all option orders must be right) — [arXiv 2307.06281](https://arxiv.org/abs/2307.06281). POPE ([arXiv 2305.10355](https://arxiv.org/abs/2305.10355)) and MME ([arXiv 2306.13394](https://arxiv.org/abs/2306.13394)) are yes/no and conventionally scored by parsing the generated word.
* **Caveats you must cite:** probability-based MCQ evaluation "inadequately aligns with generation-based prediction" — "Beyond Probabilities" [arXiv 2402.13887](https://arxiv.org/abs/2402.13887) (KnowLLM @ ACL 2024); text answers of instruction-tuned models are more robust than first-token probabilities — "Look at the Text" [arXiv 2404.08382](https://arxiv.org/abs/2404.08382). Therefore report likelihood-scored and generation-scored numbers as separate columns and never compare a likelihood-scored TernaVLM number with a generation-scored baseline number.

What is legitimately ours here is only the *deployment* argument: on a CPU with a ternary LM, scoring
K short candidates is K prefill passes (or one batched pass) with no sampling loop, which is a
measurable latency win for closed-vocabulary tasks. Measure it; do not call it a method.

## 5. CPU / edge deployment

* **bitnet.cpp** ([arXiv 2502.11880](https://arxiv.org/html/2502.11880v1), ACL 2025) — text-only lossless ternary CPU kernels for BitNet b1.58 and TriLMs.
* **llama.cpp** has ternary types `TQ1_0` (1.6875 bpw) and `TQ2_0` (2.0625 bpw) since [PR #8151](https://app.semanticdiff.com/gh/ggerganov/llama.cpp/pull/8151/overview) and a group-64 `Q2_0` added for Bonsai in 2026 ([discussion #22019](https://github.com/ggml-org/llama.cpp/discussions/22019)). Whether mainline `mtmd` works with the `bitnet-b1.58` architecture end-to-end is **[unverified]** — this is a roadmap item, not a given.
* BitVLA reports GPU numbers only (A100: 73 ms latency, 1.4 GB); Bonsai reports phone numbers for a 27B. A CPU/Raspberry-Pi tokens-per-second and RSS table for a 2B ternary VLM versus SmolVLM-256M/500M and Moondream-0.5B would be new data.

## 6. Novelty statement (draft for the paper, honest version)

> Ternary vision-language models are not new: LLaVaOLMoBitnet1B (2024) attached CLIP to a 1B ternary
> OLMo, BitVLA (2025) attached SigLIP-L to BitNet b1.58 2B4T with full-parameter quantization-aware
> training of all latent weights, and PrismML's Ternary Bonsai 27B (2026) ships a ternary Qwen3.6 with
> a 4-bit vision tower. Joint quantization of a frozen weight and a LoRA delta with a straight-through
> estimator is also established for uniform 3/4-bit quantizers (L4Q, LR-QAT). Our contribution is the
> combination and its consequences: (1) a *parameter-efficient* recipe that adapts a natively ternary
> LM to vision by training only a projector and LoRA factors, with the LoRA delta quantized jointly
> with the base under BitNet's absmean ternary / int8 activation scheme, so the exported model is
> bit-for-bit the model that was trained and stays exactly ternary; (2) a controlled comparison of this
> against mixed-precision LoRA (adapter outside the quantizer) and against full latent-weight QAT on the
> same backbone and data, isolating the train/deploy mismatch cost; (3) checkpoints of a few hundred MB
> and a training budget that fits free-tier GPUs; and (4) the first CPU-only deployment and latency
> study of a ternary VLM, including a likelihood-scoring inference mode for closed-vocabulary tasks.

Do **not** claim: "first ternary VLM", "first VLM on BitNet", "nobody has released a ternary VLM",
"novel quantization-aware LoRA", or that likelihood scoring is a new answering paradigm.

## 7. Risks to the claim

1. **BitVLA already released VL-pretrained BitNet-2B4T checkpoints.** If a reviewer sees "ternary VLM on BitNet 2B4T" they will ask why not just use BitVLA's stage-II model. Answer must be quantitative: our GPU-hours (≈35) vs their 10M-sample full QAT, our trainable state size, and a head-to-head on the same benchmarks. If we lose badly on accuracy, the paper is about cost, not quality — frame it that way from the start.
2. **BitLoRA (ESWA 2026) may already do joint ternary quantization of W+BA.** Only the abstract was accessible. Obtain the full text; if it matches, cite it as prior art for the mechanism and keep only the multimodal/CPU contributions.
3. **L4Q / LR-QAT reviewers.** The mechanism in `ternary.py` is Eq. (4) of LR-QAT with a different quantizer. Cite both in the method section and position the method as "LR-QAT-style joint quantization specialised to BitNet's absmean ternary scheme". Consider adopting LR-QAT's integer-domain parametrization as an ablation.
4. **Technical: LoRA deltas may never flip a ternary state.** With per-tensor absmean scale `α = mean|W|`, an entry changes state only when `|s·(BA)_ij|` crosses ~0.5·α (and the scale itself drifts as BA grows). With zero-initialised B and small learning rates, `Q(W + s·BA) == Q(W)` for most entries for many steps; STE still passes gradient, but the deployed model can be indistinguishable from the base. Monitor the fraction of flipped ternary entries per layer during the smoke run; if it is ~0, raise α/lr or move to LR-QAT's `W/α0 + BA` parametrization. This is the single most likely way the method silently fails.
5. **Vision encoder is full precision.** LLaVaOLMoBitnet1B made the same choice and was criticised for calling the model "ternary"; BitVLA ternarised SigLIP-L. Either report the encoder's share of parameters/FLOPs explicitly ("ternary LM, fp16 encoder") or add a Quantize-then-Distill stage.
6. **Benchmark comparability.** Baseline numbers above come from at least five different harnesses/prompts (paper tables, InternVL docs, AMD lmms-eval, Sony lmms-eval, FastVLM tables). Re-evaluate every baseline with lmms-eval yourself; a table mixing sources will be rejected.
7. **Likelihood-vs-generation mismatch.** Reporting only likelihood-scored POPE/GQA invites the "Beyond Probabilities" critique. Report both; use generation numbers as the headline.
8. **Backbone fragility.** BitNet 2B4T fine-tuning issues (#295) show that SFT on the bf16 weights is not turnkey; stage-1 projector-only training must first show that the frozen ternary LM produces sensible captions at all.
9. **Deployment claims.** Until GGUF export + `mtmd` inference is demonstrated on the BitNet architecture, "runs on a CPU" is a plan. Do not put it in the title before it works.
10. **Moving target.** Falcon-E's May-2025 blog listed a "multimodal BitNet VLM" as open; BitVLA appeared in June 2025; Bonsai in July 2026. Assume a BitNet-team or TII ternary VLM paper could appear before submission; keep the contribution centred on the PEFT recipe and the cost/deployment study, which survive that.

## 8. Items to verify before submission

* BitLoRA full text (joint vs separate quantization of the adapter) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417426003106).
* Whether `lxsy/bitvla-siglipL-224px-bf16` / `bitvla-bitsiglipL-224px-bf16` are publicly downloadable (my fetch got HTTP 401) — [GitHub README](https://github.com/ustcwhy/BitVLA).
* FastVLM Table 6 column alignment for the 0.5B rows — [arXiv 2412.13303](https://arxiv.org/html/2412.13303v2).
* Bonsai whitepaper for how the ternary weights were produced — [Bonsai-demo repo](https://github.com/PrismML-Eng/Bonsai-demo).
* llama.cpp mainline `mtmd` + `bitnet-b1.58` architecture end-to-end.
* An arXiv full-text sweep for "ternary"+"vision-language" 2026 papers (the arXiv API rate-limited my queries; keyword search found nothing beyond the items above, but this should be repeated).
