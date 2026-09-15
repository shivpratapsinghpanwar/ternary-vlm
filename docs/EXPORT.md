# Exporting TernaVLM for CPU inference

Research notes, 2026-09-15. Sources were read at these revisions:

* llama.cpp `master` = [`1bc7a5af0`](https://github.com/ggml-org/llama.cpp/commit/1bc7a5af0d14b1fb72f266abbd1237b394187115) (2026-09-15)
* bitnet.cpp = [microsoft/BitNet `main`](https://github.com/microsoft/BitNet), whose `3rdparty/llama.cpp` submodule is the fork
  [isHuangXin/llama.cpp @ `release-bitnet-embedding-0.6b-270m`](https://github.com/isHuangXin/llama.cpp/tree/release-bitnet-embedding-0.6b-270m)
  (HEAD `390c3077`, 2026-07-15). This fork is a recent llama.cpp (has `conversion/`, `tools/mtmd/`).

Everything marked **UNVERIFIED** was read from source but not executed.

## TL;DR

1. **Text model:** export the *latent* merged weights (`merge_all(model.lm, fp=True)`, fp32), convert with stock
   `convert_hf_to_gguf.py --outtype tq2_0` (mainline) or `--outtype f32` + `llama-quantize ... I2_S` (bitnet.cpp).
   The converters' own absmean `weight_quant` then reproduces exactly the ternary tensors we trained with.
   Do **not** feed already-ternary weights to either converter (see pitfall P1).
2. **Vision + projector (mmproj):** use llama.cpp projector type **`lfm2`**, not `idefics3`. The `lfm2` graph is
   SigLIP → 2x2 pixel-unshuffle → LayerNorm → Linear → GELU → Linear, exactly our `Projector`, and its preprocessor
   feeds one image (no tiles) for a 256x256 input. `idefics3` is a single bias-free `Linear` and always emits
   overview + tile (two images). Make the mmproj by presenting our checkpoint as a fake `Lfm2VlForConditionalGeneration`
   and running `convert_hf_to_gguf.py --mmproj` (recipe below), or with a ~60-line gguf-py writer.
3. **Runtime:** mainline llama.cpp has TQ1_0/TQ2_0 kernels on AVX2 and NEON and `llama-mtmd-cli`/`llama-server` with
   `--mmproj`. **But the `bitnet` graph in both mainline and the bitnet.cpp fork uses SiLU-gated FFN, while
   bitnet-b1.58-2B-4T uses ReLU²** (P2). A one-line patch (`LLM_FFN_SILU` → `LLM_FFN_RELU_SQR` in `src/models/bitnet.cpp`)
   plus a vocab override in the converter is needed on mainline. Verify with perplexity against HF before trusting any GGUF.
4. Expected decode speed for a 2.4B ternary LM on a laptop CPU: ~34 tok/s (29 ms/token, i7-13800H, 8 threads, bitnet.cpp,
   Microsoft's number). TQ2_0 on mainline should be in the same ballpark (memory-bound); UNVERIFIED for this model.

---

## 1. Does mainline llama.cpp support `model_type: bitnet`?

**Yes, partially.** Converter class: [`conversion/bitnet.py::BitnetModel`](https://github.com/ggml-org/llama.cpp/blob/master/conversion/bitnet.py),
registered for `BitnetForCausalLM` / `BitNetForCausalLM` (`conversion/__init__.py` line 35-36), `model_arch = MODEL_ARCH.BITNET`
(GGUF arch string `"bitnet"`). The `@ModelBase.example("microsoft/bitnet-b1.58-2B-4T")` decorator on that class is just an example label.

What it does:

* `weight_quant()` = per-tensor absmean: `scale = mean|W|`, `round(W/scale).clamp(-1,1)*scale`, applied to q/k/v/o/gate/up/down
  in fp32 (same formula as our `ternavlm/ternary.py::ternary_quant`, eps 1e-5 in both).
* `--outtype tq1_0 | tq2_0` (`convert_hf_to_gguf.py` line 60) stores those tensors as `GGML_TYPE_TQ1_0` (34) / `TQ2_0` (35);
  norms/1-D stay F32, token embedding stays F16 (that is why the official TQ files are ~1.8 GB; run `llama-quantize` afterwards, which
  puts `token_embd` at Q4_K and `output` at Q6_K — [PR #8151](https://github.com/ggml-org/llama.cpp/pull/8151), merged 2024-09-06).
* If the HF config carries `quantization_config.quant_method == "bitnet"` (the packed `microsoft/bitnet-b1.58-2B-4T` repo), `base.py::dequant_bitnet`
  unpacks 2-bit weights and divides by `weight_scale` first. Our export has no `quantization_config` (`vlm.py` strips it), so this path is not taken.
* GGUF tensors for arch `bitnet` (`gguf-py/gguf/constants.py` ~line 4075): `token_embd, output_norm, attn_norm, attn_q/k/v/out, ffn_norm,
  ffn_gate/up/down, attn_sub_norm, ffn_sub_norm`. HF → GGUF names (`tensor_mapping.py` 1167-1172):
  `model.layers.{i}.self_attn.inner_attn_ln` → `blk.{i}.attn_sub_norm`, `model.layers.{i}.mlp.ffn_layernorm` → `blk.{i}.ffn_sub_norm`.
  **UNVERIFIED:** the transformers `BitNetForCausalLM` (used by the `-bf16` repo) names these `self_attn.attn_sub_norm` / `mlp.ffn_sub_norm`;
  check the saved `state_dict` keys and add the alias to the mapping (or rename at export) if the converter says "Can not map tensor".

Ternary kernels in mainline (all verified in source):

| type | x86 | ARM | file |
|---|---|---|---|
| `TQ1_0` | `#if defined(__AVX2__)` | `#if defined(__ARM_NEON)` | `ggml/src/ggml-cpu/arch/{x86,arm}/quants.c` (`ggml_vec_dot_tq1_0_q8_K`) |
| `TQ2_0` | `#if defined(__AVX2__)` | `#if defined(__ARM_NEON)` | same (`ggml_vec_dot_tq2_0_q8_K`), generic scalar fallback in `ggml-cpu/quants.c` |
| `I2_S` (36), `TL1` (38), `TL2` (42) | – | – | **only in the bitnet.cpp fork** (`ggml-bitnet-compute.c`, `src/ggml-bitnet-mad.cpp`; `I2_S` has AVX2 and NEON/dotprod branches) |

Notes: no `repack.cpp` GEMM path exists for TQ types (only the vec_dot), so prompt processing is slower per weight than Q4_0; TQ2_0
needs row length % 256 == 0 (2560 and 6912 are fine). The official `ggml-model-i2_s.gguf` **cannot** be loaded by mainline: type id 36 collides
with the removed `IQ4_NL_4_4` ([issue #12997](https://github.com/ggml-org/llama.cpp/issues/12997)). CUDA TQ2_0 exists
([PR #11183](https://github.com/ggml-org/llama.cpp/pull/11183)) but is irrelevant here.

### Two blockers on mainline for *this* model

* **P2 — activation.** `src/models/bitnet.cpp` line 132 builds the FFN with `LLM_FFN_SILU, LLM_FFN_PAR` and `load_arch_hparams` reads only
  the RMS eps. bitnet-b1.58-2B-4T has `hidden_act: "relu2"` (config.json of both the packed and `-bf16` repos; transformers
  `BitNetMLP`: `down(ffn_sub_norm(relu²(gate(x)) * up(x)))`). The bitnet.cpp fork's `src/models/bitnet.cpp` is identical (SiLU) and
  `llama_model_bitnet_b158` only overrides `load_arch_hparams` (adds `case 30: LLM_TYPE_2B // bitnet2b_2501`). I could not find ReLU²
  anywhere in either tree. Fix: `LLM_FFN_SILU` → `LLM_FFN_RELU_SQR` on that line; `build_ffn` multiplies by the `up` branch afterwards when
  `type_gate == LLM_FFN_PAR` (`src/llama-graph.cpp` ~1880-1925), giving `relu(gate)² * up`. **UNVERIFIED that either binary currently produces
  sane text for 2B-4T; measure perplexity first (section 6).**
* **Vocab.** `BitnetModel.set_vocab()` calls `_set_vocab_sentencepiece()`, which raises `FileNotFoundError` without `tokenizer.model`. The
  2B-4T repos ship only `tokenizer.json` (LLaMA-3 BPE). Override to `_set_vocab_gpt2()`. The pre-tokenizer hash check should resolve to
  `llama-bpe` if the tokenizer.json is byte-identical to Meta-Llama-3's (UNVERIFIED; if it fails, add the hash to
  `conversion/base.py::get_vocab_base_pre` or pass `--vocab-pre llama-bpe` if your version has it). Adding `<image>` as a special token does not
  change the hash (it is computed over a fixed sample string). The bitnet.cpp converter (`utils/convert-hf-to-gguf-bitnet.py`) already falls back
  sentencepiece → llama_hf → gpt2.

## 2. Does bitnet.cpp support multimodal?

**Not documented, but the code is there.** microsoft/BitNet's README, `setup_env.py` and `CMakeLists.txt` mention nothing about mtmd/llava/vision.
However its submodule fork contains `tools/mtmd/` with `PROJECTOR_TYPE_IDEFICS3` and `PROJECTOR_TYPE_LFM2` (`tools/mtmd/clip-impl.h` lines 340-407 of
the fork), and `setup_env.py` configures with `-DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON`, and `tools/CMakeLists.txt`
unconditionally does `add_subdirectory(mtmd)`. So `build/bin/llama-mtmd-cli` and `llama-server --mmproj` should exist after `setup_env.py`
(UNVERIFIED). mtmd is arch-agnostic on the text side; the `I2_S` text GGUF + a normal mmproj GGUF should work. Same P2 caveat applies.

## 3. mtmd: SigLIP/SigLIP2 encoders and pixel-shuffle projectors

`tools/mtmd/README.md` lists SmolVLM/SmolVLM2 (Idefics3) among models converted with `convert_hf_to_gguf.py --mmproj`:

```
python convert_hf_to_gguf.py HuggingFaceTB/SmolVLM-256M-Instruct --mmproj --outtype f16 --outfile mmproj-smolvlm.gguf
```

Converter: [`conversion/smolvlm.py::SmolVLMModel(MmprojModel)`](https://github.com/ggml-org/llama.cpp/blob/master/conversion/smolvlm.py)
registered for `Idefics3ForConditionalGeneration` / `SmolVLMForConditionalGeneration`; writes `clip.projector_type = "idefics3"`,
`clip.vision.projector.scale_factor = config.scale_factor` (2), `clip.use_gelu = true`, `clip.vision.preproc_image_size = longest_edge`,
keeps only tensors matching `vision_tower|vision_model|model.connector`.

Graph (`tools/mtmd/models/siglip.cpp`, chosen for GEMMA3/IDEFICS3/LFM2/JANUS_PRO/PHI4 in `clip.cpp` ~line 937):
`build_vit` (learned pos-embd, optional `v.pre_ln`, per-layer `ln1`/attn/`ln2`/ffn, `v.post_ln`; `clip.use_gelu=true` → `ggml_gelu`
= tanh approximation = SigLIP's `gelu_pytorch_tanh`), then per projector type:

| projector | after ViT | preprocessing (`tools/mtmd/mtmd-image.cpp`) | verdict for us |
|---|---|---|---|
| `idefics3` | `build_patch_merge_permute(n_merge)` → `mm.model.fc.weight` (one Linear, no bias) | `mtmd_image_preprocessor_idefics3`: resize longest edge → round up to tile multiples → **always** overview + tiles; grid 1x1 still emits overview + 1 tile (128 tokens), wrapped with `<fake_token_around_image>`, `<global-img>`, `<row_1_col_1>` (looked up in our vocab → `LLAMA_TOKEN_NULL`) | wrong projector shape, wrong image count |
| `lfm2` | `build_patch_merge_permute(n_merge)` → optional `mm.input_norm.{weight,bias}` (`ggml_norm`, eps 1e-5) → `build_ffn(mm.1, GELU, mm.2)` | `mtmd_image_preprocessor_lfm2`: single image if `h*w <= image_max_pixels*2`, resized aspect-preserving into `[image_min_pixels, image_max_pixels]` aligned to 32 px; else tiles of 512 + thumbnail. `set_limit_image_tokens(64, 256)` → min 256², max 512² pixels; CLI `--image-min-tokens/--image-max-tokens` override. Adds text `<|image_start|>` … `<|image_end|>` around the image (`mtmd.cpp` ~846) | **matches our projector exactly**; feed 256x256 |
| `internvl` | pixel shuffle → LayerNorm → Linear → GELU → Linear (`mm.model.mlp.{0,1,3}`) | llava-uhd tiles | needs a CLS token (`GGML_ASSERT(model.class_embedding)`), SigLIP has none |
| `gemma3` | avg-pool → RMSNorm → Linear | fixed size | wrong projector |

`build_patch_merge_permute` (`clip.cpp` ~897) produces feature order `(dy*s + dx)*C + c` and row-major tokens — **identical to
`ternavlm/ops.py::pixel_shuffle`** (checked numerically with a 4x4x3 grid). No column permutation of `net.1.weight` is needed.

### mmproj GGUF keys and tensor names (what `clip.cpp` reads for `lfm2`)

Keys (`tools/mtmd/clip-impl.h`, `gguf-py/gguf/gguf_writer.py`):

| key | value for TernaVLM | writer call |
|---|---|---|
| `general.type` | `"mmproj"` | `add_type(GGUFType.MMPROJ)` |
| `clip.has_vision_encoder` | true | `add_clip_has_vision_encoder` |
| `clip.projector_type` | `"lfm2"` | `add_clip_projector_type` |
| `clip.vision.image_size` / `patch_size` | 256 / 16 | `add_vision_image_size`, `add_vision_patch_size` |
| `clip.vision.embedding_length` / `feed_forward_length` / `block_count` / `attention.head_count` | 768 / 3072 / 12 / 12 | `add_vision_*` |
| `clip.vision.attention.layer_norm_epsilon` | 1e-6 (SigLIP2) | `add_vision_attention_layernorm_eps` |
| `clip.vision.projection_dim` | 2560 (LM hidden) | `add_vision_projection_dim` |
| `clip.vision.projector.scale_factor` | 2 | `add_vision_projector_scale_factor` |
| `clip.use_gelu` | true | `add_vision_use_gelu` |
| `clip.vision.image_mean` / `image_std` | `[0.5,0.5,0.5]` each (SigLIP processor) | `add_vision_image_mean/std` |

Tensors (HF names as saved by `SiglipVisionModel.save_pretrained` in `export/ternavlm/vision/` and by our `Projector`; GGUF names from
`gguf-py/gguf/constants.py` ~1653-1690 and `clip-impl.h`):

| HF tensor (ours) | GGUF tensor | notes |
|---|---|---|
| `vision_model.embeddings.patch_embedding.{weight,bias}` | `v.patch_embd.weight` (768,3,16,16 conv), `v.patch_embd.bias` | keep as conv layout; forced F16/F32 by converter |
| `vision_model.embeddings.position_embedding.weight` (256,768) | `v.position_embd.weight` | 16x16 grid; `lfm2` interpolates if the image is not 256x256 |
| `vision_model.encoder.layers.{i}.layer_norm1.{w,b}` | `v.blk.{i}.ln1.{weight,bias}` | |
| `…self_attn.{q,k,v,out}_proj.{w,b}` | `v.blk.{i}.attn_{q,k,v,out}.{weight,bias}` | |
| `…layer_norm2.{w,b}` | `v.blk.{i}.ln2.{weight,bias}` | |
| `…mlp.fc1.{w,b}` / `…mlp.fc2.{w,b}` | `v.blk.{i}.ffn_up.*` / `v.blk.{i}.ffn_down.*` | |
| `vision_model.post_layernorm.{w,b}` | `v.post_ln.{weight,bias}` | we use `last_hidden_state` = after post_layernorm, matches |
| `vision_model.head.*` (attention-pool head) | **drop** | unused by us, unmapped in converter |
| `projector.net.0.{weight,bias}` (LayerNorm 3072) | `mm.input_norm.{weight,bias}` | eps 1e-5 both sides |
| `projector.net.1.{weight,bias}` (3072→2048) | `mm.1.{weight,bias}` | |
| `projector.net.3.{weight,bias}` (2048→2560) | `mm.2.{weight,bias}` | |

For reference, `idefics3` expects only `mm.model.fc.weight` (HF `model.connector.modality_projection.proj.weight`).

## 4. Feasibility and recommended recipe

### Options considered

* (a) Fake **Idefics3** HF config → stock converter. Not viable: projector shape and preprocessing mismatch (table above).
* (a') Fake **LFM2-VL** HF directory → stock `convert_hf_to_gguf.py --mmproj`. `conversion/lfm2.py::LFM2VLModel` writes exactly the keys we
  need (`image_size` forced to 256, `scale_factor = downsample_factor`, eps from `vision_config.layer_norm_eps`, `use_gelu`). The HF names it expects
  are `model.vision_tower.vision_model.*` and `model.multi_modal_projector.{layer_norm,linear_1,linear_2}`. One trap: `modify_tensors` does
  `patch_embedding.weight.view(out,16,16,3).permute(0,3,1,2)` because LFM2-VL's SigLIP2-NaFlex stores a Linear patch embedding; our Conv2d weight
  must be pre-flattened as `w.permute(0,2,3,1).reshape(768, 768)` so the converter restores the conv layout. **Recommended** (zero converter patches,
  ~40 lines of export code).
* (b) Custom gguf-py writer for the mmproj: same tensors/keys as the table, ~60 lines using `gguf.GGUFWriter`. Equivalent result; use it if (a')
  fights the converter's tensor-name checks. Text GGUF via the bitnet converter either way.

### Step 0 — decide these *before* stage 1 (cheap now, painful after training)

1. `Projector`: use `nn.GELU(approximate="tanh")` — `build_ffn(..., FFN_GELU)` uses `ggml_gelu` (tanh approx). Exact-erf GELU would be a small
   silent mismatch.
2. Add `<|image_start|>` and `<|image_end|>` as special tokens and wrap the 64 `<image>` slots with them in `data.py`; `lfm2` inserts that text
   around the image at inference and otherwise it gets BPE-split into junk tokens. (Alternative: keep the format and strip them from the prompt
   template — not possible from the CLI.)
3. Keep training preprocessing = plain resize to 256x256 (SigLIP processor). At inference always pre-resize to 256x256 client-side; then
   `lfm2` preprocessing is a no-op (`256*256` is within `[65536, 262144]`, aligned to 32, no tiling), pos-embd interpolation is skipped
   (`resize_position_embeddings` returns early for a 16x16 grid) and `clip_n_output_tokens` = `(256/32)² = 64`.
4. Make sure `config.vocab_size == len(tokenizer)` after `resize_token_embeddings` (128257) so the converter's vocab and `token_embd` agree.

### Step 1 — export (changes to `scripts/export.py`)

```python
n = merge_all(model.lm, fp=True)          # latent W + s*BA in fp32; the GGUF converter re-applies the same absmean quantizer
# sanity: ternary_quant(latent) must equal the ternary tensors the model trained with (merge() output)
model.lm.save_pretrained(out/"lm"); model.tokenizer.save_pretrained(out/"lm")   # config.json: architectures=[BitNetForCausalLM], model_type=bitnet, no quantization_config
```

Then write `out/mmproj_hf/` as a fake LFM2-VL checkpoint:

```
config.json: {"architectures": ["Lfm2VlForConditionalGeneration"], "model_type": "lfm2_vl",
              "downsample_factor": 2, "vision_feature_layer": -1, "projector_hidden_size": 2048,
              "text_config": {"hidden_size": 2560},
              "vision_config": {"model_type": "siglip2_vision_model", "hidden_size": 768, "intermediate_size": 3072,
                                "num_hidden_layers": 12, "num_attention_heads": 12, "patch_size": 16, "image_size": 256,
                                "layer_norm_eps": 1e-6, "hidden_act": "gelu_pytorch_tanh", "num_patches": 256}}
preprocessor_config.json: {"image_mean": [0.5,0.5,0.5], "image_std": [0.5,0.5,0.5]}
model.safetensors:
  model.vision_tower.vision_model.<everything except head.*>     (patch_embedding.weight flattened as described above)
  model.multi_modal_projector.layer_norm.{weight,bias}  <- projector.net.0
  model.multi_modal_projector.linear_1.{weight,bias}    <- projector.net.1
  model.multi_modal_projector.linear_2.{weight,bias}    <- projector.net.3
```

### Step 2 — GGUF conversion

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && pip install -r requirements.txt
# text model (mainline): subclass/patch BitnetModel.set_vocab -> _set_vocab_gpt2, then
python convert_hf_to_gguf.py ../export/ternavlm/lm --outtype tq2_0 --outfile ternavlm-tq2_0-raw.gguf
cmake -B build -DGGML_NATIVE=ON && cmake --build build --config Release -j     # after the P2 one-line patch
./build/bin/llama-quantize ternavlm-tq2_0-raw.gguf ternavlm-tq2_0.gguf TQ2_0    # embeddings -> Q4_K/Q6_K, ~0.7 GB total
# mmproj
python convert_hf_to_gguf.py ../export/ternavlm/mmproj_hf --mmproj --outtype f16 --outfile mmproj-ternavlm-f16.gguf
# run (pre-resize the image to 256x256 first)
./build/bin/llama-mtmd-cli -m ternavlm-tq2_0.gguf --mmproj mmproj-ternavlm-f16.gguf --image test256.jpg \
    -p "User: <__media__>\nDescribe the image. Assistant:" -t 8 --image-min-tokens 64 --image-max-tokens 64
./build/bin/llama-server -m ternavlm-tq2_0.gguf --mmproj mmproj-ternavlm-f16.gguf --no-mmproj-offload
```

bitnet.cpp variant (I2_S kernels, `git clone --recursive https://github.com/microsoft/BitNet`): run its
`utils/convert-hf-to-gguf-bitnet.py ../export/ternavlm/lm --outtype f32` then `llama-quantize [--token-embedding-type f16] f32.gguf i2s.gguf I2_S 1`
(this is what `setup_env.py -q i2_s` does), and the same mmproj file with the fork's `build/bin/llama-mtmd-cli`. Its converter's `weight_quant`
is the same absmean, so export the latent weights here too; `quantize_i2_s` (`src/ggml-bitnet-mad.cpp`) uses per-tensor absmax, so exact-ternary
input is stored losslessly. GGUF arch written by that converter is `bitnet-b1.58`. TL1/TL2 need `--outtype tl1|tl2` and are ARM/x86-specific.

### Pitfalls

* **P1 — double absmean.** Both converters call `weight_quant` (absmean) on every projection. Re-quantizing an already ternary tensor
  `{-a,0,a}` gives `{-a·f, 0, a·f}` with `f` = fraction of non-zeros (measured 0.69 on a random matrix): Q/K logits shrink by `f²`, `o_proj`/`down_proj`
  residual contributions by `f`. So either export the latent fp32 weights (recommended; `ternary_quant` and the converter use the identical formula,
  differences only in fp rounding at the `±0.5/±1.5` boundaries — rare, UNVERIFIED how many flips) or export ternary weights and override
  `weight_quant` to identity. Do **not** export ternary weights to the stock converter.
* **P2 — ReLU² vs SiLU** in `src/models/bitnet.cpp` (both trees). One-line patch; verify with perplexity.
* **P3 — vocab**: sentencepiece-only `set_vocab` in mainline `BitnetModel`.
* **P4 — GELU flavour** (projector `approximate="tanh"`) and **image wrapper tokens** (`<|image_start|>`/`<|image_end|>`), see Step 0.
* **P5 — `idefics3` always double-encodes** (overview + tile) and expects SmolVLM special tokens; don't use it.
* **P6 — dynamic resolution**: without `--image-min-tokens 64 --image-max-tokens 64` (or client-side 256x256 resize) `lfm2` will feed larger /
  non-square images (up to 512²) and tile beyond that; the LM never saw that distribution. `set_limit_image_tokens` honours the CLI override
  (`clip-model.h` line 190).
* **P7 — sub-norm tensor names** (`inner_attn_ln`/`ffn_layernorm` in the mapping vs `attn_sub_norm`/`ffn_sub_norm` in transformers), see section 1.
* **P8 — TQ2_0 scale is fp16** per 256-block (`quantize_row_tq2_0_ref`): `a ≈ 1e-2` is represented to ~1e-3 relative; negligible but not bit-exact.
* **P9 — chat template**: the official Microsoft GGUF ships a wrong template (tdh111 model card); our export writes `tokenizer_config.json` from
  our tokenizer, so pass the `User:/Assistant:` format explicitly (`-p`) or set `chat_template` before export.

## 5. Fallback: PyTorch CPU, and expected speed

* transformers `BitNetForCausalLM` runs the `-bf16` weights through `AutoBitLinear` (online absmean + int8 activations, but computed in bf16/fp32
  matmuls). Microsoft: "do NOT expect performance efficiency gains when using this model with the standard transformers library"
  ([model card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)). Our exported HF checkpoint (plain `nn.Linear`, no `quantization_config`)
  can be run with `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` (fbgemm/qnnpack int8 GEMM); expect low
  single-digit tok/s for 2.4B params on a laptop (UNVERIFIED estimate, memory-bound ~2.5 GB int8 weights). This is a correctness reference and a
  demo fallback, not the deployment path. The full `TernaVLM` module already runs end-to-end on CPU for that purpose.
* Published ternary CPU numbers:
  * BitNet b1.58 2B4T report ([arXiv 2504.12285](https://arxiv.org/abs/2504.12285), Table 1): **29 ms/token (~34 tok/s)** decoding, 0.4 GB
    non-embedding memory, on a Surface Laptop Studio 2 (i7-13800H, 8 threads, bitnet.cpp, 128 generated tokens); LLaMA-3.2-1B 48 ms, Qwen2.5-1.5B 65 ms,
    SmolLM2-1.7B 67 ms with llama.cpp.
  * bitnet.cpp report ([arXiv 2502.11880](https://arxiv.org/html/2502.11880), Table 7): 3.8B ternary on i7-13700H: Q4_0 16.3, TQ1_0 26.6,
    TL2 35.4, I2_S 35.0 tok/s; on M2 Ultra: 71.9 / 73.1 / 92.1 / 91.7 tok/s. README claims 2.37-6.17x over llama.cpp on x86, 1.37-5.07x on ARM.
  * Mainline TQ2_0 micro-benchmarks ([PR #8151](https://github.com/ggml-org/llama.cpp/pull/8151)): ~2x Q4_K throughput on an AVX2 Core m3-8100Y,
    15.8 GB/s-equivalent on a Cortex-A72 (Raspberry Pi 4 class) with NEON.
  * Rough expectation for TernaVLM on a modern laptop CPU (UNVERIFIED): 25-40 tok/s decode with TQ2_0 or I2_S; image encode = SigLIP2-base
    (93M params, 256 tokens) at f16 on CPU is a few hundred ms.

## 6. Verification checklist / open questions

1. Perplexity of the converted text GGUF (`llama-perplexity -m … -f wiki.test.raw`) vs HF fp32 on the same text, before and after the ReLU²
   patch. Also compare logits on one prompt (`llama-eval-callback` / `--verbose-prompt`). This settles P2.
2. Confirm the fork's `setup_env.py` build actually produces `llama-mtmd-cli` and that its mtmd loads an `lfm2` mmproj (UNVERIFIED).
3. Confirm `tokenizer.json` hash → `llama-bpe`, and the sub-norm key names in our saved `state_dict` (P3, P7).
4. Bit-exactness of P1: dequantize the TQ2_0 tensors with `gguf-py` (`GGUFReader` + `gguf.quants.dequantize`) and compare with `merge()` output.
5. Whether upstream would take the two small fixes (ReLU² via `hidden_act`/`<arch>.hidden_activation` — `llm_ffn_op_type_from_string` in
   `src/llama-model.cpp` does not know `relu2` — and gpt2 vocab fallback in `BitnetModel`), which would make TernaVLM run on stock llama.cpp.
6. Whether `lfm2`'s `<|image_start|>`/`<|image_end|>` handling has a way to be disabled in `llama-server` (it does not appear to; hence Step 0.2).
