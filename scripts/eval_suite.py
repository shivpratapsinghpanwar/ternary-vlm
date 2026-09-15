"""
Multi-benchmark evaluation over *streamed* Hugging Face datasets, for TernaVLM checkpoints and for
off-the-shelf HF baselines, so one script produces every row of the paper's comparison table.

    # ours
    python scripts/eval_suite.py --model ternavlm --ckpt ckpt/stage2/latest.pt \
        --benchmarks pope,textvqa,gqa,sqa,mme --n 200 --mode score

    # baselines (same subset, same prompts, same metrics)
    python scripts/eval_suite.py --model hf --hf-name HuggingFaceTB/SmolVLM-256M-Instruct --benchmarks pope,sqa,mme --n 200
    python scripts/eval_suite.py --model hf --hf-name HuggingFaceTB/SmolVLM-500M-Instruct ...
    python scripts/eval_suite.py --model hf --hf-name llava-hf/llava-onevision-qwen2-0.5b-ov-hf ...

Benchmarks (all streamed, capped with --n; nothing is downloaded in full):
    pope     lmms-lab/POPE            test        yes/no accuracy, precision/recall/F1 (yes = positive class)
    textvqa  lmms-lab/textvqa         validation  VQA soft accuracy  min(#matching answers / 3, 1) over 10 answers
    gqa      lmms-lab/GQA             testdev balanced  exact match after VQA normalisation
    sqa      lmms-lab/ScienceQA-IMG   test        multiple-choice accuracy (option letters)
    mme      lmms-lab/MME             test        perception / cognition scores = sum over categories of acc + acc+

--mode generate  : decode text and parse it (works for everything).
--mode score     : closed-form, no decoding: rank the candidate answers by teacher-forced log-likelihood
                   (pope/mme: "Yes"/"No"; sqa: option letters). textvqa and gqa are open-ended and always
                   generate, whatever --mode says.

Dataset notes (verified 2026-09-15 against https://huggingface.co/api/datasets/<name>; see BENCHMARKS):
  * lmms-lab/{POPE,textvqa,GQA,MME} were renamed to lmms-lab-encoder/*; the old ids still resolve through the
    Hub redirect. Override with `--override <bench>.dataset=<id>` if that ever breaks.
  * GQA keeps questions and images in different configs (`testdev_balanced_instructions` has `imageId`,
    `testdev_balanced_images` has `id` + `image`). We first stream the question subset, then stream the image
    config with decoding disabled and keep only the referenced images (398 images in the whole testdev set).
  * MME rows come in adjacent pairs (one "Yes" and one "No" question on the same image, same `question_id`).
    The official metric per category is acc (%) + acc+ (% of images whose *both* questions are right), max 200
    per category; perception score sums 10 categories (max 2000), cognition sums 4 (max 800). With --n the
    sums only cover the categories that were sampled (listed in the JSON), so compare subsets like-for-like.
    MME defaults to `--spread` so a small --n still touches every category.
  * ScienceQA: lmms-lab/ScienceQA-IMG is already filtered to rows with an image; derek-thomas/ScienceQA works
    too (rows without image are skipped), pass `--override sqa.dataset=derek-thomas/ScienceQA`.

Output: results/<name>.json plus one markdown table row on stdout.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ============================================================================================
# Benchmark registry.  Column names were verified on 2026-09-15 with
#   curl https://huggingface.co/api/datasets/<name>   (cardData.dataset_info.features)
#   curl https://datasets-server.huggingface.co/rows?dataset=<new id>&config=..&split=..
# `size` is the split length from the same API (used only for --spread).  Everything here can be
# overridden from the CLI:  --override gqa.dataset=lmms-lab-encoder/GQA  --override pope.cols.answer=label
# ============================================================================================
BENCHMARKS: dict[str, dict] = {
    "pope": {
        "dataset": "lmms-lab/POPE", "config": None, "split": "test", "size": 9000,
        # features: id, question_id, question, answer ("yes"/"no"), image_source, image, category
        "cols": {"image": "image", "question": "question", "answer": "answer", "category": "category"},
        "spread": False,
    },
    "textvqa": {
        "dataset": "lmms-lab/textvqa", "config": None, "split": "validation", "size": 5000,
        # features: image_id, question_id, question, question_tokens, image, image_width, image_height,
        #           flickr_original_url, flickr_300k_url, answers (list[str], 10 entries), image_classes, set_name, ocr_tokens
        "cols": {"image": "image", "question": "question", "answers": "answers"},
        "spread": False,
    },
    "gqa": {
        "dataset": "lmms-lab/GQA", "config": "testdev_balanced_instructions", "split": "testdev", "size": 12578,
        # instructions features: id, imageId, question, answer, fullAnswer, isBalanced, groups, entailed, equivalent,
        #                        types, annotations, semantic, semanticStr
        # images config `testdev_balanced_images` (398 rows): id, image      (id == instructions.imageId)
        "images_config": "testdev_balanced_images",
        "cols": {"question": "question", "answer": "answer", "image_id": "imageId", "img_key": "id", "image": "image"},
        "spread": False,
    },
    "sqa": {
        "dataset": "lmms-lab/ScienceQA-IMG", "config": None, "split": "test", "size": 2017,
        # features: image, question, choices (list[str]), answer (int8 index), hint, task, grade, subject, topic,
        #           category, skill, lecture, solution
        "cols": {"image": "image", "question": "question", "choices": "choices", "answer": "answer", "hint": "hint"},
        "spread": False,
    },
    "mme": {
        "dataset": "lmms-lab/MME", "config": None, "split": "test", "size": 2374,
        # features: question_id (e.g. "OCR/0002.jpg", shared by the yes/no pair), image, question
        #           (already ends with "Please answer yes or no."), answer ("Yes"/"No"), category
        "cols": {"image": "image", "question": "question", "answer": "answer", "category": "category", "pair_id": "question_id"},
        "spread": True,
    },
}

MME_PERCEPTION = ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"]
MME_COGNITION = ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]

PROMPT_YES_NO = "{q}\nAnswer yes or no."
PROMPT_SHORT = "{q}\nAnswer the question using a single word or phrase."
PROMPT_MC = "Answer with the option's letter from the given choices directly."
LETTERS = "ABCDEFGH"


# ============================================================================================
# Pure metric helpers (unit-tested in tests/test_eval_metrics.py; no torch needed)
# ============================================================================================
_CONTRACTIONS = {
    "dont": "don't", "doesnt": "doesn't", "didnt": "didn't", "isnt": "isn't", "arent": "aren't", "wasnt": "wasn't",
    "werent": "weren't", "cant": "can't", "couldnt": "couldn't", "wont": "won't", "wouldnt": "wouldn't",
    "shouldnt": "shouldn't", "hasnt": "hasn't", "havent": "haven't", "hadnt": "hadn't", "im": "i'm", "ive": "i've",
    "youre": "you're", "theyre": "they're", "whats": "what's", "thats": "that's", "theres": "there's", "lets": "let's",
    "hes": "he's", "shes": "she's", "whos": "who's", "youve": "you've", "wheres": "where's", "hows": "how's",
}
_NUMBER_WORDS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                 "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_PUNCT = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
_PERIOD_STRIP = re.compile(r"\.(?!\d)")  # drop periods except decimal points ("1.5" survives, "yes." -> "yes")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def normalize_vqa(text: str) -> str:
    """Compact port of the official VQA v2 evaluation normaliser: lowercase, punctuation rules
    (keep '.' and ',' inside numbers), number words -> digits, drop articles, canonical contractions."""
    out = text.strip().lower().replace("\n", " ").replace("\t", " ")
    for p in _PUNCT:
        if (p + " " in out or " " + p in out) or _COMMA_STRIP.search(out) is not None:
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _PERIOD_STRIP.sub("", out)
    words = []
    for w in out.split():
        w = _NUMBER_WORDS.get(w, w)
        if w in _ARTICLES:
            continue
        words.append(_CONTRACTIONS.get(w, w))
    return " ".join(words)


def vqa_soft_accuracy(pred: str, answers: list[str]) -> float:
    """Standard VQA soft score: min(#gold answers matching the prediction / 3, 1)."""
    p = normalize_vqa(pred)
    matches = sum(normalize_vqa(a) == p for a in answers)
    return min(matches / 3.0, 1.0)


def exact_match(pred: str, gold: str) -> bool:
    return normalize_vqa(pred) == normalize_vqa(gold)


def parse_yes_no(text: str) -> str:
    """'yes' / 'no' by whichever word appears first; defaults to 'no' (same convention as eval_pope.py)."""
    t = text.lower()
    m = re.search(r"\b(yes|no)\b", t)
    if m:
        return m.group(1)
    return "yes" if "yes" in t else "no"


def parse_choice(text: str, options: list[str]) -> int:
    """Index of the option letter found in a generated answer, else index of the option whose text
    matches, else -1."""
    n = len(options)
    t = text.strip()
    m = re.match(r"^[\s\(\[\*\"']*([A-H])(?![A-Za-z])", t)
    if m and LETTERS.index(m.group(1)) < n:
        return LETTERS.index(m.group(1))
    m = re.search(r"(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]?([A-H])(?![A-Za-z])", t, flags=re.I)
    if m and LETTERS.index(m.group(1).upper()) < n:
        return LETTERS.index(m.group(1).upper())
    tn = normalize_vqa(t)
    for i, o in enumerate(options):
        if tn == normalize_vqa(o):
            return i
    for i, o in enumerate(options):
        if normalize_vqa(o) and normalize_vqa(o) in tn:
            return i
    return -1


def pope_metrics(preds: list[str], golds: list[str]) -> dict:
    """Accuracy, precision, recall, F1 with 'yes' as the positive class, plus the yes-ratio of the predictions."""
    tp = sum(p == "yes" and g == "yes" for p, g in zip(preds, golds))
    fp = sum(p == "yes" and g == "no" for p, g in zip(preds, golds))
    fn = sum(p == "no" and g == "yes" for p, g in zip(preds, golds))
    n = len(preds)
    correct = sum(p == g for p, g in zip(preds, golds))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": n, "accuracy": correct / n if n else 0.0, "precision": precision, "recall": recall, "f1": f1,
            "yes_ratio": sum(p == "yes" for p in preds) / n if n else 0.0}


def mme_scores(records: list[dict]) -> dict:
    """records: [{"pair_id", "category", "correct"}].  Per category: acc = % correct questions,
    acc+ = % of complete pairs (both questions of an image) fully correct, score = acc + acc+ (max 200).
    perception_score / cognition_score sum the categories *present in records*."""
    cats: dict[str, dict] = {}
    for r in records:
        c = cats.setdefault(r["category"], {"n": 0, "correct": 0, "pairs": {}})
        c["n"] += 1
        c["correct"] += int(r["correct"])
        c["pairs"].setdefault(r["pair_id"], []).append(bool(r["correct"]))
    out = {"categories": {}}
    for name, c in cats.items():
        full = [v for v in c["pairs"].values() if len(v) >= 2]
        acc = 100.0 * c["correct"] / c["n"]
        acc_plus = 100.0 * sum(all(v) for v in full) / len(full) if full else 0.0
        out["categories"][name] = {"n": c["n"], "pairs": len(full), "acc": acc, "acc_plus": acc_plus, "score": acc + acc_plus}
    out["perception_categories"] = [c for c in MME_PERCEPTION if c in cats]
    out["cognition_categories"] = [c for c in MME_COGNITION if c in cats]
    out["perception_score"] = sum(out["categories"][c]["score"] for c in out["perception_categories"])
    out["cognition_score"] = sum(out["categories"][c]["score"] for c in out["cognition_categories"])
    out["perception_max"] = 200 * len(out["perception_categories"])
    out["cognition_max"] = 200 * len(out["cognition_categories"])
    n = len(records)
    out["n"] = n
    out["accuracy"] = sum(r["correct"] for r in records) / n if n else 0.0
    return out


# ============================================================================================
# Model adapters:  predict(image, prompt, max_new_tokens) -> str ;  score_options(image, prompt, options) -> int
# ============================================================================================
class Adapter:
    name: str = "?"

    def predict(self, image, prompt: str, max_new_tokens: int) -> str:
        raise NotImplementedError

    def score_options(self, image, prompt: str, options: list[str]) -> int:
        raise NotImplementedError


class TernaVLMAdapter(Adapter):
    """Wraps scripts/infer.py: same checkpoint loading, prompt format and greedy decoding as `infer.py`."""

    def __init__(self, ckpt: str, device: str):
        import torch
        from ternavlm.data import build_image_processor
        from scripts.infer import answer, build_prompt, load_model

        self.torch = torch
        self._answer, self._build_prompt = answer, build_prompt
        self.model = load_model(ckpt, device)
        self.proc = build_image_processor(self.model.cfg.vision_name)
        self.device = device
        self.name = "ternavlm:" + os.path.basename(os.path.dirname(ckpt) or ".") + "/" + os.path.basename(ckpt)

    def predict(self, image, prompt, max_new_tokens):
        text, _ = self._answer(self.model, self.proc, image, prompt, max_new_tokens=max_new_tokens)
        return text

    def score_options(self, image, prompt, options):
        """eval_pope.score_yes_no generalised to N candidates: one batched teacher-forced forward, the
        candidate with the highest summed log-likelihood (candidate tokens + EOS) wins."""
        torch = self.torch
        model, tok = self.model, self.model.tokenizer
        with torch.no_grad():
            prompt_ids = self._build_prompt(model, prompt)[0]
            seqs, labels = [], []
            for c in options:
                c_ids = tok.encode(c, add_special_tokens=False) + [tok.eos_token_id]
                seqs.append(torch.cat([prompt_ids, torch.tensor(c_ids)]))
                labels.append(torch.cat([torch.full_like(prompt_ids, -100), torch.tensor(c_ids)]))
            N, L = len(seqs), max(s.numel() for s in seqs)
            ids = torch.full((N, L), tok.eos_token_id)
            lab = torch.full((N, L), -100)
            att = torch.zeros((N, L), dtype=torch.long)
            for i, (s, l) in enumerate(zip(seqs, labels)):
                ids[i, : s.numel()] = s
                lab[i, : l.numel()] = l
                att[i, : s.numel()] = 1
            pix = self.proc(images=image, return_tensors="pt")["pixel_values"]
            pix = pix.to(self.device, next(model.vision.parameters()).dtype).expand(N, -1, -1, -1)
            out = model(ids.to(self.device), att.to(self.device), pix)
            logp = torch.log_softmax(out.logits[:, :-1].float(), -1)
            tgt = lab[:, 1:].to(self.device)
            mask = tgt != -100
            ll = (logp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1) * mask).sum(-1)
            return int(ll.argmax())


class HFAdapter(Adapter):
    """Baselines through transformers' AutoProcessor + AutoModelForImageTextToText and the model's own chat
    template (SmolVLM-256M/500M-Instruct = Idefics3, LLaVA-OneVision-0.5B = LlavaOnevision, ...)."""

    def __init__(self, hf_name: str, device: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(hf_name)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(hf_name, dtype=self.dtype)
        except TypeError:  # transformers < 5 spelling
            self.model = AutoModelForImageTextToText.from_pretrained(hf_name, torch_dtype=self.dtype)
        self.model.to(device).eval()
        self.tok = self.processor.tokenizer
        self.name = "hf:" + hf_name

    def _inputs(self, image, prompt: str):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = self.processor(text=[text], images=[image], return_tensors="pt")
        enc = {k: (v.to(self.device, self.dtype) if getattr(v, "is_floating_point", lambda: False)() else v.to(self.device))
               if hasattr(v, "to") else v for k, v in enc.items()}
        return text, enc

    def predict(self, image, prompt, max_new_tokens):
        torch = self.torch
        _, enc = self._inputs(image, prompt)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[0][enc["input_ids"].shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True).strip()

    def score_options(self, image, prompt, options):
        """Log-likelihood ranking. If every candidate is a single token, one forward on the prompt gives all
        scores from the last-position distribution; otherwise one full forward per candidate (N is small)."""
        torch = self.torch
        text, enc = self._inputs(image, prompt)
        lead = "" if text.endswith((" ", "\n")) else " "  # continue "Assistant:" with " Yes", not "Yes"
        cand_ids = [self.tok(lead + o, add_special_tokens=False)["input_ids"] for o in options]
        with torch.no_grad():
            out = self.model(**enc)
            last = torch.log_softmax(out.logits[0, -1].float(), -1)
            if all(len(c) == 1 for c in cand_ids):
                scores = [float(last[c[0]]) for c in cand_ids]
                return int(max(range(len(scores)), key=scores.__getitem__))
            scores = []
            for c in cand_ids:
                ids = torch.cat([enc["input_ids"], torch.tensor([c], device=self.device)], dim=1)
                kw = dict(enc)
                kw["input_ids"] = ids
                if "attention_mask" in kw:
                    kw["attention_mask"] = torch.ones_like(ids)
                logits = self.model(**kw).logits[0].float()
                logp = torch.log_softmax(logits[-len(c) - 1: -1], -1)
                scores.append(float(logp.gather(-1, torch.tensor(c, device=self.device).unsqueeze(-1)).sum()))
        return int(max(range(len(scores)), key=scores.__getitem__))


# ============================================================================================
# Streaming helpers
# ============================================================================================
def to_pil(x):
    from PIL import Image
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, dict):  # datasets Image(decode=False)
        if x.get("bytes"):
            return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
        if x.get("path"):
            return Image.open(x["path"]).convert("RGB")
    raise TypeError(f"cannot turn {type(x)} into an image")


def stream_rows(dataset, config, split, n, spread=False, size=None, image_cols=(), pair=False, decode=True):
    """Yield up to n rows of a streamed split. spread=True keeps every stride-th row (or row *pair*)
    so a small subset spans the split; images are left undecoded (raw bytes dict, see to_pil) while
    skipping, or always when decode=False."""
    from datasets import Image as HFImage, load_dataset
    ds = load_dataset(dataset, config, split=split, streaming=True)
    stride = 1
    if spread and n > 0 and size:
        units, want = (size // 2, max(1, n // 2)) if pair else (size, n)
        stride = max(1, units // want)
    if stride > 1 or not decode:
        for c in image_cols:
            ds = ds.cast_column(c, HFImage(decode=False))
    taken = 0
    for i, row in enumerate(ds):
        if n > 0 and taken >= n:
            break
        unit = i // 2 if pair else i
        if unit % stride:
            continue
        taken += 1
        yield row


class Progress:
    def __init__(self, tag, every=20):
        self.tag, self.every, self.t0, self.n, self.err = tag, every, time.time(), 0, 0

    def step(self, running: str):
        self.n += 1
        if self.n % self.every == 0:
            print(f"[{self.tag}] {self.n}: {running}  ({self.n / (time.time() - self.t0):.2f} it/s)", flush=True)

    def done(self) -> dict:
        dt = time.time() - self.t0
        return {"seconds": round(dt, 1), "samples_per_s": round(self.n / max(dt, 1e-6), 3), "errors": self.err}


# ============================================================================================
# Benchmark runners
# ============================================================================================
def _yes_no(adapter: Adapter, image, prompt: str, mode: str, max_new_tokens: int) -> str:
    if mode == "score":
        return ["yes", "no"][adapter.score_options(image, prompt, ["Yes", "No"])]
    return parse_yes_no(adapter.predict(image, prompt, max_new_tokens))


def run_pope(adapter, spec, a):
    c = spec["cols"]
    preds, golds, samples = [], [], []
    by_cat: dict[str, list] = {}
    pr = Progress("pope")
    for row in stream_rows(spec["dataset"], spec["config"], spec["split"], a.n, spec["spread"], spec["size"], (c["image"],)):
        try:
            q = row[c["question"]].strip()
            gold = row[c["answer"]].strip().lower()
            pred = _yes_no(adapter, to_pil(row[c["image"]]), PROMPT_YES_NO.format(q=q), a.mode, 4)
        except Exception as e:  # noqa: BLE001
            pr.err += 1
            print(f"[pope] skip: {type(e).__name__}: {e}", flush=True)
            continue
        preds.append(pred); golds.append(gold)
        by_cat.setdefault(str(row.get(c["category"], "all")), []).append((pred, gold))
        samples.append({"q": q, "gold": gold, "pred": pred})
        pr.step(f"acc {sum(p == g for p, g in zip(preds, golds)) / len(preds):.3f}")
    res = pope_metrics(preds, golds)
    res["by_category"] = {k: pope_metrics([p for p, _ in v], [g for _, g in v]) for k, v in by_cat.items()}
    res["mode"] = a.mode
    res.update(pr.done())
    return res, samples


def run_textvqa(adapter, spec, a):
    c = spec["cols"]
    scores, samples = [], []
    pr = Progress("textvqa")
    for row in stream_rows(spec["dataset"], spec["config"], spec["split"], a.n, spec["spread"], spec["size"], (c["image"],)):
        try:
            q = row[c["question"]].strip()
            answers = list(row[c["answers"]])
            pred = adapter.predict(to_pil(row[c["image"]]), PROMPT_SHORT.format(q=q), a.max_new_tokens)
        except Exception as e:  # noqa: BLE001
            pr.err += 1
            print(f"[textvqa] skip: {type(e).__name__}: {e}", flush=True)
            continue
        s = vqa_soft_accuracy(pred, answers)
        scores.append(s)
        samples.append({"q": q, "gold": answers, "pred": pred, "score": s})
        pr.step(f"soft-acc {sum(scores) / len(scores):.3f}")
    res = {"n": len(scores), "accuracy": sum(scores) / len(scores) if scores else 0.0, "mode": "generate (open-ended)"}
    res.update(pr.done())
    return res, samples


def run_gqa(adapter, spec, a):
    c = spec["cols"]
    # pass 1: the question subset (text only, cheap)
    qs = []
    for row in stream_rows(spec["dataset"], spec["config"], spec["split"], a.n, spec["spread"], spec["size"]):
        qs.append({"q": row[c["question"]].strip(), "gold": row[c["answer"]].strip(), "img": row[c["image_id"]]})
    need = {x["img"] for x in qs}
    # pass 2: stream the image config without decoding, decode only the referenced images, stop early
    images = {}
    print(f"[gqa] fetching {len(need)} images from config {spec['images_config']}", flush=True)
    for row in stream_rows(spec["dataset"], spec["images_config"], spec["split"], 0, False, None, (c["image"],), decode=False):
        if row[c["img_key"]] in need:
            images[row[c["img_key"]]] = row[c["image"]]  # undecoded bytes; decoded lazily below
            if len(images) == len(need):
                break
    correct, samples = 0, []
    pr = Progress("gqa")
    for x in qs:
        if x["img"] not in images:
            pr.err += 1
            continue
        try:
            pred = adapter.predict(to_pil(images[x["img"]]), PROMPT_SHORT.format(q=x["q"]), a.max_new_tokens)
        except Exception as e:  # noqa: BLE001
            pr.err += 1
            print(f"[gqa] skip: {type(e).__name__}: {e}", flush=True)
            continue
        ok = exact_match(pred, x["gold"])
        correct += int(ok)
        samples.append({"q": x["q"], "gold": x["gold"], "pred": pred, "correct": ok})
        pr.step(f"acc {correct / len(samples):.3f}")
    n = len(samples)
    res = {"n": n, "accuracy": correct / n if n else 0.0, "mode": "generate (open-ended)",
           "images_missing": len(need - set(images))}
    res.update(pr.done())
    return res, samples


def sqa_prompt(question: str, choices: list[str], hint: str = "") -> str:
    lines = []
    if hint and hint.strip():
        lines.append(f"Context: {hint.strip()}")
    lines.append(f"Question: {question.strip()}")
    lines.append("Options:")
    lines += [f"{LETTERS[i]}. {ch}" for i, ch in enumerate(choices)]
    lines.append(PROMPT_MC)
    return "\n".join(lines)


def run_sqa(adapter, spec, a):
    c = spec["cols"]
    correct, samples = 0, []
    pr = Progress("sqa")
    # over-fetch so the cap applies to rows that *have* an image (derek-thomas/ScienceQA mixes both)
    for row in stream_rows(spec["dataset"], spec["config"], spec["split"], 0, spec["spread"], spec["size"], (c["image"],)):
        if a.n > 0 and len(samples) >= a.n:
            break
        if row.get(c["image"]) is None:
            continue
        try:
            choices = list(row[c["choices"]])
            gold = int(row[c["answer"]])
            prompt = sqa_prompt(row[c["question"]], choices, row.get(c["hint"]) or "")
            image = to_pil(row[c["image"]])
            if a.mode == "score":
                pred = adapter.score_options(image, prompt, list(LETTERS[: len(choices)]))
            else:
                pred = parse_choice(adapter.predict(image, prompt, 8), choices)
        except Exception as e:  # noqa: BLE001
            pr.err += 1
            print(f"[sqa] skip: {type(e).__name__}: {e}", flush=True)
            continue
        ok = pred == gold
        correct += int(ok)
        samples.append({"q": row[c["question"]], "choices": choices, "gold": gold, "pred": pred, "correct": ok})
        pr.step(f"acc {correct / len(samples):.3f}")
    n = len(samples)
    res = {"n": n, "accuracy": correct / n if n else 0.0, "mode": a.mode}
    res.update(pr.done())
    return res, samples


def run_mme(adapter, spec, a):
    c = spec["cols"]
    records, samples = [], []
    pr = Progress("mme")
    for row in stream_rows(spec["dataset"], spec["config"], spec["split"], a.n, spec["spread"], spec["size"],
                           (c["image"],), pair=True):
        try:
            q = row[c["question"]].strip()  # MME questions already end with "Please answer yes or no."
            gold = row[c["answer"]].strip().lower()
            pred = _yes_no(adapter, to_pil(row[c["image"]]), q, a.mode, 4)
        except Exception as e:  # noqa: BLE001
            pr.err += 1
            print(f"[mme] skip: {type(e).__name__}: {e}", flush=True)
            continue
        rec = {"pair_id": row[c["pair_id"]], "category": row[c["category"]], "correct": pred == gold}
        records.append(rec)
        samples.append({**rec, "q": q, "gold": gold, "pred": pred})
        pr.step(f"acc {sum(r['correct'] for r in records) / len(records):.3f}")
    res = mme_scores(records)
    res["mode"] = a.mode
    res.update(pr.done())
    return res, samples


RUNNERS = {"pope": run_pope, "textvqa": run_textvqa, "gqa": run_gqa, "sqa": run_sqa, "mme": run_mme}


# ============================================================================================
# CLI
# ============================================================================================
def apply_overrides(specs: dict, overrides: list[str]) -> dict:
    """--override gqa.dataset=lmms-lab-encoder/GQA   --override pope.cols.answer=label   --override mme.spread=false"""
    specs = copy.deepcopy(specs)
    for ov in overrides:
        key, _, val = ov.partition("=")
        parts = key.split(".")
        if len(parts) < 2 or parts[0] not in specs:
            raise SystemExit(f"bad --override {ov!r}: expected <bench>.<key>[.<sub>]=<value>, bench in {list(specs)}")
        d = specs[parts[0]]
        for p in parts[1:-1]:
            d = d.setdefault(p, {})
        leaf = parts[-1]
        if val.lower() in ("true", "false"):
            d[leaf] = val.lower() == "true"
        elif val.lower() in ("none", "null", ""):
            d[leaf] = None
        elif val.isdigit():
            d[leaf] = int(val)
        else:
            d[leaf] = val
    return specs


def fmt_pct(x, digits=1):
    return f"{100 * x:.{digits}f}"


def markdown_row(name: str, results: dict) -> tuple[str, str]:
    header = "| model | POPE acc / F1 | TextVQA | GQA | SQA-IMG | MME-P / MME-C |"
    cells = [name]
    r = results.get("pope")
    cells.append(f"{fmt_pct(r['accuracy'])} / {fmt_pct(r['f1'])}" if r else "-")
    r = results.get("textvqa")
    cells.append(fmt_pct(r["accuracy"]) if r else "-")
    r = results.get("gqa")
    cells.append(fmt_pct(r["accuracy"]) if r else "-")
    r = results.get("sqa")
    cells.append(fmt_pct(r["accuracy"]) if r else "-")
    r = results.get("mme")
    cells.append(f"{r['perception_score']:.0f}/{r['perception_max']} / {r['cognition_score']:.0f}/{r['cognition_max']}" if r else "-")
    return header, "| " + " | ".join(cells) + " |"


def build_adapter(a) -> Adapter:
    if a.model == "ternavlm":
        if not a.ckpt:
            raise SystemExit("--ckpt is required with --model ternavlm")
        return TernaVLMAdapter(a.ckpt, a.device)
    if not a.hf_name:
        raise SystemExit("--hf-name is required with --model hf")
    return HFAdapter(a.hf_name, a.device)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["ternavlm", "hf"], required=True)
    ap.add_argument("--ckpt", help="train.py checkpoint (ternavlm)")
    ap.add_argument("--hf-name", help="HF model id, e.g. HuggingFaceTB/SmolVLM-256M-Instruct (hf)")
    ap.add_argument("--benchmarks", default="pope,textvqa,gqa,sqa,mme")
    ap.add_argument("--n", type=int, default=200, help="rows per benchmark (0 = whole split)")
    ap.add_argument("--mode", choices=["generate", "score"], default="generate")
    ap.add_argument("--max-new-tokens", type=int, default=16, help="for open-ended VQA (textvqa, gqa)")
    ap.add_argument("--spread", dest="spread", action="store_true", default=None,
                    help="take every k-th row so the subset spans the split (default: on for mme only)")
    ap.add_argument("--no-spread", dest="spread", action="store_false")
    ap.add_argument("--override", action="append", default=[], metavar="BENCH.KEY=VAL",
                    help="override a BENCHMARKS entry, e.g. pope.cols.answer=label, gqa.dataset=lmms-lab-encoder/GQA")
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    ap.add_argument("--out", default=None, help="default results/<model-name>.json")
    ap.add_argument("--save-preds", action="store_true", help="keep per-sample predictions in the JSON")
    a = ap.parse_args(argv)

    if a.device is None:
        import torch
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    specs = apply_overrides(BENCHMARKS, a.override)
    if a.spread is not None:
        for s in specs.values():
            s["spread"] = a.spread
    benches = [b.strip() for b in a.benchmarks.split(",") if b.strip()]
    unknown = [b for b in benches if b not in RUNNERS]
    if unknown:
        raise SystemExit(f"unknown benchmarks {unknown}; choose from {list(RUNNERS)}")

    adapter = build_adapter(a)
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", adapter.name.split(":", 1)[1])
    out_path = a.out or os.path.join("results", f"{tag}.json")

    results, preds = {}, {}
    for b in benches:
        spec = specs[b]
        print(f"\n=== {b}: {spec['dataset']} [{spec['config']}] split={spec['split']} n={a.n} mode={a.mode} "
              f"spread={spec['spread']}", flush=True)
        res, samples = RUNNERS[b](adapter, spec, a)
        res["dataset"] = {k: spec[k] for k in ("dataset", "config", "split") if k in spec}
        results[b] = res
        preds[b] = samples
        print(json.dumps({k: v for k, v in res.items() if k not in ("by_category", "categories")}, indent=2), flush=True)
        if b == "mme":
            print(json.dumps(res["categories"], indent=2), flush=True)

    header, row = markdown_row(adapter.name, results)
    payload = {"model": adapter.name, "mode": a.mode, "n": a.n, "device": a.device, "benchmarks": results,
               "markdown_header": header, "markdown_row": row}
    if a.save_preds:
        payload["predictions"] = preds
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    sep = "|" + "---|" * (header.count("|") - 1)
    print(f"\nwrote {out_path}\n\n{header}\n{sep}\n{row}")


if __name__ == "__main__":
    main()
