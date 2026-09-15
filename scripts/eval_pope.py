"""
POPE object-hallucination eval (yes/no questions) for a train.py checkpoint.

    python scripts/eval_pope.py --ckpt ckpt/stage2/latest.pt --n 300
    python scripts/eval_pope.py --ckpt ckpt/stage2/latest.pt --n 300 --scoring   # no-decode mode

--scoring skips generation entirely: it scores the two completions "Yes" and "No" by teacher-forced
log-likelihood in one batched forward, which is the cheap closed-form path the README argues for.

Dataset: lmms-lab/POPE, split "test", columns image / question / answer / category (column names are args).
"""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ternavlm.data import build_image_processor  # noqa: E402
from scripts.infer import answer, build_prompt, load_model  # noqa: E402


@torch.no_grad()
def score_yes_no(model, proc, image, question) -> str:
    device = next(model.parameters()).device
    tok = model.tokenizer
    prompt = build_prompt(model, question)[0]
    cands = ["Yes", "No"]
    seqs, labels = [], []
    for c in cands:
        c_ids = tok.encode(c, add_special_tokens=False) + [tok.eos_token_id]
        seqs.append(torch.cat([prompt, torch.tensor(c_ids)]))
        labels.append(torch.cat([torch.full_like(prompt, -100), torch.tensor(c_ids)]))
    L = max(s.numel() for s in seqs)
    ids = torch.full((2, L), tok.eos_token_id); lab = torch.full((2, L), -100); att = torch.zeros((2, L), dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labels)):
        ids[i, : s.numel()] = s; lab[i, : l.numel()] = l; att[i, : s.numel()] = 1
    pix = proc(images=image, return_tensors="pt")["pixel_values"].to(device, next(model.vision.parameters()).dtype)
    pix = pix.expand(2, -1, -1, -1)
    out = model(ids.to(device), att.to(device), pix)
    logp = torch.log_softmax(out.logits[:, :-1].float(), -1)
    tgt = lab[:, 1:].to(device)
    mask = tgt != -100
    ll = (logp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1) * mask).sum(-1)
    return cands[int(ll.argmax())].lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--scoring", action="store_true", help="score Yes/No instead of generating")
    ap.add_argument("--out", default="results/pope.json")
    ap.add_argument("--dataset", default="lmms-lab/POPE")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cols", default="image,question,answer,category")
    a = ap.parse_args()
    c_img, c_q, c_a, c_cat = a.cols.split(",")

    model = load_model(a.ckpt, a.device)
    proc = build_image_processor(model.cfg.vision_name)
    ds = load_dataset(a.dataset, split=a.split, streaming=True)

    stats = {"n": 0, "correct": 0, "yes": 0, "by_cat": {}}
    for i, row in enumerate(ds):
        if i >= a.n:
            break
        q = row[c_q].strip()
        gold = row[c_a].strip().lower()
        if a.scoring:
            pred = score_yes_no(model, proc, row[c_img].convert("RGB"), q)
        else:
            text, _ = answer(model, proc, row[c_img].convert("RGB"), q + " Answer yes or no.", max_new_tokens=4)
            pred = "yes" if "yes" in text.lower() else "no"
        cat = str(row.get(c_cat, "all"))
        s = stats["by_cat"].setdefault(cat, {"n": 0, "correct": 0})
        s["n"] += 1; stats["n"] += 1
        s["correct"] += int(pred == gold); stats["correct"] += int(pred == gold)
        stats["yes"] += int(pred == "yes")
        if stats["n"] % 20 == 0:
            print(f"{stats['n']}: acc {stats['correct']/stats['n']:.3f} yes-ratio {stats['yes']/stats['n']:.2f}", flush=True)

    stats["accuracy"] = stats["correct"] / max(1, stats["n"])
    stats["yes_ratio"] = stats["yes"] / max(1, stats["n"])
    for s in stats["by_cat"].values():
        s["accuracy"] = s["correct"] / max(1, s["n"])
    stats["mode"] = "scoring" if a.scoring else "generate"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(stats, open(a.out, "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
