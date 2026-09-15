"""
Streaming LLaVA-format data for Kaggle: no giant image zips, resumable by sample offset.

Expected dataset rows (HF `datasets`, streaming):
    image:         PIL image
    conversations: [{"from": "human"|"gpt", "value": "..."}] (LLaVA style)
or  caption/text:  str   (pretrain-style, turned into a single caption turn)

Known good sources:
    stage 1: lmms-lab/LLaVA-ReCap-558K   (images embedded, captions)
    stage 2: lmms-lab/LLaVA-OneVision-Data (pick a few subsets) or HuggingFaceM4/the_cauldron
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import AutoImageProcessor

IGNORE = -100


@dataclass
class DataConfig:
    name: str
    subset: str | None = None
    split: str = "train"
    max_samples: int | None = None
    max_len: int = 512
    shuffle_buffer: int = 2000
    seed: int = 0
    caption_prompts: tuple[str, ...] = (
        "Describe the image.",
        "What is in this picture?",
        "Write a short caption for this image.",
    )


def _to_turns(row: dict, rng: random.Random, prompts) -> list[tuple[str, str]]:
    """Return [(user, assistant), ...]. Strips the literal '<image>\\n' LLaVA puts in the first turn."""
    if "conversations" in row and row["conversations"]:
        conv = row["conversations"]
        turns = []
        for i in range(0, len(conv) - 1, 2):
            u = conv[i]["value"].replace("<image>", "").strip()
            a = conv[i + 1]["value"].strip()
            turns.append((u, a))
        return turns
    text = row.get("caption") or row.get("text") or ""
    return [(rng.choice(prompts), text.strip())]


class LlavaStream(IterableDataset):
    def __init__(self, cfg: DataConfig, tokenizer, image_processor, image_token: str, num_image_tokens: int,
                 skip: int = 0, rank: int = 0, world_size: int = 1):
        """`skip` = number of samples this *shard* already consumed (trainer divides the global count)."""
        self.cfg = cfg
        self.tok = tokenizer
        self.proc = image_processor
        self.image_token = image_token
        self.n_img = num_image_tokens
        self.skip = skip
        self.rank = rank
        self.world = world_size
        self.rng = random.Random(cfg.seed)

    def _encode(self, row: dict):
        turns = _to_turns(row, self.rng, self.cfg.caption_prompts)
        if not turns:
            return None
        img_block = self.image_token * self.n_img
        ids: list[int] = []
        labels: list[int] = []
        bos = [self.tok.bos_token_id] if self.tok.bos_token_id is not None else []
        ids += bos
        labels += [IGNORE] * len(bos)
        for i, (u, a) in enumerate(turns):
            user_text = (img_block + "\n" + u) if i == 0 else u
            # Minimal, template-agnostic format. Swap for tokenizer.apply_chat_template if the LM has one.
            p_ids = self.tok.encode(f"User: {user_text}\nAssistant:", add_special_tokens=False)
            a_ids = self.tok.encode(" " + a, add_special_tokens=False) + [self.tok.eos_token_id]
            ids += p_ids + a_ids
            labels += [IGNORE] * len(p_ids) + a_ids
        if len(ids) > self.cfg.max_len:
            ids, labels = ids[: self.cfg.max_len], labels[: self.cfg.max_len]
            if (torch.tensor(ids) == self.tok.convert_tokens_to_ids(self.image_token)).sum() != self.n_img:
                return None  # image block got truncated; drop
        pix = self.proc(images=row["image"].convert("RGB"), return_tensors="pt")["pixel_values"][0]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "pixel_values": pix}

    def __iter__(self) -> Iterator[dict]:
        ds = load_dataset(self.cfg.name, self.cfg.subset, split=self.cfg.split, streaming=True)
        ds = ds.shuffle(seed=self.cfg.seed, buffer_size=self.cfg.shuffle_buffer)
        # shard across DDP ranks *and* DataLoader workers, otherwise every worker replays the whole stream
        wi = torch.utils.data.get_worker_info()
        n_workers = wi.num_workers if wi else 1
        worker = wi.id if wi else 0
        shard, n_shards = self.rank * n_workers + worker, self.world * n_workers
        seen = 0
        for i, row in enumerate(ds):
            if self.cfg.max_samples is not None and i >= self.cfg.max_samples:
                break
            if i % n_shards != shard:
                continue
            if seen < self.skip:
                seen += 1
                continue
            seen += 1
            try:
                ex = self._encode(row)
            except Exception:
                continue
            if ex is not None:
                yield ex


def collate(batch: list[dict], pad_id: int) -> dict:
    L = max(b["input_ids"].shape[0] for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), IGNORE, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        ids[i, :n] = b["input_ids"]
        lab[i, :n] = b["labels"]
        att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att,
            "pixel_values": torch.stack([b["pixel_values"] for b in batch])}


def build_image_processor(vision_name: str):
    return AutoImageProcessor.from_pretrained(vision_name)
