"""
Shard-local LLaVA-format data for Kaggle: one parquet shard on disk at a time, resumable by (epoch, shard, row).

Why not `datasets` streaming
----------------------------
`load_dataset(..., streaming=True)` reads remote parquet through fsspec/pyarrow with up to 10 shards open in
parallel. On Kaggle's fast network that reader buffered far more than it yielded and the OOM killer took the
process out ~50 s into every run (kernels v1-v3). Here each rank downloads *one* shard (~500 MB) to local disk
with `hf_hub_download`, reads it with pyarrow, and prefetches the next shard in a background thread. Memory
stays flat (a shuffle buffer of raw JPEG bytes), disk stays bounded (`keep_shards`), and resume is instant:
the checkpoint stores the reader cursor and we seek straight to that row group.

Expected dataset rows (parquet columns):
    image:         HF Image struct {bytes, path} (or raw bytes)
    conversations: [{"from": "human"|"gpt", "value": "..."}] (LLaVA style)
or  caption/text:  str   (pretrain-style, turned into a single caption turn)

Known good sources:
    stage 1: lmms-lab/LLaVA-ReCap-558K                      (26 shards under data/)
    stage 2: lmms-lab/LLaVA-OneVision-Data, subset e.g. "sharegpt4v(coco)"   (shards under <subset>/)

DDP: every rank reads the same shard sequence and takes rows with global_row % world_size == rank, so ranks stay
in lockstep and share one download (hf_hub_download serialises concurrent downloads of a file with a lock).
"""

from __future__ import annotations

import io
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import AutoImageProcessor

IGNORE = -100
STATE_KEY = "_state"


@dataclass
class DataConfig:
    name: str
    subset: str | None = None
    split: str = "train"
    max_samples: int | None = None       # global rows to advance (over all ranks and epochs); None = unbounded
    max_len: int = 512
    shuffle_buffer: int = 2000
    seed: int = 0
    caption_prompts: tuple[str, ...] = (
        "Describe the image.",
        "What is in this picture?",
        "Write a short caption for this image.",
    )
    shard_dir: str | None = None         # where shards are downloaded; default $TERNAVLM_SHARD_DIR or ~/.cache/ternavlm
    keep_shards: int = 2                 # shards kept on disk behind the cursor (rank 0 deletes older ones)
    local_files: tuple[str, ...] | None = None   # offline/test mode: read these parquet files, no download/delete
    read_batch: int = 64                 # rows per pyarrow batch


def default_shard_dir() -> str:
    return os.environ.get("TERNAVLM_SHARD_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "ternavlm", "shards")


def list_shards(name: str, subset: str | None = None, split: str = "train") -> list[str]:
    """Parquet files of one split of a HF dataset repo, in repo order (sorted)."""
    from huggingface_hub import HfApi

    files = [f for f in HfApi().list_repo_files(name, repo_type="dataset") if f.endswith(".parquet")]
    prefix = f"{subset}/" if subset else "data/"
    cands = [f for f in files if f.startswith(prefix)] or ([] if subset else files)
    by_split = [f for f in cands if os.path.basename(f).startswith(f"{split}-") or f"/{split}/" in f]
    out = sorted(by_split or cands)
    if not out:
        raise RuntimeError(f"no parquet shards for {name} subset={subset!r} split={split!r}")
    return out


class ShardCache:
    """Downloads parquet shards of a HF dataset repo to a local dir one at a time, prefetching the next in a thread."""

    def __init__(self, repo: str, files: list[str], local_dir: str, keep: int = 2, local: bool = False):
        self.repo, self.files, self.dir, self.keep, self.local = repo, list(files), local_dir, keep, local
        self._threads: dict[int, threading.Thread] = {}
        if not local:
            os.makedirs(local_dir, exist_ok=True)

    def __len__(self):
        return len(self.files)

    def path(self, i: int) -> str:
        return self.files[i] if self.local else os.path.join(self.dir, self.files[i])

    def _fetch(self, i: int) -> str:
        if self.local:
            return self.files[i]
        from huggingface_hub import hf_hub_download

        return hf_hub_download(self.repo, self.files[i], repo_type="dataset", local_dir=self.dir)

    def get(self, i: int) -> str:
        """Blocking: returns the local path of shard i (downloads if missing, waits for a running prefetch)."""
        t = self._threads.pop(i, None)
        if t is not None:
            t.join()
        err = None
        for attempt in range(5):
            try:
                return self._fetch(i)
            except Exception as e:  # network hiccup: retry, the download resumes/reuses what is on disk
                err = e
                print(f"[data] download of {self.files[i]} failed (attempt {attempt + 1}/5): {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"could not download {self.files[i]}") from err

    def prefetch(self, i: int) -> None:
        if self.local or i in self._threads or not (0 <= i < len(self.files)):
            return
        if os.path.exists(self.path(i)):
            return
        t = threading.Thread(target=self._quiet_fetch, args=(i,), daemon=True, name=f"shard-prefetch-{i}")
        t.start()
        self._threads[i] = t

    def _quiet_fetch(self, i: int) -> None:
        try:
            self._fetch(i)
        except Exception as e:  # get() will retry synchronously
            print(f"[data] prefetch of {self.files[i]} failed: {e}", flush=True)

    def release(self, i: int) -> None:
        """Delete shard i from disk (only for shards well behind the cursor; only rank 0 should call this)."""
        if self.local or not (0 <= i < len(self.files)):
            return
        try:
            os.remove(self.path(i))
        except FileNotFoundError:
            pass


def _to_turns(row: dict, rng: random.Random, prompts) -> list[tuple[str, str]]:
    """Return [(user, assistant), ...]. Strips the literal '<image>\\n' LLaVA puts in the first turn."""
    if row.get("conversations"):
        conv = row["conversations"]
        turns = []
        for i in range(0, len(conv) - 1, 2):
            u = conv[i]["value"].replace("<image>", "").strip()
            a = conv[i + 1]["value"].strip()
            turns.append((u, a))
        return turns
    text = row.get("caption") or row.get("text") or ""
    return [(rng.choice(prompts), text.strip())]


def _open_image(v) -> Image.Image:
    if isinstance(v, Image.Image):
        return v
    if isinstance(v, dict):
        if v.get("bytes") is not None:
            return Image.open(io.BytesIO(v["bytes"]))
        return Image.open(v["path"])
    return Image.open(io.BytesIO(v))


def iter_shard_rows(path: str, columns: list[str] | None, row_start: int, world: int, rank: int, global_start: int,
                    batch_rows: int = 64) -> Iterator[tuple[int, dict]]:
    """Yield (global_row, row_dict) for this rank's stripe of a parquet shard, starting at in-shard row `row_start`
    (which corresponds to global row `global_start`). Whole row groups before row_start are skipped without reading."""
    pf = pq.ParquetFile(path)
    md = pf.metadata
    cols = [c for c in (columns or pf.schema_arrow.names) if c in pf.schema_arrow.names]
    base = 0  # first in-shard row of the current row group
    for rg in range(md.num_row_groups):
        n = md.row_group(rg).num_rows
        if base + n <= row_start:
            base += n
            continue
        table = pf.read_row_group(rg, columns=cols)
        off = 0
        while off < n:
            rb = table.slice(off, batch_rows)
            m = rb.num_rows
            first = base + off  # in-shard index of rb[0]
            idx = [j for j in range(m) if first + j >= row_start and (global_start + (first + j - row_start)) % world == rank]
            if idx:
                for j, r in zip(idx, rb.take(pa.array(idx, type=pa.int32())).to_pylist()):
                    yield global_start + (first + j - row_start), r
            off += m
        base += n


class LlavaStream(IterableDataset):
    """
    Resumable, shard-local stream of {"input_ids", "labels", "pixel_values", "_state"} examples.

    State (also attached to every example under STATE_KEY, so the trainer can checkpoint it exactly):
        {"epoch": e, "pos": p, "row": r, "global": g}
    = the reader cursor: next global row g to read is in-shard row r of the p-th shard of epoch e's shard order.
    Resuming re-reads from the cursor; up to `shuffle_buffer` rows that were buffered but not yet yielded are lost.
    """

    COLUMNS = ("image", "conversations", "caption", "text")

    def __init__(self, cfg: DataConfig, tokenizer, image_processor, image_token: str, num_image_tokens: int,
                 rank: int = 0, world_size: int = 1, state: dict | None = None):
        self.cfg = cfg
        self.tok = tokenizer
        self.proc = image_processor
        self.image_token = image_token
        self.n_img = num_image_tokens
        self.rank = rank
        self.world = world_size
        self.state = dict(state) if state else {"epoch": 0, "pos": 0, "row": 0, "global": 0}
        self.rng = random.Random(cfg.seed * 7919 + rank)
        self.files = list(cfg.local_files) if cfg.local_files else list_shards(cfg.name, cfg.subset, cfg.split)
        self.cache = ShardCache(cfg.name, self.files, cfg.shard_dir or default_shard_dir(), keep=cfg.keep_shards,
                                local=bool(cfg.local_files))
        self.dropped = 0

    # ---- state -------------------------------------------------------------------
    def state_dict(self) -> dict:
        return dict(self.state)

    def load_state_dict(self, state: dict) -> None:
        self.state = dict(state)

    def shard_order(self, epoch: int) -> list[int]:
        order = list(range(len(self.files)))
        random.Random(self.cfg.seed + 1000003 * epoch).shuffle(order)
        return order

    # ---- encoding ----------------------------------------------------------------
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
            # Minimal, template-agnostic format (scripts/infer.py builds the same prompt for inference).
            p_ids = self.tok.encode(f"User: {user_text}\nAssistant:", add_special_tokens=False)
            a_ids = self.tok.encode(" " + a, add_special_tokens=False) + [self.tok.eos_token_id]
            ids += p_ids + a_ids
            labels += [IGNORE] * len(p_ids) + a_ids
        if len(ids) > self.cfg.max_len:
            ids, labels = ids[: self.cfg.max_len], labels[: self.cfg.max_len]
            if ids.count(self.tok.convert_tokens_to_ids(self.image_token)) != self.n_img:
                return None  # image block got truncated; drop
        img = _open_image(row["image"]).convert("RGB")
        pix = self.proc(images=img, return_tensors="pt")["pixel_values"][0]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "pixel_values": pix}

    # ---- reading -----------------------------------------------------------------
    def _rows(self) -> Iterator[dict]:
        """This rank's stripe of rows, across shards and epochs, from the current cursor; updates self.state."""
        st = self.state
        cols = list(self.COLUMNS)
        while True:
            order = self.shard_order(st["epoch"])
            while st["pos"] < len(order):
                if self.cfg.max_samples is not None and st["global"] >= self.cfg.max_samples:
                    return
                shard = order[st["pos"]]
                path = self.cache.get(shard)
                if st["pos"] + 1 < len(order):
                    self.cache.prefetch(order[st["pos"] + 1])
                else:
                    self.cache.prefetch(self.shard_order(st["epoch"] + 1)[0])
                n_rows = pq.ParquetFile(path).metadata.num_rows
                row_start, g_start = st["row"], st["global"]
                for g, row in iter_shard_rows(path, cols, row_start, self.world, self.rank, g_start, self.cfg.read_batch):
                    if self.cfg.max_samples is not None and g >= self.cfg.max_samples:
                        break
                    # cursor = next row to read (this rank's stripe is fixed by global index, so it is rank-agnostic)
                    st["row"], st["global"] = row_start + (g - g_start) + 1, g + 1
                    yield row
                # whole shard consumed (or max_samples hit inside it)
                consumed = min(n_rows - row_start, max(0, (self.cfg.max_samples or 10**18) - g_start))
                st["row"], st["global"] = 0, g_start + consumed
                st["pos"] += 1
                old = st["pos"] - 1 - self.cfg.keep_shards
                if self.rank == 0 and old >= 0:
                    self.cache.release(order[old])
            st["epoch"] += 1
            st["pos"], st["row"] = 0, 0

    def __iter__(self) -> Iterator[dict]:
        # DataLoader workers > 0 would each own a copy of the cursor; run in-process (train.py enforces workers=0).
        wi = torch.utils.data.get_worker_info()
        assert wi is None or wi.num_workers <= 1, "LlavaStream must run in-process (workers=0)"
        buf: list[dict] = []
        B = max(1, self.cfg.shuffle_buffer)
        for row in self._rows():
            buf.append(row)
            if len(buf) < B:
                continue
            j = self.rng.randrange(len(buf))
            buf[j], buf[-1] = buf[-1], buf[j]
            ex = self._safe_encode(buf.pop())
            if ex is not None:
                yield ex
        self.rng.shuffle(buf)
        for row in buf:
            ex = self._safe_encode(row)
            if ex is not None:
                yield ex

    def _safe_encode(self, row: dict):
        try:
            ex = self._encode(row)
        except Exception as e:  # corrupt image / odd row: drop it
            self.dropped += 1
            if self.dropped <= 5:
                print(f"[data] dropping row: {type(e).__name__}: {e}", flush=True)
            return None
        if ex is None:
            self.dropped += 1
            return None
        ex[STATE_KEY] = dict(self.state)
        return ex


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
    out = {"input_ids": ids, "labels": lab, "attention_mask": att,
           "pixel_values": torch.stack([b["pixel_values"] for b in batch])}
    if STATE_KEY in batch[-1]:
        out["data_state"] = batch[-1][STATE_KEY]
    return out


def build_image_processor(vision_name: str):
    return AutoImageProcessor.from_pretrained(vision_name)
