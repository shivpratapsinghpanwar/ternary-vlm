"""Offline tests for the shard-local data stream (synthetic parquet shards, stub tokenizer/processor)."""

import io
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ternavlm.data import DataConfig, LlavaStream, STATE_KEY, collate, iter_shard_rows  # noqa: E402


class StubTok:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None

    def encode(self, text, add_special_tokens=False):
        # <image> -> id 3, everything else one id per character (deterministic, no downloads)
        out = []
        i = 0
        while i < len(text):
            if text.startswith("<image>", i):
                out.append(3)
                i += len("<image>")
            else:
                out.append(10 + ord(text[i]) % 200)
                i += 1
        return out

    def convert_tokens_to_ids(self, t):
        return 3 if t == "<image>" else 0


class StubProc:
    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": torch.zeros(1, 3, 8, 8)}


def _jpeg(i):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (i % 256, 0, 0)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    d = tmp_path_factory.mktemp("shards")
    files = []
    n = 0
    for s, rows in enumerate([25, 17, 30]):  # uneven shard sizes, several row groups each
        recs = []
        for _ in range(rows):
            recs.append({"id": n, "image": {"bytes": _jpeg(n), "path": None},
                         "conversations": [{"from": "human", "value": f"<image>\nq{n}"}, {"from": "gpt", "value": f"a{n}"}]})
            n += 1
        table = pa.Table.from_pylist(recs)
        p = str(d / f"train-{s:05d}-of-00003.parquet")
        pq.write_table(table, p, row_group_size=7)
        files.append(p)
    return files, n


def _stream(files, rank=0, world=1, state=None, **kw):
    cfg = DataConfig(name="local", local_files=tuple(files), max_len=512, shuffle_buffer=kw.pop("buffer", 1), **kw)
    return LlavaStream(cfg, StubTok(), StubProc(), "<image>", 2, rank=rank, world_size=world, state=state)


def _ids_of(ex):
    """Recover the synthetic row id from the encoded question 'q{n}' (chars after the <image> block + newline)."""
    ids = ex["input_ids"].tolist()
    # layout: BOS, 'User: ' (6), <image> x2, '\n', 'q', digits..., '\nAssistant:' ...
    start = 1 + 6 + 2 + 1 + 1
    digits = []
    for t in ids[start:]:
        ch = None
        for c in "0123456789":
            if 10 + ord(c) % 200 == t:
                ch = c
        if ch is None:
            break
        digits.append(ch)
    return int("".join(digits))


def test_iter_shard_rows_skips_row_groups(shards):
    files, _ = shards
    rows = list(iter_shard_rows(files[0], ["id"], row_start=10, world=1, rank=0, global_start=100, batch_rows=4))
    assert [g for g, _ in rows] == list(range(100, 115))
    assert [r["id"] for _, r in rows] == list(range(10, 25))


def test_ranks_partition_rows(shards):
    files, n = shards
    seen = {}
    for rank in range(2):
        for ex in _stream(files, rank=rank, world=2, max_samples=n):
            seen.setdefault(rank, []).append(_ids_of(ex))
    assert sorted(seen[0] + seen[1]) == list(range(n))
    assert not set(seen[0]) & set(seen[1])
    assert abs(len(seen[0]) - len(seen[1])) <= 1


def test_resume_from_state_is_exact_without_buffer(shards):
    files, n = shards
    ds = _stream(files, max_samples=n)
    first, state = [], None
    for k, ex in enumerate(ds):
        first.append(_ids_of(ex))
        if k == 30:
            state = ex[STATE_KEY]
            break
    rest = [_ids_of(ex) for ex in _stream(files, max_samples=n, state=state)]
    assert sorted(first + rest) == list(range(n))  # no repeats, no gaps


def test_epochs_and_max_samples(shards):
    files, n = shards
    ids = [_ids_of(ex) for ex in _stream(files, max_samples=2 * n + 5)]
    assert len(ids) == 2 * n + 5
    assert sorted(ids[:n]) == list(range(n)) and sorted(ids[n:2 * n]) == list(range(n))
    # state after the run points into epoch 2
    ds = _stream(files, max_samples=2 * n + 5)
    last = None
    for ex in ds:
        last = ex[STATE_KEY]
    assert last["epoch"] == 2 and last["global"] == 2 * n + 5


def test_shuffle_buffer_yields_everything(shards):
    files, n = shards
    ids = [_ids_of(ex) for ex in _stream(files, max_samples=n, buffer=16)]
    assert sorted(ids) == list(range(n))
    assert ids != sorted(ids)  # actually shuffled


def test_collate_carries_state(shards):
    files, n = shards
    exs = []
    for ex in _stream(files, max_samples=n):
        exs.append(ex)
        if len(exs) == 3:
            break
    b = collate(exs, pad_id=0)
    assert b["input_ids"].shape[0] == 3 and b["data_state"] == exs[-1][STATE_KEY]
    assert (b["labels"][b["attention_mask"] == 0] == -100).all()
