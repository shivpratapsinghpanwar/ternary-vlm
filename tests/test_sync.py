"""CheckpointSync without network: disabled by default, snapshots files, never lets an upload failure escape."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ternavlm.sync import CheckpointSync  # noqa: E402


def test_disabled_without_repo_or_token(monkeypatch, tmp_path):
    monkeypatch.delenv("TERNAVLM_HF_REPO", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    s = CheckpointSync(stage="stage2")
    p = tmp_path / "latest.pt"
    p.write_bytes(b"x")
    assert not s.enabled and s.push(str(p), 1) is False


def test_uploads_frozen_snapshot_and_survives_failure(tmp_path):
    ck = tmp_path / "latest.pt"
    ck.write_bytes(b"step1")
    diag = tmp_path / "diag.jsonl"
    diag.write_bytes(b"{}\n")
    seen = {}

    class FakeApi:
        def create_repo(self, *a, **k):
            pass

        def upload_file(self, path_or_fileobj, path_in_repo, **k):
            seen[path_in_repo] = open(path_or_fileobj, "rb").read()

    s = CheckpointSync(repo="u/r", token="t", stage="stage2")
    s._api = lambda: FakeApi()
    assert s.push(str(ck), 50, extra=[str(diag), str(tmp_path / "missing")])
    ck.write_bytes(b"step2")  # trainer overwrites while the upload is in flight: the snapshot must be unaffected
    s.wait(10)
    assert seen == {"stage2/latest.pt": b"step1", "stage2/diag.jsonl": b"{}\n"} and s.last_ok_step == 50
    assert not any(n.startswith(".sync_") for n in os.listdir(tmp_path))  # staging cleaned up

    class Boom(FakeApi):
        def upload_file(self, *a, **k):
            raise ConnectionError("network down")

    s._api = lambda: Boom()
    assert s.push(str(ck), 100)
    s.wait(10)  # failure is logged, not raised
    assert s.last_ok_step == 50
