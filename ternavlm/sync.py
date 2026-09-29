"""
Off-machine checkpoint sync: after every save, copy the checkpoint aside and upload it to a private Hugging Face
model repo in a background thread, so a cancelled Kaggle kernel, an exhausted quota, or a reclaimed cloud GPU costs at
most one save interval instead of the whole session (Kaggle discards the output of a kernel that does not end
normally).

Enabled only when both are set (anywhere: Kaggle secret, local shell, cloud VM):
    TERNAVLM_HF_REPO   e.g. "<hf-user>/ternavlm-ckpt"   (created private on first upload)
    HF_TOKEN           a Hugging Face token with write access
Files land in the repo as <stage>/latest.pt and <stage>/diag.jsonl. Restore with:
    huggingface-cli download <repo> <stage>/latest.pt --local-dir ckpt/..      (or scripts via hf_hub_download)
"""

from __future__ import annotations

import os
import shutil
import threading
import time


class CheckpointSync:
    def __init__(self, repo: str | None = None, token: str | None = None, stage: str = "run"):
        self.repo = repo if repo is not None else os.environ.get("TERNAVLM_HF_REPO", "")
        self.token = token if token is not None else os.environ.get("HF_TOKEN", "")
        self.stage = stage
        self.enabled = bool(self.repo and self.token)
        self._thread: threading.Thread | None = None
        self._created = False
        self.last_ok_step: int | None = None

    def _api(self):
        from huggingface_hub import HfApi

        return HfApi(token=self.token)

    def _upload(self, files: dict[str, str], step: int, staging: str) -> None:
        t0 = time.time()
        try:
            api = self._api()
            if not self._created:
                api.create_repo(self.repo, repo_type="model", private=True, exist_ok=True)
                self._created = True
            for dst, src in files.items():
                api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=self.repo, repo_type="model",
                                commit_message=f"{self.stage} step {step}")
            self.last_ok_step = step
            print(f"[sync] step {step} uploaded to {self.repo} in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # never kill training over a failed upload; the next save retries
            print(f"[sync] upload of step {step} failed: {type(e).__name__}: {e}", flush=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def push(self, ckpt_path: str, step: int, extra: list[str] | None = None) -> bool:
        """Snapshot ckpt_path (+ extra files) and upload in the background. Skips if the previous upload is still
        running (the next save will carry a newer checkpoint anyway). Returns True if an upload was started."""
        if not self.enabled:
            return False
        if self._thread is not None and self._thread.is_alive():
            print(f"[sync] previous upload still running; skipping step {step}", flush=True)
            return False
        staging = os.path.join(os.path.dirname(ckpt_path) or ".", f".sync_{step}")
        os.makedirs(staging, exist_ok=True)
        files = {}
        for p in [ckpt_path] + [e for e in (extra or []) if os.path.exists(e)]:
            snap = os.path.join(staging, os.path.basename(p))
            shutil.copy(p, snap)  # the trainer overwrites latest.pt at the next save; upload a frozen copy
            files[f"{self.stage}/{os.path.basename(p)}"] = snap
        self._thread = threading.Thread(target=self._upload, args=(files, step, staging), daemon=True,
                                        name=f"ckpt-sync-{step}")
        self._thread.start()
        return True

    def wait(self, timeout: float = 1800) -> None:
        """Block until the in-flight upload finishes (called once at the end of training)."""
        if self._thread is not None:
            self._thread.join(timeout)
