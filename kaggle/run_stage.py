"""
Kaggle session driver. Paste into a Kaggle notebook cell (or run as a Kaggle Script) with
accelerator = "GPU T4 x2" and internet enabled.

Each session:
  1. installs deps, clones the repo
  2. pulls the latest checkpoint for this stage from a Hugging Face model repo (if any)
  3. trains until the time budget, saving ckpt/<stage>/latest.pt every N steps
  4. pushes latest.pt back to the Hub so the next session resumes

Secrets needed in Kaggle "Add-ons > Secrets":  HF_TOKEN  (write access)
Set REPO / HF_REPO / STAGE below.
"""

import os
import subprocess
import sys

REPO = "https://github.com/<you>/ternary-vlm.git"   # <- your repo
HF_REPO = "<you>/ternavlm-checkpoints"                # <- private HF model repo for checkpoints
STAGE = os.environ.get("STAGE", "stage1")             # smoke | stage1 | stage2
TIME_BUDGET_MIN = 680                                  # 11h20m; Kaggle kills at 12h
INIT_FROM = {"stage2": "stage1"}.get(STAGE)            # stage2 starts from stage1 weights


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

    sh(f"pip install -q -r <(curl -s {REPO.replace('.git','')}/raw/main/requirements.txt) 2>/dev/null || true")
    if not os.path.exists("ternary-vlm"):
        sh(f"git clone -q {REPO}")
    os.chdir("ternary-vlm")
    sh("pip install -q -r requirements.txt")

    from huggingface_hub import HfApi, hf_hub_download, create_repo
    api = HfApi()
    create_repo(HF_REPO, private=True, exist_ok=True)

    # --- fetch checkpoints ---------------------------------------------------------------
    resume = f"ckpt/{STAGE}/latest.pt"
    init = None
    os.makedirs(f"ckpt/{STAGE}", exist_ok=True)
    try:
        hf_hub_download(HF_REPO, f"{STAGE}/latest.pt", local_dir="ckpt_hub")
        os.replace(f"ckpt_hub/{STAGE}/latest.pt", resume)
        print(f"[hub] resuming {STAGE}")
    except Exception as e:
        print(f"[hub] no checkpoint for {STAGE} ({type(e).__name__}); starting fresh")
    if INIT_FROM and not os.path.exists(resume):
        hf_hub_download(HF_REPO, f"{INIT_FROM}/latest.pt", local_dir="ckpt_hub")
        init = f"ckpt_hub/{INIT_FROM}/latest.pt"
        print(f"[hub] initialising {STAGE} from {INIT_FROM}")

    # --- train ----------------------------------------------------------------------------
    n_gpu = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
    launcher = f"torchrun --nproc_per_node={n_gpu}" if n_gpu > 1 else sys.executable
    cmd = f"{launcher} train.py --config configs/{STAGE}.yaml --resume {resume} --time-budget-min {TIME_BUDGET_MIN}"
    if init:
        cmd += f" --init {init}"
    try:
        sh(cmd)
    finally:
        # --- push whatever we have, even on failure -------------------------------------
        if os.path.exists(resume):
            api.upload_file(path_or_fileobj=resume, path_in_repo=f"{STAGE}/latest.pt", repo_id=HF_REPO)
            print("[hub] checkpoint pushed")


if __name__ == "__main__":
    main()
