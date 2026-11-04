"""Phase A: restructure the v1 HF repo.

Server-side (no re-upload):
  pd12m_train_full/  -> gt/
  pd12m_train_full_scores.csv -> gt_scores.csv
  contact_pd12m_train_full.png -> contact_gt.png
  delete: small/, mini/, bench/, trainval/, their sheets + contact sheets

Local (generated + uploaded): x2/ and x4/ shards.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from huggingface_hub import (
    CommitOperationCopy,
    CommitOperationDelete,
    HfApi,
    list_repo_files,
)

REPO = "Phips/lucid-pd12m-512"
BASE = Path("/media/phhofm/SSD/pd12m_v1")
HF = "/home/phhofm/.local/bin/hf"

api = HfApi()
files = list_repo_files(REPO, repo_type="dataset")
print("repo files:", len(files))

ops = []
gt = [f for f in files if f.startswith("pd12m_train_full/")]
for f in gt:
    ops.append(CommitOperationCopy(src_path_in_repo=f,
                                   path_in_repo=f.replace("pd12m_train_full/", "gt/", 1)))
if "pd12m_train_full_scores.csv" in files:
    ops.append(CommitOperationCopy(src_path_in_repo="pd12m_train_full_scores.csv",
                                   path_in_repo="gt_scores.csv"))
if "contact_pd12m_train_full.png" in files:
    ops.append(CommitOperationCopy(src_path_in_repo="contact_pd12m_train_full.png",
                                   path_in_repo="contact_gt.png"))

to_delete = [f for f in files if f.startswith(("pd12m_train_full/", "pd12m_train_small/",
                                               "pd12m_train_mini/", "pd12m_bench/",
                                               "pd12m_trainval/"))]
to_delete += [f for f in ("pd12m_train_full_scores.csv", "pd12m_train_small_scores.csv",
                          "pd12m_train_mini_scores.csv", "pd12m_bench_scores.csv",
                          "pd12m_trainval_scores.csv", "contact_pd12m_train_full.png",
                          "contact_pd12m_train_small.png", "contact_pd12m_train_mini.png",
                          "contact_pd12m_bench.png", "contact_pd12m_trainval.png")
              if f in files]
for f in to_delete:
    ops.append(CommitOperationDelete(path_in_repo=f))

print(f"copy ops: {len(gt)+2}, delete ops: {len(to_delete)}")
api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                  commit_message="restructure: gt/ layout, drop small/mini/bench/val")
print("server-side restructure done")
print("files now:", len(list_repo_files(REPO, repo_type="dataset")))
