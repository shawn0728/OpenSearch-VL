#!/usr/bin/env python3
"""Write an aria2c input file for (parts of) Hugging Face repos via a mirror.

aria2c with many connections per file is far faster than huggingface-cli /
hf_transfer on hosts with poor per-connection bandwidth. Files that already
exist locally with the expected size are skipped, so re-running is cheap.

Usage:
  python hf_aria2.py --out /tmp/dl.txt \\
      --repo model:Qwen/Qwen3.5-4B=/root/models/Qwen3.5-4B \\
      --repo dataset:OpenSearch-VL/Search-VL-SFT-36K=/root/datasets/sft36k \\
      --exclude '_plan_mode' --include-glob '*'
  aria2c -i /tmp/dl.txt -x 8 -s 8 -j 4 -c
"""
import argparse
import fnmatch
import json
import os
import urllib.parse
import urllib.request


def list_files(endpoint: str, kind: str, repo: str, rev: str) -> list[dict]:
    api = f"{endpoint}/api/{kind}s/{repo}/tree/{rev}?recursive=true"
    with urllib.request.urlopen(api, timeout=60) as r:
        return [x for x in json.load(r) if x.get("type") == "file"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.getenv("HF_ENDPOINT", "https://hf-mirror.com"))
    ap.add_argument("--repo", action="append", required=True,
                    help="kind:owner/name[@rev]=local_dir[:glob,glob]  (kind = model|dataset)")
    ap.add_argument("--exclude", action="append", default=[], help="substring to skip")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lines, n_skip, total = [], 0, 0
    for spec in args.repo:
        kind_repo, local = spec.split("=", 1)
        globs = ["*"]
        if ":" in local:
            local, g = local.split(":", 1)
            globs = g.split(",")
        kind, repo = kind_repo.split(":", 1)
        repo, _, rev = repo.partition("@")
        rev = rev or "main"
        for f in list_files(args.endpoint, kind, repo, rev):
            path, size = f["path"], f.get("size", 0)
            if any(x in path for x in args.exclude) or not any(fnmatch.fnmatch(path, g) for g in globs):
                continue
            dst = os.path.join(local, path)
            if os.path.isfile(dst) and os.path.getsize(dst) == size and not os.path.exists(dst + ".aria2"):
                n_skip += 1
                continue
            prefix = "datasets/" if kind == "dataset" else ""
            url = f"{args.endpoint}/{prefix}{repo}/resolve/{rev}/{urllib.parse.quote(path)}"
            lines += [url, f"  dir={os.path.dirname(dst)}", f"  out={os.path.basename(dst)}"]
            total += size
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{len(lines) // 3} files to fetch ({total / 1e9:.2f} GB), {n_skip} already complete -> {args.out}")


if __name__ == "__main__":
    main()
