#!/usr/bin/env python3
"""Call the running layout-parsing service on real SFT document images.

Prints the raw block labels next to what search.layout_parsing keeps, so a label
the parser drops (and an empty OCR result the agent would see) is visible. Exits
non-zero when the service errors or no text survives on any image.

    python tests/probe_ocr_service.py --subset /root/datasets/sft_subset_2048 --samples 4
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--url", default="http://127.0.0.1:8080/layout-parsing")
    args = ap.parse_args()
    os.environ.setdefault("LAYOUT_PARSING_API_URL", args.url)

    from opensearch_infer import search

    data = json.load(open(os.path.join(args.subset, "sft_subset.json"), encoding="utf-8"))
    rows = [json.loads(line) for line in open(os.path.join(args.subset, "manifest.jsonl"), encoding="utf-8")]
    # Trajectories whose first tool call OCRs the original image, so it holds text.
    picked = [
        d["images"][0]
        for d, m in zip(data, rows)
        if m["tools"] and m["tools"][0] == "layout_parsing" and "SUCCESS" in d["conversations"][2]["value"]
    ][: args.samples]
    if not picked:
        print("no document images found")
        return 1

    labels, kept_total, failures = collections.Counter(), 0, 0
    for path in picked:
        start = time.time()
        with open(path, "rb") as fh:
            payload = {"file": base64.b64encode(fh.read()).decode("ascii"), "fileType": 1}
        response = requests.post(args.url, json=payload, timeout=300)
        seconds = time.time() - start
        if response.status_code != 200:
            failures += 1
            print(f"HTTP {response.status_code}: {response.text[:300]}")
            continue
        blocks = response.json()["result"]["layoutParsingResults"][0]["prunedResult"]["parsing_res_list"]
        labels.update(b.get("block_label") for b in blocks)
        result = search.layout_parsing(path)
        kept = result.get("rec_texts", [])
        kept_total += len(kept)
        print(f"{os.path.basename(path)}: {len(blocks)} blocks, parser keeps {len(kept)}, {seconds:.1f}s")
        print("   text:", " | ".join(kept)[:200])
    print("labels seen:", dict(labels.most_common()))
    return 1 if failures or not kept_total else 0


if __name__ == "__main__":
    sys.exit(main())
