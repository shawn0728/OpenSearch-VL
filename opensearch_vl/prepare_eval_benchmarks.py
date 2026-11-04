#!/usr/bin/env python3
"""Build one FVQA-style parquet from Osilly/Vision-DeepResearch-Eval files.

The eval bundle stores ``question`` as ``image_id: x.jpg Question: ...`` and the
images as base64 in ``image_packed``. Output rows: ``data_id`` (``<bench>__<id>``),
``prompt``, ``images`` (raw bytes), ``category`` (the benchmark), ``answer``. The
sample is seeded per benchmark, so a larger --per-bench only adds cases: a small
sample is a prefix of a bigger one and finished trajectories stay reusable.

    python prepare_eval_benchmarks.py --src /root/datasets/vdr-eval/data \
        --out /root/datasets/eval/paper7_n50.parquet --per-bench 50
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import re
from pathlib import Path

import pandas as pd

# The paper's 7 benchmarks; BrowseComp-VL has two levels.
BENCHMARKS = {
    "simplevqa_300": "simplevqa",
    "testmini500_v1": "vdr",
    "mmsearch_vision_171": "mmsearch",
    "livevqa_webwatcher_subset_300": "livevqa",
    "bcvl_level1_199": "bcvl_l1",
    "bcvl_level2_200": "bcvl_l2",
    "fvqa_300": "fvqa",
    "infoseek_300": "infoseek",
}
PREFIX_RE = re.compile(r"^\s*image_id:.*?Question:\s*", re.S | re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bench", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20261104)
    args = ap.parse_args()

    rows, counts = [], {}
    for stem, bench in BENCHMARKS.items():
        df = pd.read_parquet(Path(args.src) / f"{stem}.parquet")
        order = list(range(len(df)))
        random.Random(f"{args.seed}:{bench}").shuffle(order)
        kept = 0
        for i in order:
            if kept >= args.per_bench:
                break
            row = df.iloc[i]
            packed = row["image_packed"]
            packed = json.loads(packed) if isinstance(packed, str) else list(packed)
            images = [{"bytes": base64.b64decode(x["data"])} for x in packed if x.get("data")]
            answer = row["answer"]
            if not images or answer is None or not str(answer).strip():
                continue
            case_id = re.sub(r"[^\w.\-]", "_", f"{bench}__{row['id']}")
            rows.append(
                {
                    "data_id": case_id,
                    "id": case_id,
                    "category": bench,
                    "data_source": stem,
                    "prompt": [{"role": "user", "content": PREFIX_RE.sub("", str(row["question"]))}],
                    "images": images,
                    "answer": str(answer),
                }
            )
            kept += 1
        counts[bench] = kept

    assert len({r["data_id"] for r in rows}) == len(rows), "duplicate case ids"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    print(f"wrote {len(rows)} cases to {out}: {counts}")


if __name__ == "__main__":
    main()
