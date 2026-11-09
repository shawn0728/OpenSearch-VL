#!/usr/bin/env python3
"""Select a stratified, validated SFT subset from Search-VL-SFT-36K.

Pipeline
--------
1. Parse every trajectory of the 7 "normal" subsets (the ``*_plan_mode``
   variants are not used, matching the paper's SFT config).
2. Validate each sample and record a rejection reason:
   - structure: human → (gpt tool_call → observation)* → gpt final; each
     intermediate gpt turn has exactly one parsable ``<tool_call>`` naming a
     declared tool; balanced ``<think>`` tags;
   - proper ending: final gpt turn has no tool call and ends with exactly one
     non-empty ``<response>…</response>`` block (the protocol the system
     prompt and the RL agent/reward use; ``\\boxed{}`` is optional, only an
     unbalanced one is rejected);
   - images: ``<image>`` count == len(images); every file exists, decodes,
     is ≥ 28 px per side and has aspect ratio < 200 (Qwen processor limits);
   - loops: the same tool call (name + arguments) twice in one trajectory;
   - length: text tokens (target model's tokenizer) + vision tokens (smart_resize at
     ``--image-max-pixels``) + per-message overhead ≤ ``--cutoff-len``
     (with ``--length-margin`` slack) — samples are dropped, never truncated;
   - test overlap: question text (exact normalised match, or 5-gram Jaccard
     ≥ 0.7 for questions with ≥ 8 words) or primary-image perceptual hash
     (pHash Hamming ≤ 6) against every eval parquet in ``--eval``.
3. Categorise by tools used (priority order):
   enhancement (sharpen / super_resolution / perspective_correct)
   > crop_ocr (crop / layout_parsing) > retrieval (search tools only).
4. Stratified sampling per category (default 60/30/10 of ``--n``): strata are
   (source subset, tool-call-count bin) with quota ∝ sqrt(stratum size) so
   large subsets do not dominate; inside a stratum, round-robin over tool
   combinations (rarest first) for tool coverage. Shortfalls in a category
   are redistributed to the others and reported.

Outputs (in ``--out``): ``sft_subset.json`` (LLaMA-Factory sharegpt records,
image paths made absolute), ``manifest.jsonl`` (one line per selected sample),
``report.json`` (rejection counts, pool sizes, distributions, overlap examples).
"""
from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import math
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SOURCES = {
    "fvqa": "fvqa_llama_factory_clean.json",
    "palace": "palace_llama_factory_filtered.json",
    "webqa": "webqa_llama_factory_filtered.json",
    "wiki_art": "wikiart_llama_factory_filtered.json",
    "wiki_en": "wiki_en_llama_factory_filtered.json",
    "wiki_zh": "wiki_zh_llama_factory_filtered.json",
    "livevqa": "livevqa_llama_factory_filtered.json",
}
ENHANCE_TOOLS = {"sharpen", "super_resolution", "perspective_correct"}
CROP_OCR_TOOLS = {"crop", "layout_parsing"}
# Heuristic, report-only: observation looks like a failed tool call.
TOOL_ERROR_RE = re.compile(r"tool execution error|\berror\b|failed to|timed? ?out|no results? found", re.I)
RETRIEVAL_TOOLS = {"text_search", "image_search", "web_search", "visit"}
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
EVAL_Q_PREFIX_RE = re.compile(r"^\s*image_id:.*?Question:\s*", re.S | re.I)
WORD_RE = re.compile(r"\w+", re.U)


# ─────────────────────────── helpers ────────────────────────────────────
def norm_text(s: str) -> str:
    s = s.replace("<image>", " ").lower()
    return " ".join(WORD_RE.findall(s))


def shingles(words: list[str], k: int = 5) -> set[str]:
    return {" ".join(words[i:i + k]) for i in range(max(0, len(words) - k + 1))}


def boxed_spans(text: str) -> list[str]:
    """Contents of every brace-balanced \\boxed{...}; None-like '' if unbalanced."""
    out, i = [], 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            return out
        k, depth = j + len("\\boxed{"), 1
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        out.append(text[j + 7:k - 1] if depth == 0 else None)
        i = k


def smart_resize_tokens(h: int, w: int, max_pixels: int, min_pixels: int,
                        factor: int) -> int:
    """Vision tokens after Qwen-VL smart_resize (factor = patch * merge)."""
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return (hb // factor) * (wb // factor)


def phash_int(img) -> int:
    import imagehash
    return int(str(imagehash.phash(img.convert("RGB"))), 16)


# ─────────────────────── per-sample structure check ──────────────────────
def check_structure(sample: dict) -> tuple[str | None, dict]:
    conv = sample.get("conversations") or []
    info = {"tools": [], "n_calls": 0, "tool_error": False}
    seen_calls = set()
    if len(conv) < 2 or conv[0].get("from") != "human":
        return "struct:first_not_human", info
    declared = set()
    try:
        tools_field = sample.get("tools") or "[]"
        tl = json.loads(tools_field) if isinstance(tools_field, str) else tools_field
        for t in tl:
            fn = t.get("function", t)
            declared.add(fn.get("name"))
    except Exception:
        return "struct:bad_tools_field", info
    expect = "gpt"
    for idx, msg in enumerate(conv[1:], 1):
        role, val = msg.get("from"), msg.get("value") or ""
        if role != expect:
            return f"struct:role_order({role}@{idx})", info
        if role == "gpt":
            if val.count("<think>") != val.count("</think>"):
                return "struct:unbalanced_think", info
            calls = TOOL_CALL_RE.findall(val)
            last = idx == len(conv) - 1
            if last:
                if calls or "<tool_call>" in val:
                    return "end:final_turn_has_tool_call", info
                # Protocol (system prompt + RL agent/reward): the final answer is
                # the content of <response>…</response>; \boxed{} is optional
                # (the prompt's own example writes a bare `boxed{…}`).
                tail = val.rstrip()
                if not tail.endswith("</response>") or "<response>" not in tail:
                    return "end:no_response_block", info
                if tail.count("<response>") != 1 or tail.count("</response>") != 1:
                    return "end:multiple_response_blocks", info
                body = tail[tail.find("<response>") + len("<response>"):-len("</response>")]
                if len(body.strip()) < 2:
                    return "end:empty_response", info
                spans = boxed_spans(body)
                if any(s is None for s in spans):
                    return "end:boxed_unbalanced", info
                info["boxed"] = "backslash" if spans else ("bare" if "boxed{" in body else "none")
            else:
                if len(calls) != 1:
                    return f"struct:tool_calls_per_turn_{len(calls)}", info
                try:
                    call = json.loads(calls[0])
                except Exception:
                    return "struct:tool_call_bad_json", info
                name = call.get("name")
                if not isinstance(call.get("arguments"), dict):
                    return "struct:tool_call_no_args", info
                if declared and name not in declared:
                    return f"struct:undeclared_tool({name})", info
                # The same call twice = the agent is going in circles; the Qwen3.5 SFT eval's
                # 13 max-turn cases (all wrong) were such loops. Don't teach it.
                key = (name, json.dumps(call["arguments"], sort_keys=True, ensure_ascii=False))
                if key in seen_calls:
                    return "loop:repeated_tool_call", info
                seen_calls.add(key)
                info["tools"].append(name)
            expect = "observation"
        else:  # observation
            if TOOL_ERROR_RE.search(val[:600]):
                info["tool_error"] = True  # report-only: error-then-recover examples
            expect = "gpt"
    if conv[-1].get("from") != "gpt":
        return "end:last_not_gpt", info
    info["n_calls"] = len(info["tools"])
    return None, info


def categorize(tools: list[str]) -> str | None:
    s = set(tools)
    if s & ENHANCE_TOOLS:
        return "enhancement"
    if s & CROP_OCR_TOOLS:
        return "crop_ocr"
    if s & RETRIEVAL_TOOLS:
        return "retrieval"
    return None  # no tool / python-only


# ─────────────────── image check (runs in worker processes) ──────────────
def check_images(args) -> tuple[str | None, list, int | None]:
    paths, want_hash = args
    from PIL import Image
    sizes, ph = [], None
    for i, p in enumerate(paths):
        if not os.path.isfile(p):
            return "img:missing", [], None
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
                w, h = im.size
                if i == 0 and want_hash:
                    ph = phash_int(im)
        except Exception:
            return "img:undecodable", [], None
        if min(w, h) < 28:
            return "img:too_small", [], None
        if max(w, h) / max(1, min(w, h)) >= 200:
            return "img:bad_aspect", [], None
        sizes.append((h, w))
    return None, sizes, ph


def load_eval(paths: list[str]) -> tuple[list[dict], dict]:
    """Return eval items [{bench, id, qnorm, phash}] from parquet files."""
    import pyarrow.parquet as pq
    from PIL import Image
    items, per_bench = [], collections.Counter()
    for p in paths:
        bench = Path(p).stem
        t = pq.read_table(p)
        cols = set(t.schema.names)
        qcol = next((c for c in ("question", "query", "prompt", "problem") if c in cols), None)
        for row in t.to_pylist():
            q = EVAL_Q_PREFIX_RE.sub("", str(row.get(qcol, "") or ""))
            imgs = []
            if row.get("image_packed"):
                packed = row["image_packed"]
                packed = json.loads(packed) if isinstance(packed, str) else packed
                imgs = [base64.b64decode(x["data"]) for x in packed if x.get("data")]
            else:
                for c in ("image", "images", "img"):
                    v = row.get(c)
                    vs = v if isinstance(v, list) else [v]
                    for x in vs:
                        if isinstance(x, dict) and x.get("bytes"):
                            imgs.append(x["bytes"])
                        elif isinstance(x, (bytes, bytearray)):
                            imgs.append(bytes(x))
            hashes = []
            for b in imgs:
                try:
                    hashes.append(phash_int(Image.open(io.BytesIO(b))))
                except Exception:
                    pass
            items.append({"bench": bench, "id": str(row.get("id", "")),
                          "qnorm": norm_text(q), "phashes": hashes, "q": q[:200]})
            per_bench[bench] += 1
    return items, dict(per_bench)


class OverlapIndex:
    def __init__(self, items: list[dict], jaccard: float, min_words: int, max_ham: int):
        import numpy as np
        self.items, self.jaccard, self.min_words, self.max_ham = items, jaccard, min_words, max_ham
        self.exact = collections.defaultdict(list)
        self.sh_index = collections.defaultdict(set)
        self.sh_sets = {}
        hashes, owners = [], []
        for i, it in enumerate(items):
            self.exact[it["qnorm"]].append(i)
            words = it["qnorm"].split()
            if len(words) >= min_words:
                s = shingles(words)
                self.sh_sets[i] = s
                for g in s:
                    self.sh_index[g].add(i)
            for h in it["phashes"]:
                hashes.append(h)
                owners.append(i)
        self.hashes = np.array(hashes, dtype=np.uint64)
        self.owners = owners

    def match(self, qnorm: str, ph: int | None) -> tuple[str, dict] | None:
        import numpy as np
        if qnorm and qnorm in self.exact:
            return "text_exact", self.items[self.exact[qnorm][0]]
        words = qnorm.split()
        if len(words) >= self.min_words:
            s = shingles(words)
            cand = collections.Counter(i for g in s for i in self.sh_index.get(g, ()))
            for i, inter in cand.most_common(5):
                union = len(s | self.sh_sets[i])
                if union and inter / union >= self.jaccard:
                    return "text_jaccard", self.items[i]
        if ph is not None and len(self.hashes):
            x = np.bitwise_xor(self.hashes, np.uint64(ph))
            ham = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
            j = int(ham.argmin())
            if ham[j] <= self.max_ham:
                return f"image_phash(d={int(ham[j])})", self.items[self.owners[j]]
        return None


# ───────────────────────────── selection ─────────────────────────────────
def turn_bin(n: int) -> str:
    return str(n) if n < 6 else "6+"


def sample_category(pool: list[dict], k: int, rng: random.Random) -> list[dict]:
    if k >= len(pool):
        return list(pool)
    strata = collections.defaultdict(list)
    for r in pool:
        strata[(r["source"], turn_bin(r["n_calls"]))].append(r)
    # quota ∝ sqrt(size), capped by size; iterate until k is distributed.
    quota = {s: 0 for s in strata}
    remaining = k
    while remaining > 0:
        open_ = [s for s in strata if quota[s] < len(strata[s])]
        if not open_:
            break
        w = {s: math.sqrt(len(strata[s])) for s in open_}
        tot = sum(w.values())
        alloc = {s: min(len(strata[s]) - quota[s], int(remaining * w[s] / tot)) for s in open_}
        if sum(alloc.values()) == 0:  # hand out leftovers one by one, biggest weight first
            for s in sorted(open_, key=lambda s: -w[s])[:remaining]:
                alloc[s] = 1
        for s, a in alloc.items():
            quota[s] += a
            remaining -= a
    picked = []
    for s, q in quota.items():
        by_combo = collections.defaultdict(list)
        for r in strata[s]:
            by_combo[tuple(sorted(set(r["tools"])))].append(r)
        for lst in by_combo.values():
            rng.shuffle(lst)
        combos = sorted(by_combo, key=lambda c: (len(by_combo[c]), c))  # rarest first
        got = []
        while len(got) < q:
            for c in combos:
                if by_combo[c] and len(got) < q:
                    got.append(by_combo[c].pop())
        picked.extend(got)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--eval", nargs="*", default=[], help="eval parquet files for overlap check")
    ap.add_argument("--model", required=True, help="tokenizer / processor path (the model being fine-tuned)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-name", default="imagentrl_sft_2048", help="key in the written dataset_info.json")
    ap.add_argument("--exclude-manifest", default=None,
                    help="manifest.jsonl of a subset to hold out from: its samples and overlaps are removed")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--ratios", default="retrieval=0.6,crop_ocr=0.3,enhancement=0.1")
    ap.add_argument("--cutoff-len", type=int, default=32000)
    ap.add_argument("--length-margin", type=float, default=0.97)
    ap.add_argument("--image-max-pixels", type=int, default=262144)
    ap.add_argument("--image-min-pixels", type=int, default=32 * 32)
    ap.add_argument("--msg-overhead", type=int, default=6, help="template tokens per message")
    ap.add_argument("--jaccard", type=float, default=0.7)
    ap.add_argument("--min-words", type=int, default=8)
    ap.add_argument("--max-ham", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 4))
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        ip = AutoProcessor.from_pretrained(args.model, trust_remote_code=True).image_processor
        factor = int(ip.patch_size) * int(getattr(ip, "merge_size", 2))
    except Exception:
        factor = 32
    print(f"[init] vision factor={factor} cutoff={args.cutoff_len} max_pixels={args.image_max_pixels}", flush=True)

    ratios = {k: float(v) for k, v in (kv.split("=") for kv in args.ratios.split(","))}
    reasons = collections.Counter()
    reasons_by_src = collections.defaultdict(collections.Counter)
    tool_hist = collections.Counter()
    records = []

    # 1. structure + categorisation
    for src, fname in SOURCES.items():
        src_dir = Path(args.data_root) / src
        data = json.load(open(src_dir / fname, encoding="utf-8"))
        print(f"[load] {src}: {len(data)} samples", flush=True)
        for idx, s in enumerate(data):
            sid = f"{src}:{idx}"
            err, info = check_structure(s)
            for t in info["tools"]:
                tool_hist[t] += 1
            n_img_tags = sum((m.get("value") or "").count("<image>") for m in s.get("conversations", []))
            imgs = s.get("images") or []
            if err is None and n_img_tags != len(imgs):
                err = "img:tag_count_mismatch"
            cat = categorize(info["tools"]) if err is None else None
            if err is None and cat is None:
                err = "cat:no_category_tools"
            if err:
                reasons[err.split("(")[0]] += 1
                reasons_by_src[src][err.split("(")[0]] += 1
                continue
            abs_imgs = [str((src_dir / p).resolve()) if not os.path.isabs(p) else p for p in imgs]
            records.append({"id": sid, "source": src, "cat": cat, "tools": info["tools"],
                            "n_calls": info["n_calls"], "boxed": info.get("boxed", "none"),
                            "tool_error": info["tool_error"],
                            "images": abs_imgs, "sample": s})
    print(f"[struct] {len(records)} passed structure; reasons={dict(reasons)}", flush=True)
    unknown = set(tool_hist) - ENHANCE_TOOLS - CROP_OCR_TOOLS - RETRIEVAL_TOOLS - {"PythonInterpreter", "python_interpreter"}
    if unknown:
        print(f"[warn] tool names outside known sets: {sorted(unknown)}", flush=True)

    # 2. images (+ pHash of primary image)
    with ProcessPoolExecutor(args.workers) as ex:
        img_results = list(ex.map(check_images, [(r["images"], True) for r in records], chunksize=64))
    kept = []
    for r, (err, sizes, ph) in zip(records, img_results):
        if err:
            reasons[err] += 1
            reasons_by_src[r["source"]][err] += 1
            continue
        r["img_sizes"], r["phash"] = sizes, ph
        kept.append(r)
    records = kept
    print(f"[images] {len(records)} passed image checks", flush=True)

    # Held-out selection: drop the given subset, and later anything sharing a
    # question or primary image with it (handled like a test set below).
    train_items = []
    if args.exclude_manifest:
        with open(args.exclude_manifest, encoding="utf-8") as fh:
            excluded = {json.loads(line)["id"] for line in fh}
        kept = []
        for r in records:
            if r["id"] not in excluded:
                kept.append(r)
                continue
            q = norm_text(r["sample"]["conversations"][0]["value"])
            train_items.append({"bench": "exclude_manifest", "id": r["id"], "qnorm": q,
                                "phashes": [] if r["phash"] is None else [r["phash"]], "q": q[:200]})
            reasons["excluded:manifest"] += 1
        assert len(train_items) == len(excluded), (
            f"only {len(train_items)}/{len(excluded)} excluded samples found; their overlap check would be incomplete")
        records = kept
        print(f"[exclude] {len(train_items)} removed, {len(records)} remain", flush=True)

    # 3. length
    limit = int(args.cutoff_len * args.length_margin)
    sys_cache = {}
    texts = [[m.get("value") or "" for m in r["sample"]["conversations"]] for r in records]
    flat = [t.replace("<image>", "") for ts in texts for t in ts]
    lens = tok(flat, add_special_tokens=False)["input_ids"]
    pos = 0
    kept, length_stats = [], []
    for r, ts in zip(records, texts):
        n_msg_tok = sum(len(x) for x in lens[pos:pos + len(ts)])
        pos += len(ts)
        s = r["sample"]
        key = (s.get("system") or "") + "\x00" + (s.get("tools") if isinstance(s.get("tools"), str) else json.dumps(s.get("tools")))
        if key not in sys_cache:
            sys_cache[key] = len(tok(key.replace("\x00", "\n"), add_special_tokens=False)["input_ids"])
        vis = sum(smart_resize_tokens(h, w, args.image_max_pixels, args.image_min_pixels, factor) + 2
                  for h, w in r["img_sizes"])
        total = sys_cache[key] + n_msg_tok + vis + args.msg_overhead * (len(ts) + 1)
        r["tokens"] = total
        if total > limit:
            reasons["len:over_cutoff"] += 1
            reasons_by_src[r["source"]]["len:over_cutoff"] += 1
            continue
        length_stats.append(total)
        kept.append(r)
    records = kept
    print(f"[length] {len(records)} within {limit} tokens", flush=True)

    # 4. test-set overlap
    overlap_examples = []
    eval_counts = {}
    if args.eval or train_items:
        items, eval_counts = load_eval(args.eval)
        print(f"[eval] {len(items)} eval items from {eval_counts}", flush=True)
        items += train_items
        idx = OverlapIndex(items, args.jaccard, args.min_words, args.max_ham)
        kept = []
        for r in records:
            q = norm_text(r["sample"]["conversations"][0]["value"])
            m = idx.match(q, r["phash"])
            if m:
                kind, it = m
                key = "overlap:" + kind.split("(")[0]
                reasons[key] += 1
                reasons_by_src[r["source"]][key] += 1
                if len(overlap_examples) < 200:
                    overlap_examples.append({"train_id": r["id"], "kind": kind, "bench": it["bench"],
                                             "eval_id": it["id"], "train_q": q[:200], "eval_q": it["q"]})
                continue
            kept.append(r)
        records = kept
        print(f"[overlap] {len(records)} after removing test overlap", flush=True)

    # 5. stratified selection
    rng = random.Random(args.seed)
    pools = collections.defaultdict(list)
    for r in sorted(records, key=lambda r: r["id"]):
        pools[r["cat"]].append(r)
    targets = {c: round(args.n * ratios.get(c, 0)) for c in ratios}
    targets[max(targets, key=targets.get)] += args.n - sum(targets.values())
    shortfall = {c: max(0, targets[c] - len(pools[c])) for c in targets}
    extra = sum(shortfall.values())
    while extra > 0:
        roomy = [c for c in targets if len(pools[c]) > targets[c]]
        if not roomy:
            break
        w = sum(ratios[c] for c in roomy)
        for c in roomy:
            add = min(len(pools[c]) - targets[c], max(1, round(extra * ratios[c] / w)))
            targets[c] += add
            extra -= add
            if extra <= 0:
                break
    targets = {c: min(t, len(pools[c])) for c, t in targets.items()}
    selected = []
    for c, k in targets.items():
        selected.extend(sample_category(pools[c], k, rng))
    rng.shuffle(selected)

    # 6. write outputs
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for r in selected:
        s = dict(r["sample"])
        s["images"] = r["images"]
        recs.append(s)
    json.dump(recs, open(out / "sft_subset.json", "w", encoding="utf-8"), ensure_ascii=False)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps({k: r[k] for k in ("id", "source", "cat", "n_calls", "tools", "tokens")},
                               ensure_ascii=False) + "\n")

    def dist(key):
        return dict(collections.Counter(key(r) for r in selected).most_common())

    sel_tokens = sorted(r["tokens"] for r in selected)
    report = {
        "params": vars(args),
        "vision_factor": factor,
        "tool_name_histogram_all": dict(tool_hist.most_common()),
        "rejections_total": dict(reasons.most_common()),
        "rejections_by_source": {s: dict(c.most_common()) for s, c in reasons_by_src.items()},
        "eval_items_per_bench": eval_counts,
        "pool_sizes": {c: len(v) for c, v in pools.items()},
        "targets_after_redistribution": targets,
        "shortfall_vs_requested": shortfall,
        "selected_total": len(selected),
        "selected_by_category": dist(lambda r: r["cat"]),
        "selected_by_source": dist(lambda r: r["source"]),
        "selected_by_category_source": dist(lambda r: f"{r['cat']}/{r['source']}"),
        "selected_tool_calls_hist": dist(lambda r: turn_bin(r["n_calls"])),
        "selected_answer_boxed_style": dist(lambda r: r["boxed"]),
        "selected_with_tool_error_then_response": sum(r["tool_error"] for r in selected),
        "selected_tool_usage": dict(collections.Counter(t for r in selected for t in r["tools"]).most_common()),
        "selected_samples_using_tool": dict(collections.Counter(t for r in selected for t in set(r["tools"])).most_common()),
        "selected_tokens": {"min": sel_tokens[0], "p50": sel_tokens[len(sel_tokens) // 2],
                            "p95": sel_tokens[int(len(sel_tokens) * 0.95)], "max": sel_tokens[-1]} if sel_tokens else {},
        "overlap_examples": overlap_examples,
    }
    json.dump(report, open(out / "report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # LLaMA-Factory registry next to the data (dataset_dir = --out).
    json.dump({args.dataset_name: {
        "file_name": "sft_subset.json", "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images", "system": "system", "tools": "tools"},
        "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human",
                 "assistant_tag": "gpt", "observation_tag": "observation"},
    }}, open(out / "dataset_info.json", "w", encoding="utf-8"), indent=1)
    if os.getenv("OUTPUT_ROOT"):  # projtool run dir → synced back to the local output/ mirror
        import shutil
        for name in ("report.json", "manifest.jsonl"):
            shutil.copy(out / name, Path(os.environ["OUTPUT_ROOT"]) / f"selection_{name}")
        json.dump({"selected_total": len(selected), **{f"selected_{c}": n for c, n in report["selected_by_category"].items()},
                   **{f"pool_{c}": n for c, n in report["pool_sizes"].items()}},
                  open(Path(os.environ["OUTPUT_ROOT"]) / "metrics.json", "w"))
    print(json.dumps({k: report[k] for k in ("rejections_total", "pool_sizes", "targets_after_redistribution",
                                             "selected_by_category", "selected_by_source",
                                             "selected_tool_calls_hist", "selected_samples_using_tool",
                                             "selected_tokens")}, ensure_ascii=False, indent=1), flush=True)
    print(f"[done] wrote {len(selected)} samples to {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
