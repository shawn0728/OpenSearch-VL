#!/usr/bin/env python3
"""Compare judged runs: accuracy with Wilson CIs, paired counts, agent behaviour.

    python summarize_sft_eval.py --run base=/path/base --run sft=/path/sft \
        --baseline base --out /path/summary
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def sign_test_p(b: int, c: int) -> float:
    """Exact two-sided sign test on the discordant pairs (McNemar exact)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def load_run(path: Path) -> dict[str, dict]:
    judged = {}
    for details in path.glob("eval_report_*_details.jsonl"):
        for line in open(details, encoding="utf-8"):
            r = json.loads(line)
            judged[r["case_id"]] = r
    cases = {}
    for f in path.glob("*_trajectory.json"):
        t = json.load(open(f, encoding="utf-8"))
        cid = t["case_id"]
        turns = t.get("turns", [])
        tools, errors, xml, json_calls, unparsed = [], 0, 0, 0, 0
        for turn in turns:
            if turn.get("tool_call"):
                tools.append(json.loads(turn["tool_call"]).get("name", "?"))
            if "Tool execution error" in (turn.get("tool_output") or ""):
                errors += 1
            text = turn.get("response_text") or ""
            if "<function=" in text:
                xml += 1
            elif "<tool_call>" in text:
                json_calls += 1
            if ("<function=" in text or "<tool_call>" in text) and not turn.get("tool_call"):
                unparsed += 1  # malformed call that could not be executed
        j = judged.get(cid)
        cases[cid] = {
            "bench": t.get("category", cid.split("__")[0]),
            "acc": None if j is None or j.get("judge_failed") else j["acc"],
            "stop": t.get("stop_reason", "?"),
            "tools": tools,
            "tool_errors": errors,
            "xml_calls": xml,
            "json_calls": json_calls,
            "unparsed_calls": unparsed,
            "turns": len(turns),
        }
    return cases


def metrics(cases: list[dict]) -> dict:
    judged = [c for c in cases if c["acc"] is not None]
    k, n = sum(c["acc"] for c in judged), len(judged)
    lo, hi = wilson(k, n)
    calls = sum(len(c["tools"]) for c in cases)
    return {
        "cases": len(cases),
        "judged": n,
        "correct": k,
        "acc": k / n if n else 0.0,
        "ci95": [lo, hi],
        "finished_with_response": sum(c["stop"] == "response" for c in cases) / max(1, len(cases)),
        "stop_reasons": dict(collections.Counter(c["stop"] for c in cases).most_common()),
        "tool_calls_per_case": calls / max(1, len(cases)),
        "cases_with_no_tool": sum(not c["tools"] for c in cases) / max(1, len(cases)),
        "tool_error_rate": sum(c["tool_errors"] for c in cases) / max(1, calls),
        "xml_format_calls": sum(c["xml_calls"] for c in cases),
        "json_format_calls": sum(c["json_calls"] for c in cases),
        "unparsed_calls": sum(c["unparsed_calls"] for c in cases),
        "tool_usage": dict(collections.Counter(t for c in cases for t in c["tools"]).most_common()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="tag=dir")
    ap.add_argument("--baseline", action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = {tag: load_run(Path(p)) for tag, p in (r.split("=", 1) for r in args.run)}
    report: dict = {"overall": {}, "by_bench": {}, "paired": {}}
    benches = sorted({c["bench"] for cases in runs.values() for c in cases.values()})
    for tag, cases in runs.items():
        report["overall"][tag] = metrics(list(cases.values()))
        report["by_bench"][tag] = {
            b: metrics([c for c in cases.values() if c["bench"] == b]) for b in benches
        }

    for base in args.baseline:
        for tag, cases in runs.items():
            if tag == base:
                continue
            common = [
                cid
                for cid in cases
                if cid in runs[base]
                and cases[cid]["acc"] is not None
                and runs[base][cid]["acc"] is not None
            ]
            wins = sum(cases[c]["acc"] == 1 and runs[base][c]["acc"] == 0 for c in common)
            losses = sum(cases[c]["acc"] == 0 and runs[base][c]["acc"] == 1 for c in common)
            report["paired"][f"{tag}_vs_{base}"] = {
                "cases": len(common),
                "only_this_correct": wins,
                "only_baseline_correct": losses,
                "diff": (wins - losses) / max(1, len(common)),
                "exact_mcnemar_p": sign_test_p(wins, losses),
            }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out / "summary.json", "w"), ensure_ascii=False, indent=1)

    lines = ["| run | bench | n | acc | 95% CI | finished | tools/case | no-tool | tool err | json/xml/unparsed |", "|---|---|---|---|---|---|---|---|---|---|"]
    for tag in runs:
        rows = [("ALL", report["overall"][tag])] + list(report["by_bench"][tag].items())
        for b, m in rows:
            lines.append(
                f"| {tag} | {b} | {m['judged']} | {m['acc']:.1%} | {m['ci95'][0]:.1%}–{m['ci95'][1]:.1%} "
                f"| {m['finished_with_response']:.0%} | {m['tool_calls_per_case']:.1f} "
                f"| {m['cases_with_no_tool']:.0%} | {m['tool_error_rate']:.0%} "
                f"| {m['json_format_calls']}/{m['xml_format_calls']}/{m['unparsed_calls']} |"
            )
    text = "\n".join(lines) + "\n\n" + json.dumps(report["paired"], indent=1) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)

    if os.getenv("OUTPUT_ROOT"):
        flat = {}
        for tag, m in report["overall"].items():
            flat[f"{tag}_acc"] = m["acc"]
            flat[f"{tag}_judged"] = m["judged"]
            flat[f"{tag}_finished"] = m["finished_with_response"]
        for key, p in report["paired"].items():
            flat[f"{key}_diff"] = p["diff"]
            flat[f"{key}_p"] = p["exact_mcnemar_p"]
        json.dump(flat, open(Path(os.environ["OUTPUT_ROOT"]) / "metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
