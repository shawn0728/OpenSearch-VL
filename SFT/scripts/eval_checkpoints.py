#!/usr/bin/env python3
"""Validation loss of the base model and every SFT checkpoint on a held-out set.

Each model goes through LLaMA-Factory's own eval path, so preprocessing and the
loss match training. Selection rule, fixed before seeing results: the lowest
validation loss, but keep the final checkpoint unless the lowest beats it by more
than --keep-margin (relative), since small gaps on 200 samples are mostly noise.

Writes val_losses.json, metrics.json and metrics.jsonl to $OUTPUT_ROOT, and the
chosen adapter directory to --best-file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--best-file", required=True)
    ap.add_argument("--lanes", type=int, default=3)
    ap.add_argument("--keep-margin", type=float, default=0.005)
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(
        (p for p in Path(args.ckpt_root).glob("checkpoint-*") if re.fullmatch(r"checkpoint-\d+", p.name)),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not ckpts:
        print(f"no checkpoints under {args.ckpt_root}")
        return 1
    env = {**os.environ, "FORCE_TORCHRUN": "0", "WANDB_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    def run(name: str, adapter: str | None) -> float | None:
        out = work / name
        result = out / "eval_results.json"
        if not result.exists():
            cfg = {
                "model_name_or_path": args.model, "image_max_pixels": 262144, "trust_remote_code": True,
                "enable_liger_kernel": True, "stage": "sft", "do_train": False, "do_eval": True,
                "finetuning_type": "lora", "dataset_dir": args.dataset_dir, "eval_dataset": args.dataset_name,
                "template": "qwen3_5", "cutoff_len": 32000, "overwrite_cache": False,
                "preprocessing_num_workers": 16, "dataloader_num_workers": 2, "output_dir": str(out),
                "overwrite_output_dir": True, "per_device_eval_batch_size": 1, "predict_with_generate": False,
                "prediction_loss_only": True, "report_to": "none", "bf16": True, "seed": 42,
            }
            if adapter:
                cfg["adapter_name_or_path"] = adapter
            out.mkdir(parents=True, exist_ok=True)
            config = work / f"{name}.yaml"
            config.write_text(yaml.safe_dump(cfg))
            with open(work / f"{name}.log", "w") as log:
                code = subprocess.call(["llamafactory-cli", "train", str(config)], stdout=log, stderr=subprocess.STDOUT, env=env)
            if code != 0 or not result.exists():
                print(f"{name}: failed (exit {code}), see {work / (name + '.log')}", flush=True)
                return None
        loss = json.loads(result.read_text())["eval_loss"]
        print(f"{name}: val loss {loss:.4f}", flush=True)
        return loss

    losses: dict[str, float | None] = {"base": run("base", None)}  # also fills the preprocessing cache
    if losses["base"] is None:
        return 1
    with ThreadPoolExecutor(args.lanes) as pool:
        for ckpt, loss in zip(ckpts, pool.map(lambda c: run(c.name, str(c)), ckpts)):
            losses[ckpt.name] = loss
    for ckpt in ckpts:  # a long sample can OOM when lanes overlap; retry alone
        if losses[ckpt.name] is None:
            losses[ckpt.name] = run(ckpt.name, str(ckpt))
    if any(v is None for v in losses.values()):
        return 1

    final = ckpts[-1].name
    lowest = min((c.name for c in ckpts), key=lambda n: losses[n])
    best = lowest if losses[lowest] < losses[final] * (1 - args.keep_margin) else final
    best_path = str(Path(args.ckpt_root) / best)
    Path(args.best_file).write_text(best_path + "\n")
    print(f"lowest {lowest} ({losses[lowest]:.4f}), final {final} ({losses[final]:.4f}), chosen {best}", flush=True)
    if losses[best] >= losses["base"]:
        print(f"WARNING: chosen checkpoint is not better than the base model ({losses['base']:.4f})", flush=True)

    root = os.environ.get("OUTPUT_ROOT")
    if root:
        summary = {"losses": losses, "lowest": lowest, "final": final, "chosen": best, "adapter": best_path}
        Path(root, "val_losses.json").write_text(json.dumps(summary, indent=1))
        rows = [json.dumps({"__meta__": {"schema": 1, "run_id": ""}})]
        rows += [json.dumps({"step": int(c.name.split("-")[1]), "name": "val/loss", "value": losses[c.name]}) for c in ckpts]
        Path(root, "metrics.jsonl").write_text("\n".join(rows) + "\n")
        Path(root, "metrics.json").write_text(json.dumps({
            "val_loss_base": losses["base"], "val_loss_final": losses[final], "val_loss_chosen": losses[best],
            "chosen_step": int(best.split("-")[1]),
        }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
