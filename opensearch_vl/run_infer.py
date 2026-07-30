#!/usr/bin/env python3
"""Unified entrypoint for OpenSearch-VL inference.

Usage examples
--------------

Run the dense 8B model on the FVQA training split using two GPUs::

    python run_infer.py --model 8b --gpus 0,1 \
        --data-path /data/fvqa_train.parquet \
        --output-dir ./outputs/fvqa_train_8b

Use Claude Opus 4.5 (no local GPUs needed; requires CLAUDE_API_* env vars)::

    python run_infer.py --model claude \
        --data-path /data/fvqa_test.parquet \
        --output-dir ./outputs/fvqa_test_claude

Use MiniMax-M3 (no local GPUs needed; requires MINIMAX_API_KEY)::

    python run_infer.py --model minimax-m3 \
        --data-path /data/fvqa_test.parquet \
        --output-dir ./outputs/fvqa_test_minimax_m3

Switch to the MoE 30B-A3B variant::

    python run_infer.py --model 30b-a3b --gpus 0,1,2,3 \
        --checkpoint /models/Qwen3-VL-30B-A3B-Instruct \
        --data-path /data/fvqa_train.parquet \
        --output-dir ./outputs/fvqa_train_30b_a3b

The model registry is defined in ``opensearch_infer/config.py``. All
default values pull from environment variables so the codebase is safe
to open-source.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from typing import Optional

import pandas as pd

from opensearch_infer import config, pipeline
from opensearch_infer.runners import InferenceConfig, build_runner


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "OpenSearch-VL Visual Investigation Agent. Supported tools: "
            "web_search, text_search, image_search, crop, layout_parsing, "
            "perspective_correct, super_resolution, sharpen."
        )
    )
    parser.add_argument(
        "--model",
        choices=sorted(config.MODEL_REGISTRY.keys()),
        default="8b",
        help="Model variant to run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Override the checkpoint path or HuggingFace id. Defaults to "
            "the per-model env variable, then the registry default."
        ),
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help=(
            "Comma-separated CUDA device ids for Qwen3-VL. Single id uses "
            "single-GPU placement; multiple ids enable model parallelism "
            'via device_map="auto".'
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Floating point dtype for Qwen3-VL weights.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Parquet file with FVQA-style rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where per-case trajectories and artifacts are written.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Tag stored in trajectories; controls intermediate filename prefix.",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="First row index (inclusive)."
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last row index (exclusive). Defaults to len(df).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N rows starting from --start.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Restrict to rows whose 'category' column equals this value.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Per-turn generation cap.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 disables sampling).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Root logger level.",
    )
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        format="[%(asctime)s][%(levelname)5s][%(name)s] %(message)s",
        level=getattr(logging, level),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    logger = logging.getLogger("run_infer")

    runner = build_runner(
        model_name=args.model,
        checkpoint=args.checkpoint,
        gpus=args.gpus,
        dtype=args.dtype,
    )
    logger.info("Model: %s", runner.display_name)

    try:
        runner.load()
    except Exception as exc:
        logger.error("Failed to initialize runner: %s", exc, exc_info=True)
        return 1

    if not os.path.exists(args.data_path):
        logger.error("Data file not found: %s", args.data_path)
        return 1

    df = pd.read_parquet(args.data_path)
    logger.info("Loaded %d rows from %s", len(df), args.data_path)

    if args.category:
        df = df[df["category"] == args.category]
        logger.info("Filtered to category=%s: %d rows", args.category, len(df))

    start = max(0, int(args.start))
    if args.end is not None:
        end = min(int(args.end), len(df))
    elif args.limit is not None:
        end = min(start + int(args.limit), len(df))
    else:
        end = len(df)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Processing rows [%d, %d) -> %s", start, end, args.output_dir)

    inference_cfg = InferenceConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    success, failure = 0, 0
    for idx in range(start, end):
        try:
            row = df.iloc[idx]
            pipeline.process_single_case(
                row=row,
                runner=runner,
                output_dir=args.output_dir,
                case_idx=idx,
                dataset_type=args.dataset,
                inference_cfg=inference_cfg,
            )
            success += 1
        except Exception as exc:
            failure += 1
            logger.error("Case %d failed: %s", idx, exc)
            traceback.print_exc()

    logger.info("Done. success=%d failure=%d output=%s", success, failure, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
