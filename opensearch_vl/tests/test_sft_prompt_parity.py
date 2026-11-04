#!/usr/bin/env python3
"""Inference prompt == SFT training prompt, token for token (needs the instance).

For real training trajectories, rebuilds the pipeline's message list turn by
turn and checks that ``Qwen35SFTRunner``'s prompt is an exact prefix of the ids
LLaMA-Factory's *training* preprocessing produces, and that the assistant target
follows it. Tokenizer, processor and template come from LLaMA-Factory's own
training loaders with the SFT config's values.

    python tests/test_sft_prompt_parity.py --model /root/models/Qwen3.5-4B \
        --data /root/datasets/sft_subset_2048/sft_subset.json --samples 40
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--samples", type=int, default=40)
    args = ap.parse_args()

    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer

    from opensearch_infer import config
    from opensearch_infer.runners import Qwen35SFTRunner

    model_args = ModelArguments(model_name_or_path=args.model, image_max_pixels=262144)
    data_args = DataArguments(template="qwen3_5", cutoff_len=32000)
    loaded = load_tokenizer(model_args)
    tokenizer, processor = loaded["tokenizer"], loaded["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    train_proc = SupervisedDatasetProcessor(template, tokenizer, processor, data_args)

    runner = Qwen35SFTRunner(config.MODEL_REGISTRY["qwen3.5-4b"], args.model)
    runner.load_prompt_stack()

    data = json.load(open(args.data, encoding="utf-8"))
    # Most image-producing trajectories first, then the rest.
    data.sort(key=lambda s: -sum(m["value"].count("<image>") for m in s["conversations"]))
    picked = data[: args.samples // 2] + data[-(args.samples - args.samples // 2) :]

    role = {"human": "user", "gpt": "assistant", "observation": "observation"}
    checked_turns = grids = 0
    for sample in picked:
        conv = sample["conversations"]
        messages = [{"role": role[m["from"]], "content": m["value"]} for m in conv]
        train_ids, _ = train_proc._encode_data_example(
            prompt=messages[:-1],
            response=messages[-1:],
            system=sample["system"],
            tools=sample["tools"],
            images=sample["images"],
            videos=[],
            audios=[],
        )

        contents, used = [], 0
        for m in conv:
            text = m["value"]
            if m["from"] == "gpt":
                msgs, images = runner.contents_to_template(contents)
                ids = runner.build_prompt_ids(msgs, images)
                assert train_ids[: len(ids)] == ids, (
                    f"prompt mismatch at turn {checked_turns}: first diff at "
                    f"{next(i for i, (a, b) in enumerate(zip(train_ids, ids)) if a != b)}"
                )
                # The target this prompt was trained to produce comes next.
                # (The template prepends an empty think block to targets without one.)
                shown = text if "<think>" in text else "<think>\n\n</think>\n\n" + text
                target = tokenizer(shown, add_special_tokens=False)["input_ids"][:8]
                assert train_ids[len(ids) : len(ids) + len(target)] == target, "target misaligned"
                checked_turns += 1
                contents.append({"role": "model", "parts": [{"text": text}]})
                continue
            n = text.count("<image>")
            assert text.startswith("<image>" * n), "image not at the start of the turn"
            parts = []
            for path in sample["images"][used : used + n]:
                with open(path, "rb") as fh:
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64.b64encode(fh.read()).decode(),
                            }
                        }
                    )
            used += n
            parts.append({"text": text[len("<image>") * n :]})
            contents.append({"role": "user", "parts": parts})
        assert used == len(sample["images"])

        # Same pixel input as training (image_max_pixels etc.).
        _, images = runner.contents_to_template(contents)
        if images:
            ours = runner._template.mm_plugin._get_mm_inputs(images, [], [], runner._processor)
            theirs = template.mm_plugin._get_mm_inputs(sample["images"], [], [], processor)
            assert (ours["image_grid_thw"] == theirs["image_grid_thw"]).all(), "grid differs"
            assert (ours["pixel_values"] - theirs["pixel_values"]).abs().max() < 1e-4, "pixels differ"
            grids += len(images)

    print(f"OK: {len(picked)} trajectories, {checked_turns} assistant turns, {grids} images match training")
    return 0


if __name__ == "__main__":
    sys.exit(main())
