#!/usr/bin/env python3
"""
Layer 1 Dataset Converter — Raw JSONL → Unsloth/TRL Training Format
Converts generated synthetic data into chat-template format for Qwen2.5-3B fine-tuning.

Usage:
    python convert_layer1.py --input ./layer1_train --output ./layer1_formatted
"""

import json
import argparse
from pathlib import Path
from typing import Iterator


from prompts.layer1_prompts import ROUTER_SYSTEM_PROMPT


def load_raw_samples(input_dir: Path) -> Iterator[dict]:
    """Load all raw JSONL files from directory."""
    for jsonl_file in sorted(input_dir.glob("part_*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def format_for_training(raw_sample: dict) -> dict:
    """Convert raw sample to Unsloth/TRL chat format."""
    prompt_text = raw_sample["prompt"]
    label = raw_sample["label"]
    reasoning = raw_sample.get("reasoning", "")

    # The assistant should output the label + reasoning
    assistant_text = f"Classification: {label}\nReasoning: {reasoning}"

    return {
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": assistant_text},
        ],
        # Metadata for filtering/analysis
        "metadata": {
            "original_label": label,
            "category": raw_sample.get("category", ""),
            "sample_id": raw_sample.get("sample_id", ""),
        }
    }


def split_dataset(samples: list, train_ratio=0.85, val_ratio=0.10) -> tuple:
    """Split into train/val/test with stratification by label."""
    from collections import defaultdict
    import random

    # Group by label
    by_label = defaultdict(list)
    for s in samples:
        by_label[s["metadata"]["original_label"]].append(s)

    train, val, test = [], [], []

    for label, group in by_label.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def write_jsonl(samples: list, filepath: Path) -> None:
    """Write samples to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./layer1_train", help="Raw JSONL input dir")
    parser.add_argument("--output", type=str, default="./layer1_formatted", help="Formatted output dir")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    print(f"Loading raw samples from {input_dir}...")
    raw_samples = list(load_raw_samples(input_dir))
    print(f"Loaded {len(raw_samples)} raw samples")

    print("Converting to training format...")
    formatted = [format_for_training(s) for s in raw_samples]

    print("Splitting dataset...")
    train, val, test = split_dataset(formatted, args.train_ratio, args.val_ratio)

    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    ## randomise filenames so that previous datasets don't get overwritten.
    # Write splits
    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(val, output_dir / "val.jsonl")
    write_jsonl(test, output_dir / "test.jsonl")

    # Write stats
    stats = {
        "total": len(formatted),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "label_distribution": {
            "train": _count_labels(train),
            "val": _count_labels(val),
            "test": _count_labels(test),
        }
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone! Formatted dataset written to {output_dir}")
    print(f"  train.jsonl, val.jsonl, test.jsonl")


def _count_labels(samples: list) -> dict:
    from collections import Counter
    return dict(Counter(s["metadata"]["original_label"] for s in samples))


if __name__ == "__main__":
    main()