#!/usr/bin/env python3
"""
Layer 2 Dataset Converter — Raw JSONL → Unsloth/TRL Training Format
Converts generated synthetic data into chat-template format for Qwen3-8B fine-tuning.

Usage:
    python convert_layer2.py --input ./layer2_train --output ./layer2_formatted
"""

import json
import argparse
from pathlib import Path
from typing import Iterator


from prompts.layer2_prompts import SPECIALIST_SYSTEM_PROMPT


def load_raw_samples(input_dir: Path) -> Iterator[dict]:
    """Load all raw JSONL files from directory."""
    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def format_for_training(raw_sample: dict) -> dict:
    """Convert raw sample to Unsloth/TRL chat format."""
    original_prompt = raw_sample["original_prompt"]
    rewritten = raw_sample.get("rewritten_prompt")
    action = raw_sample["action"]
    reasoning = raw_sample["reasoning"]
    confidence = raw_sample.get("confidence", 0.90)
    threat_type = raw_sample.get("threat_type", "")
    subcategory = raw_sample.get("subcategory", "")
    severity = raw_sample.get("severity", "HIGH")
    entities = raw_sample.get("entities_detected", [])

    # Build structured assistant response as JSON
    assistant_dict = {
        "action": action,
        "confidence": confidence,
        "threat_type": threat_type,
        "subcategory": subcategory,
        "severity": severity,
        "entities_detected": entities,
        "reasoning": reasoning,
        "rewritten_prompt": rewritten
    }
    
    assistant_text = json.dumps(assistant_dict, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SPECIALIST_SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze and sanitize this prompt:\n\n{original_prompt}"},
            {"role": "assistant", "content": assistant_text},
        ],
        "metadata": {
            "threat_type": threat_type,
            "subcategory": subcategory,
            "action": action,
            "severity": severity,
            "sample_id": raw_sample.get("sample_id", ""),
        }
    }


def split_dataset(samples: list, train_ratio=0.85, val_ratio=0.10) -> tuple:
    """Split into train/val/test with stratification by threat type."""
    from collections import defaultdict
    import random

    by_threat = defaultdict(list)
    for s in samples:
        by_threat[s["metadata"]["threat_type"]].append(s)

    train, val, test = [], [], []

    for threat, group in by_threat.items():
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
    parser.add_argument("--input", type=str, default="../../datasets/preprocessed/layer2", help="Raw JSONL input dir")
    parser.add_argument("--output", type=str, default="../../datasets/formatted/layer2", help="Formatted output dir")
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

    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(val, output_dir / "val.jsonl")
    write_jsonl(test, output_dir / "test.jsonl")

    stats = {
        "total": len(formatted),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "threat_distribution": {
            "train": _count_threats(train),
            "val": _count_threats(val),
            "test": _count_threats(test),
        },
        "action_distribution": {
            "train": _count_actions(train),
            "val": _count_actions(val),
            "test": _count_actions(test),
        }
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone! Formatted dataset written to {output_dir}")
    print(f"  train.jsonl, val.jsonl, test.jsonl")


def _count_threats(samples: list) -> dict:
    from collections import Counter
    return dict(Counter(s["metadata"]["threat_type"] for s in samples))


def _count_actions(samples: list) -> dict:
    from collections import Counter
    return dict(Counter(s["metadata"]["action"] for s in samples))


if __name__ == "__main__":
    main()