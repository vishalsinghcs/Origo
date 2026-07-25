#!/usr/bin/env python3
"""
Layer 1 Synthetic Data Generator — Async Parallel Generation
Generates training data for the 3B Semantic Router (SAFE/SUSPICIOUS/UNSAFE)
Uses multiple Groq + SambaNova API keys for true parallelism via asyncio.

Usage:
    python generate_layer1.py --target 5000 --output ./layer1_train --chunk-size 1000
"""

import asyncio
import aiohttp
import json
import hashlib
import os
import sys
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Iterator
from dataclasses import dataclass, asdict


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Add your API keys here
# ═══════════════════════════════════════════════════════════════════════════════

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

GROQ_KEYS = [os.getenv(f"GROQ_API_KEY_{i}") for i in range(1, 23+1) if os.getenv(f"GROQ_API_KEY_{i}")]
if not GROQ_KEYS:
    print("Error: No GROQ_API_KEY_17...21 found in .env.")
    sys.exit(1)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_PRIMARY = "llama-3.1-8b-instant"
MODEL_SECONDARY = "llama-3.1-8b-instant"

# ═══════════════════════════════════════════════════════════════════════════════

# Target distribution (uneven, security-focused)
CLASS_DISTRIBUTION = {
    "SAFE": 0.40,
    "SUSPICIOUS": 0.30,
    "UNSAFE": 0.30,
}

# Sub-categories for diversity within each class
SAFE_CATEGORIES = [
    "general_coding", "business_email", "data_analysis", "documentation",
    "math_problem", "creative_writing", "translation", "summarization",
    "scheduling", "travel_planning", "recipe_request", "fitness_advice",
]

SUSPICIOUS_CATEGORIES = [
    "credential_in_context", "pii_in_question", "api_key_mentioned",
    "ssn_in_text", "password_hint", "internal_doc_reference",
    "partial_secret", "email_with_domain", "phone_number_in_query",
]

UNSAFE_CATEGORIES = [
    "jailbreak_dan", "prompt_injection_system", "ignore_previous",
    "roleplay_exploit", "encoding_bypass", "social_engineering",
    "exploit_request", "malware_generation", "phishing_template",
    "harmful_content", "discrimination", "illegal_activity",
]


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS — One per class, with category injection
# ═══════════════════════════════════════════════════════════════════════════════


from prompts.layer1_prompts import SAFE_SYSTEM_PROMPT, SUSPICIOUS_SYSTEM_PROMPT, UNSAFE_SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Layer1Sample:
    sample_id: str
    prompt: str
    label: str
    reasoning: str
    category: str
    sensitive_elements: list = None
    attack_vectors: list = None
    severity: str = None
    metadata: dict = None

    def to_dict(self):
        d = asdict(self)
        # Remove None fields for cleaner JSON
        return {k: v for k, v in d.items() if v is not None}


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AsyncDataGenerator:
    def __init__(
        self,
        groq_keys: list[str],
        groq_url: str = GROQ_URL,
        max_retries: int = 15,
        timeout: int = 120,
    ):
        self.groq_keys = groq_keys
        self.groq_url = groq_url
        self.max_retries = max_retries
        self.timeout = timeout

        # Round-robin key indices
        self.groq_idx = 0
        self.lock = asyncio.Lock()

        # Semaphore to limit concurrent requests per provider
        self.groq_semaphore = asyncio.Semaphore(len(groq_keys))

    async def _get_next_key(self) -> tuple[str, str, int]:
        async with self.lock:
            key_idx = self.groq_idx % len(self.groq_keys)
            key = self.groq_keys[key_idx]
            self.groq_idx += 1
            
            if key_idx < 8:
                model = MODEL_PRIMARY
            else:
                model = MODEL_SECONDARY
                
            return key, model, key_idx

    async def _call_api(
        self,
        session: aiohttp.ClientSession,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.9,
    ) -> tuple[dict, str]:
        """Make a single API call with retry logic."""
        key, model, key_idx = await self._get_next_key()

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_completion_tokens": 1024,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(self.max_retries):
            try:
                async with session.post(self.groq_url, headers=headers, json=payload, timeout=self.timeout) as response:
                    err_msg = ""
                    if response.status != 200:
                        try:
                            err_data = await response.json()
                            err_msg = err_data.get("error", {}).get("message", "No message provided")
                        except Exception:
                            err_msg = await response.text()

                    if response.status == 429:
                        # Extract detailed error message from Groq

                        # Rate limited — Check for Retry-After header or fallback to exponential backoff
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            wait = float(retry_after)
                        else:
                            wait = 2 ** attempt + random.uniform(1, 3)
                            
                        print(f"  [!] 429 Rate Limit on key {key_idx + 1}. Wait {wait:.2f}s (Attempt {attempt + 1}/{self.max_retries}) | Groq Error: {err_msg}")
                        await asyncio.sleep(wait)
                        continue

                    if response.status == 401:
                        raise RuntimeError(f"API Key {key_idx + 1} is INVALID or UNAUTHORIZED! (401) | Groq Error: {err_msg}")

                    response.raise_for_status()
                    data = await response.json()
                    raw_content = data["choices"][0]["message"]["content"]
                    print(f"  ✓ Got a sample using api key {key_idx + 1} in attempt {attempt + 1}")
                    return json.loads(raw_content), model

            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed after {self.max_retries} attempts due to error: {e}")
                wait = 2 ** attempt + random.uniform(1, 3)
                print(f"  [!] Error on key {key_idx + 1}: {e}. Retrying in {wait:.2f}s... | Groq Error: {err_msg}")
                await asyncio.sleep(wait)

        raise RuntimeError(f"Failed after {self.max_retries} attempts (Continuous 429 Rate Limits)")

    async def generate_single(
        self,
        session: aiohttp.ClientSession,
        sample_idx: int,
        label: str,
        category: str,
    ) -> Layer1Sample:
        """Generate one sample with true parallelism."""

        semaphore = self.groq_semaphore

        # Build system prompt
        if label == "SAFE":
            system_prompt = SAFE_SYSTEM_PROMPT.format(category=category)
        elif label == "SUSPICIOUS":
            system_prompt = SUSPICIOUS_SYSTEM_PROMPT.format(category=category)
        else:
            system_prompt = UNSAFE_SYSTEM_PROMPT.format(category=category)

        user_prompt = f"Generate sample #{sample_idx} for category: {category}. Make it unique and realistic."

        async with semaphore:
            raw, model_used = await self._call_api(session, system_prompt, user_prompt)

        # Build sample
        sample = Layer1Sample(
            sample_id=f"l1_{sample_idx:06d}",
            prompt=raw["prompt"],
            label=raw["label"],
            reasoning=raw["reasoning"],
            category=raw.get("category", category),
            sensitive_elements=raw.get("sensitive_elements"),
            attack_vectors=raw.get("attack_vectors"),
            severity=raw.get("severity"),
            metadata={
                "provider": "groq",
                "model": model_used,
                "generated_at": datetime.utcnow().isoformat(),
                "system_prompt_hash": hashlib.sha256(system_prompt.encode()).hexdigest()[:16],
            },
        )
        return sample

    async def generate_batch(
        self,
        target_total: int,
        output_dir: Path,
        chunk_size: int = 1000,
    ) -> None:
        """Generate the full dataset with true parallelism."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate per-class targets
        class_targets = {
            label: int(target_total * ratio)
            for label, ratio in CLASS_DISTRIBUTION.items()
        }
        # Adjust for rounding
        class_targets["SAFE"] += target_total - sum(class_targets.values())

        # Build work queue: list of (global_idx, label, category)
        work_queue = []
        global_idx = 0
        for label, count in class_targets.items():
            categories = {
                "SAFE": SAFE_CATEGORIES,
                "SUSPICIOUS": SUSPICIOUS_CATEGORIES,
                "UNSAFE": UNSAFE_CATEGORIES,
            }[label]
            for i in range(count):
                category = categories[i % len(categories)]
                work_queue.append((global_idx, label, category))
                global_idx += 1

        # Shuffle for random ordering
        random.shuffle(work_queue)

        print(f"[Layer 1] Generating {target_total} samples across {len(work_queue)} tasks")
        print(f"  Distribution: SAFE={class_targets['SAFE']}, SUSPICIOUS={class_targets['SUSPICIOUS']}, UNSAFE={class_targets['UNSAFE']}")
        print(f"  Chunk size: {chunk_size} | Output: {output_dir}")

        # Track progress
        generated = 0
        failed = 0
        current_chunk = []
        chunk_num = 1

        async with aiohttp.ClientSession() as session:
            # Fire all tasks concurrently — true parallelism
            tasks = [
                self.generate_single(session, idx, label, category)
                for idx, label, category in work_queue
            ]

            # Process results as they complete
            for coro in asyncio.as_completed(tasks):
                try:
                    sample = await coro
                    current_chunk.append(sample.to_dict())
                    generated += 1

                    # Flush chunk when full
                    if len(current_chunk) >= chunk_size:
                        chunk_num = self._write_chunk(output_dir, chunk_num, current_chunk)
                        print(f"  ✓ Chunk {chunk_num:03d} written ({len(current_chunk)} samples) | Total: {generated}/{target_total}")
                        current_chunk = []
                        chunk_num += 1

                except Exception as e:
                    failed += 1
                    if failed <= 5 or failed % 100 == 0:
                        print(f"  ✗ Failed sample: {e}")

        # Flush remaining
        if current_chunk:
            chunk_num = self._write_chunk(output_dir, chunk_num, current_chunk)
            print(f"  ✓ Chunk {chunk_num:03d} written ({len(current_chunk)} samples)")

        print(f"\n[Done] Generated: {generated} | Failed: {failed} | Chunks: {chunk_num}")
        self._write_manifest(output_dir, target_total, generated, failed, chunk_num)

    def _write_chunk(self, output_dir: Path, chunk_num: int, samples: list) -> int:
        """Write a chunk to disk immediately, avoiding overwrites."""
        filepath = output_dir / f"part_{chunk_num:04d}.jsonl"
        while filepath.exists():
            chunk_num += 1
            filepath = output_dir / f"part_{chunk_num:04d}.jsonl"
            
        with open(filepath, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        return chunk_num

    def _write_manifest(
        self,
        output_dir: Path,
        target: int,
        generated: int,
        failed: int,
        chunks: int,
    ) -> None:
        """Write generation manifest."""
        manifest = {
            "target_total": target,
            "generated": generated,
            "failed": failed,
            "chunks": chunks,
            "class_distribution": CLASS_DISTRIBUTION,
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Layer 1 Synthetic Data Generator")
    parser.add_argument("--target", type=int, default=5000, help="Total samples to generate")
    parser.add_argument("--output", type=str, default="./layer1_train", help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Samples per JSONL file")
    args = parser.parse_args()

    generator = AsyncDataGenerator(
        groq_keys=GROQ_KEYS,
    )

    asyncio.run(generator.generate_batch(
        target_total=args.target,
        output_dir=Path(args.output),
        chunk_size=args.chunk_size,
    ))


if __name__ == "__main__":
    main()