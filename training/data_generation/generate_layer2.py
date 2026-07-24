#!/usr/bin/env python3
"""
Layer 2 Synthetic Data Generator — Async Parallel Generation
Generates training data for the 8B Security-Aware Prompt Rewriter.
Uses multiple Groq + SambaNova API keys for true parallelism via asyncio.

Usage:
    python generate_layer2.py --target 8000 --output ./layer2_train --chunk-size 1000
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
    print("Error: No GROQ_API_KEY_1...16 found in .env.")
    sys.exit(1)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL_PRIMARY = "groq/compound"
MODEL_SECONDARY = "groq/compound"

## is this threat distribution correct and contains all types?

# Target distribution by threat type (uneven, severity-weighted)
THREAT_DISTRIBUTION = {
    "pii_phi_exposure": 0.15,
    "credential_leakage": 0.15,
    "prompt_injection": 0.15,
    "jailbreak": 0.15,
    "source_code_ip_leak": 0.10,
    "toxicity_policy_violation": 0.05,
    "safe_baseline": 0.10,
    "false_positive_safe": 0.15,
}

# Sub-categories for diversity
THREAT_SUBCATEGORIES = {
    "pii_phi_exposure": [
        "ssn_in_prompt", "credit_card_in_text", "medical_record_phi",
        "passport_number", "bank_account_details", "home_address_full",
        "date_of_birth_with_name", "employee_id_with_name", "patient_id_reference",
    ],
    "credential_leakage": [
        "aws_access_key", "github_token", "database_password_inline",
        "jwt_token_pasted", "api_key_in_url", "ssh_private_key",
        "oauth_secret", "slack_webhook_url", "docker_registry_password",
    ],
    "prompt_injection": [
        "system_override_prefix", "ignore_all_instructions", "new_role_assignment",
        "delimiter_escape", "encoding_trick_base64", "markdown_injection",
        "xml_tag_injection", "json_injection", "translation_exploit",
    ],
    "jailbreak": [
        "dan_mode", "developer_mode", "jailbreak_character",
        "hypothetical_framing", "research_pretext", "grandma_exploit",
        "token_smuggling", "refusal_suppression", "authority_impersonation",
    ],
    "source_code_ip_leak": [
        "proprietary_algorithm", "internal_api_endpoint", "unreleased_feature_code",
        "patent_pending_logic", "trade_secret_formula", "client_source_code",
        "internal_database_schema", "proprietary_ml_model_weights",
    ],
    "toxicity_policy_violation": [
        "hate_speech", "harassment", "discrimination", "violence_incitement",
        "self_harm_content", "sexual_content", "misinformation",
    ],
    "safe_baseline": [
        "general_coding", "business_email", "data_analysis", "documentation",
        "math_problem", "creative_writing", "translation", "summarization",
        "scheduling", "travel_planning", "recipe_request", "fitness_advice",
        "staging_environment_secret", "example_config_with_placeholders",
        "password_reset_flow", "account_recovery_legitimate",
        "security_audit_request", "penetration_testing_authorized",
        "vulnerability_disclosure_legitimate", "bug_bounty_query", 
        "secrets_manager_integration", "encryption_best_practices",
        "pii_detection_implementation", "access_control_design",
        "audit_logging_setup", "data_masking_request",
        "security_architecture_review", "compliance_gap_analysis"
    ],
    "false_positive_safe": [
        "educational_pattern_reference", "documentation_example", 
        "test_data_explicitly_labeled", "public_information_query",
        "synthetic_data_request", "regex_tutorial", "coding_interview_question",
        "security_best_practice_question", "compliance_policy_query"
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — Single prompt that handles all threat types via category injection
# ═══════════════════════════════════════════════════════════════════════════════

from prompts.layer2_prompts import REWRITER_SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Entity:
    type: str
    value: str
    position: list  # [start, end]
    redaction: str

    def to_dict(self):
        return {
            "type": self.type,
            "value": self.value,
            "position": self.position,
            "redaction": self.redaction,
        }


@dataclass
class Layer2Sample:
    sample_id: str
    original_prompt: str
    rewritten_prompt: str | None
    reasoning: str
    action: str
    confidence: float
    threat_type: str
    subcategory: str
    severity: str
    entities_detected: list
    policy_violated: str
    metadata: dict

    def to_dict(self):
        return {
            "sample_id": self.sample_id,
            "original_prompt": self.original_prompt,
            "rewritten_prompt": self.rewritten_prompt,
            "reasoning": self.reasoning,
            "action": self.action,
            "confidence": self.confidence,
            "threat_type": self.threat_type,
            "subcategory": self.subcategory,
            "severity": self.severity,
            "entities_detected": [e.to_dict() for e in self.entities_detected],
            "policy_violated": self.policy_violated,
            "metadata": self.metadata,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AsyncLayer2Generator:
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

        self.groq_idx = 0
        self.lock = asyncio.Lock()

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
        temperature: float = 0.85,
    ) -> tuple[dict, str]:
        """Make a single API call with retry logic."""
        key, model, key_idx = await self._get_next_key()

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_completion_tokens": 2048,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    self.groq_url, headers=headers, json=payload, timeout=self.timeout
                ) as response:
                    if response.status == 429:
                        try:
                            err_data = await response.json()
                            err_msg = err_data.get("error", {}).get("message", "No message provided")
                        except Exception:
                            err_msg = await response.text()

                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            wait = float(retry_after)
                        else:
                            wait = 2 ** attempt + random.uniform(1, 3)
                            
                        print(f"  [!] 429 Rate Limit on key {key_idx + 1}. Wait {wait:.2f}s (Attempt {attempt + 1}/{self.max_retries}) | Groq Error: {err_msg}")
                        await asyncio.sleep(wait)
                        continue

                    if response.status == 401:
                        raise RuntimeError(f"API Key {key_idx + 1} is INVALID or UNAUTHORIZED! (401)")

                    response.raise_for_status()
                    data = await response.json()
                    raw_content = data["choices"][0]["message"]["content"]
                    print(f"  ✓ Got a sample using api key {key_idx + 1} in attempt {attempt + 1}")
                    return json.loads(raw_content), model

            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed after {self.max_retries} attempts due to error: {e}")
                wait = 2 ** attempt + random.uniform(1, 3)
                print(f"  [!] Error on key {key_idx + 1}: {e}. Retrying in {wait:.2f}s...")
                await asyncio.sleep(wait)

        raise RuntimeError(f"Failed after {self.max_retries} attempts (Continuous 429 Rate Limits)")

    async def generate_single(
        self,
        session: aiohttp.ClientSession,
        sample_idx: int,
        threat_type: str,
        subcategory: str,
    ) -> Layer2Sample:
        """Generate one Layer 2 sample with true parallelism."""

        semaphore = self.groq_semaphore

        system_prompt = REWRITER_SYSTEM_PROMPT.format(
            threat_type=threat_type,
            subcategory=subcategory,
        )

        if threat_type == "safe_baseline":
            user_prompt = (
                f"Generate sample #{sample_idx} for a safe baseline prompt "
                f"(category: {subcategory}). Make it a completely realistic, benign "
                f"enterprise query with NO sensitive elements or attacks. Include 2-3 sentences of context."
            )
        elif threat_type == "false_positive_safe":
            user_prompt = (
                f"Generate sample #{sample_idx} for a false positive edge case "
                f"(category: {subcategory}). This should LOOK like it contains a threat "
                f"(e.g., regex patterns, fake PII used for testing, educational examples) "
                f"but is actually completely SAFE and benign. Include 2-3 sentences of context."
            )
        else:
            user_prompt = (
                f"Generate sample #{sample_idx} for threat type '{threat_type}' "
                f"(subcategory: {subcategory}). Make it realistic, detailed, and unique. "
                f"Include at least 2-3 sentences of context before the sensitive element."
            )

        async with semaphore:
            raw, model_used = await self._call_api(session, system_prompt, user_prompt)

        # Parse entities
        entities = []
        for ent in raw.get("entities_detected", []):
            entities.append(Entity(
                type=ent.get("type", "UNKNOWN"),
                value=ent.get("value", ""),
                position=ent.get("position", [0, 0]),
                redaction=ent.get("redaction", "[REDACTED]"),
            ))

        sample = Layer2Sample(
            sample_id=f"l2_{sample_idx:06d}",
            original_prompt=raw["original_prompt"],
            rewritten_prompt=raw.get("rewritten_prompt"),
            reasoning=raw["reasoning"],
            action=raw["action"],
            confidence=raw.get("confidence", 0.90),
            threat_type=raw.get("threat_type", threat_type),
            subcategory=raw.get("subcategory", subcategory),
            severity=raw.get("severity", "HIGH"),
            entities_detected=entities,
            policy_violated=raw.get("policy_violated", "CONTENT_POLICY"),
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
        """Generate the full Layer 2 dataset with true parallelism."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate per-threat targets
        threat_targets = {
            threat: int(target_total * ratio)
            for threat, ratio in THREAT_DISTRIBUTION.items()
        }
        # Adjust for rounding
        threat_targets["pii_phi_exposure"] += target_total - sum(threat_targets.values())

        # Build work queue
        work_queue = []
        global_idx = 0
        for threat, count in threat_targets.items():
            subcategories = THREAT_SUBCATEGORIES[threat]
            for i in range(count):
                subcategory = subcategories[i % len(subcategories)]
                work_queue.append((global_idx, threat, subcategory))
                global_idx += 1

        random.shuffle(work_queue)

        print(f"[Layer 2] Generating {target_total} samples across {len(work_queue)} tasks")
        print(f"  Distribution: " + ", ".join(f"{k}={v}" for k, v in threat_targets.items()))
        print(f"  Chunk size: {chunk_size} | Output: {output_dir}")

        generated = 0
        failed = 0
        current_chunk = []
        chunk_num = 1

        async with aiohttp.ClientSession() as session:
            # Fire ALL tasks concurrently — true parallelism across all 20 API keys
            tasks = [
                self.generate_single(session, idx, threat, subcategory)
                for idx, threat, subcategory in work_queue
            ]

            for coro in asyncio.as_completed(tasks):
                try:
                    sample = await coro
                    current_chunk.append(sample.to_dict())
                    generated += 1

                    if len(current_chunk) >= chunk_size:
                        chunk_num = self._write_chunk(output_dir, chunk_num, current_chunk)
                        print(f"  ✓ Chunk {chunk_num:03d} written ({len(current_chunk)} samples) | Total: {generated}/{target_total}")
                        current_chunk = []
                        chunk_num += 1

                except Exception as e:
                    failed += 1
                    if failed <= 5 or failed % 100 == 0:
                        print(f"  ✗ Failed sample: {e}")

        if current_chunk:
            chunk_num = self._write_chunk(output_dir, chunk_num, current_chunk)
            print(f"  ✓ Chunk {chunk_num:03d} written ({len(current_chunk)} samples)")

        print(f"\n[Done] Generated: {generated} | Failed: {failed} | Chunks: {chunk_num}")
        self._write_manifest(output_dir, target_total, generated, failed, chunk_num)

    def _write_chunk(self, output_dir: Path, chunk_num: int, samples: list) -> int:
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
        manifest = {
            "target_total": target,
            "generated": generated,
            "failed": failed,
            "chunks": chunks,
            "threat_distribution": THREAT_DISTRIBUTION,
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Layer 2 Synthetic Data Generator")
    parser.add_argument("--target", type=int, default=8000, help="Total samples to generate")
    parser.add_argument("--output", type=str, default="./layer2_train", help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Samples per JSONL file")
    args = parser.parse_args()

    generator = AsyncLayer2Generator(
        groq_keys=GROQ_KEYS,
    )

    asyncio.run(generator.generate_batch(
        target_total=args.target,
        output_dir=Path(args.output),
        chunk_size=args.chunk_size,
    ))


if __name__ == "__main__":
    main()