import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set

def get_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generate char n-grams for a given text."""
    # Convert to lowercase and remove spaces for robust matching
    text = text.lower().replace(" ", "")
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text)-n+1)}

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

class QualityFilter:
    def __init__(self, jaccard_threshold: float = 0.8):
        self.jaccard_threshold = jaccard_threshold
        self.seen_exact_hashes: Set[str] = set()
        # To avoid O(N^2) complexity with fuzzy matching, we can store 
        # the n-grams of accepted prompts and compare against them.
        self.accepted_ngrams: List[Set[str]] = []
        
        self.stats = {
            "total_processed": 0,
            "rejected_format": 0,
            "rejected_exact_duplicate": 0,
            "rejected_fuzzy_duplicate": 0,
            "accepted": 0
        }

    def is_valid_format(self, record: dict) -> bool:
        """Check if the record has the required fields."""
        # Layer 1 required fields
        if "prompt" in record and "label" in record:
            return True
        # Layer 2 required fields
        if "original_prompt" in record and "action" in record:
            return True
        return False
        
    def get_prompt_text(self, record: dict) -> str:
        """Extract the main text to check for duplication."""
        if "prompt" in record:
            return record["prompt"]
        if "original_prompt" in record:
            return record["original_prompt"]
        return ""

    def process_record(self, record: dict) -> bool:
        """Returns True if accepted, False if rejected."""
        self.stats["total_processed"] += 1
        
        # 1. Format Check
        if not self.is_valid_format(record):
            self.stats["rejected_format"] += 1
            return False
            
        prompt_text = self.get_prompt_text(record)
        if not prompt_text:
            self.stats["rejected_format"] += 1
            return False

        # 2. Exact Deduplication
        # Use a hash to save memory
        exact_hash = hash(prompt_text.strip().lower())
        if exact_hash in self.seen_exact_hashes:
            self.stats["rejected_exact_duplicate"] += 1
            return False
            
        # 3. Fuzzy Deduplication (Diversity)
        record_ngrams = get_ngrams(prompt_text)
        
        for accepted_ngram_set in self.accepted_ngrams:
            sim = jaccard_similarity(record_ngrams, accepted_ngram_set)
            if sim >= self.jaccard_threshold:
                self.stats["rejected_fuzzy_duplicate"] += 1
                return False

        # If it passes all checks, accept it
        self.seen_exact_hashes.add(exact_hash)
        self.accepted_ngrams.append(record_ngrams)
        self.stats["accepted"] += 1
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Filter generated data for quality and diversity.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file or directory containing JSONL files")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file for accepted samples")
    parser.add_argument("--threshold", type=float, default=0.8, help="Jaccard similarity threshold for fuzzy matching (default 0.8)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist.")
        sys.exit(1)
        
    # Gather input files
    input_files = []
    if input_path.is_file():
        input_files.append(input_path)
    elif input_path.is_dir():
        input_files.extend(input_path.glob("*.jsonl"))
        
    if not input_files:
        print(f"No JSONL files found in {input_path}")
        sys.exit(1)
        
    # If input is a directory, output must be a directory
    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    qf = QualityFilter(jaccard_threshold=args.threshold)
    
    print(f"Starting filtering process on {len(input_files)} file(s)...")
    print(f"Similarity threshold: {args.threshold}")
    
    for file_path in input_files:
        print(f"Processing {file_path.name}...")
        
        # Determine specific output file path
        if input_path.is_dir():
            current_out_path = output_path / file_path.name
        else:
            current_out_path = output_path
            
        with open(current_out_path, "w", encoding="utf-8") as out_f:
            with open(file_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        if qf.process_record(record):
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    except json.JSONDecodeError:
                        qf.stats["total_processed"] += 1
                        qf.stats["rejected_format"] += 1
                        
    print("\n--- Filtering Complete ---")
    for key, value in qf.stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
