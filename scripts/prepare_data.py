import json
import hashlib
import random
import os
import argparse

def normalize_text(text: str) -> str:
    """Basic normalization: strip whitespace."""
    return text.strip()

def compute_hash(text: str) -> str:
    """Compute exact SHA-256 dedup hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def is_quality(text: str) -> bool:
    """
    Quality filter:
    - Reject if under 50 chars.
    - Reject if mostly non-alphanumeric boilerplate.
    """
    if len(text) < 50:
        return False

    alnum_count = sum(c.isalnum() for c in text)
    if alnum_count / max(1, len(text)) < 0.5:
        return False

    return True

def process_file(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    seen_hashes = set()
    total = 0
    kept = 0

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(train_path, 'w', encoding='utf-8') as f_train, \
         open(val_path, 'w', encoding='utf-8') as f_val:

        for line in fin:
            if not line.strip():
                continue

            total += 1
            record = json.loads(line)

            # Normalization
            text = normalize_text(record.get("text", ""))
            if not text:
                continue

            # Quality filter
            if not is_quality(text):
                continue

            # Dedup
            text_hash = compute_hash(text)
            if text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)

            # Update record
            record["text"] = text
            record["dedup_hash"] = text_hash

            # Shard output
            out_file = f_val if random.random() < 0.1 else f_train
            out_file.write(json.dumps(record) + "\n")
            kept += 1

    print(f"Processed {total} records. Kept {kept} after quality/dedup filters.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and clean raw text data.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for train.jsonl and val.jsonl")
    args = parser.parse_args()

    process_file(args.input, args.output_dir)
