import json
import argparse
import os
import numpy as np
from transformers import AutoTokenizer

def tokenize_and_pack(input_path: str, output_prefix: str, tokenizer_name: str, max_seq_len: int):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    bin_file = f"{output_prefix}.bin"
    idx_file = f"{output_prefix}.idx"

    buffer = []
    total_chunks = 0

    # We will write uint16 tokens to a raw binary file to act as the memory map target
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(bin_file, 'wb') as fout:

        for line in fin:
            if not line.strip():
                continue

            record = json.loads(line)
            text = record.get("text", "")

            # Tokenize text
            tokens = tokenizer.encode(text, add_special_tokens=True)
            buffer.extend(tokens)

            # Pack into fixed-length chunks
            while len(buffer) >= max_seq_len:
                chunk = buffer[:max_seq_len]
                buffer = buffer[max_seq_len:]

                # Convert to numpy uint16 and write directly to the bin file
                arr = np.array(chunk, dtype=np.uint16)
                fout.write(arr.tobytes())
                total_chunks += 1

        # Write remaining if buffer is not empty and we want to pad it out or keep it (for testing/demo purposes we pad with 0)
        if len(buffer) > 0:
            chunk = buffer + [0] * (max_seq_len - len(buffer))
            arr = np.array(chunk, dtype=np.uint16)
            fout.write(arr.tobytes())
            total_chunks += 1

    # Write the index file summarizing the shard
    index_data = {
        "num_chunks": total_chunks,
        "max_seq_len": max_seq_len,
        "dtype": "uint16"
    }

    with open(idx_file, 'w', encoding='utf-8') as fidx:
        json.dump(index_data, fidx, indent=2)

    print(f"Tokenization complete. Wrote {total_chunks} chunks of size {max_seq_len} to {bin_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize and pack text data into binary shards.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file.")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for output .bin and .idx files")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="HuggingFace tokenizer name")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Sequence length for packing")
    args = parser.parse_args()

    tokenize_and_pack(args.input, args.output_prefix, args.tokenizer, args.max_seq_len)
