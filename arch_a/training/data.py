import json
import os
import random
import numpy as np
import torch
from typing import List, Dict, Any, Iterator

class StreamingShardLoader:
    def __init__(self, bin_files: List[str], batch_size: int, shuffle: bool = True, seed: int = 42):
        self.bin_files = bin_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = random.Random(seed)

        # Cursor tracking for resumability
        self.current_file_idx = 0
        self.current_chunk_idx = 0
        self.epoch = 0

        self._init_epoch()

    def _init_epoch(self):
        """Prepare the file list for the current epoch."""
        if self.shuffle:
            # We seed based on base seed + epoch to ensure different shuffles each epoch
            epoch_rng = random.Random(self.seed + self.epoch)
            epoch_rng.shuffle(self.bin_files)

        self.current_file_idx = 0
        self.current_chunk_idx = 0

    def _load_shard(self, file_path: str):
        """Load a binary shard and its index via numpy memmap."""
        idx_path = file_path.replace(".bin", ".idx")
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"Missing index file for shard: {file_path}")

        with open(idx_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        max_seq_len = meta["max_seq_len"]
        num_chunks = meta["num_chunks"]
        dtype_str = meta["dtype"]

        dtype_map = {"uint16": np.uint16, "int32": np.int32}
        dtype = dtype_map.get(dtype_str, np.uint16)

        mmap_arr = np.memmap(file_path, dtype=dtype, mode='r', shape=(num_chunks, max_seq_len))
        return mmap_arr, num_chunks

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:  # Infinite epoch loop
            while self.current_file_idx < len(self.bin_files):
                file_path = self.bin_files[self.current_file_idx]
                mmap_arr, num_chunks = self._load_shard(file_path)

                # If we resume mid-file, start from the recorded chunk index
                start_chunk = self.current_chunk_idx

                # Create a chunk ordering (shuffled or sequential)
                chunk_indices = list(range(num_chunks))
                if self.shuffle:
                    # We want deterministic chunk shuffling per file per epoch
                    file_rng = random.Random(self.seed + self.epoch + hash(file_path))
                    file_rng.shuffle(chunk_indices)

                batch = []
                for idx in range(start_chunk, num_chunks):
                    self.current_chunk_idx = idx + 1

                    real_idx = chunk_indices[idx]
                    # We copy the slice from the memory map to RAM
                    chunk_data = mmap_arr[real_idx].copy()

                    # Convert to PyTorch tensor (int64 for embeddings)
                    tensor = torch.from_numpy(chunk_data.astype(np.int64))
                    batch.append(tensor)

                    if len(batch) == self.batch_size:
                        yield torch.stack(batch, dim=0)
                        batch = []

                # Yield remainder if any (optional, depending on strict batch size requirements)
                if len(batch) > 0 and len(batch) == self.batch_size:
                     yield torch.stack(batch, dim=0)

                self.current_file_idx += 1
                self.current_chunk_idx = 0

            # Epoch finished
            self.epoch += 1
            self._init_epoch()

    def state_dict(self) -> Dict[str, Any]:
        """Return the cursor state for precise resumption."""
        return {
            "epoch": self.epoch,
            "current_file_idx": self.current_file_idx,
            "current_chunk_idx": self.current_chunk_idx,
            "seed": self.seed
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restore the cursor state."""
        self.epoch = state_dict["epoch"]
        self.seed = state_dict["seed"]

        # We must call init_epoch BEFORE restoring indices so the file list is shuffled identically
        # to how it was during that specific epoch
        self._init_epoch()

        self.current_file_idx = state_dict["current_file_idx"]
        self.current_chunk_idx = state_dict["current_chunk_idx"]
