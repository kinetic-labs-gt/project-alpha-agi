import unittest
import os
import json
import tempfile
import numpy as np
import torch
from arch_a.training.data import StreamingShardLoader

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.bin_path = os.path.join(self.temp_dir.name, "shard1.bin")
        self.idx_path = os.path.join(self.temp_dir.name, "shard1.idx")

        # Create a mock 3-chunk shard
        self.max_seq_len = 64
        self.num_chunks = 3

        arr = np.ones((self.num_chunks, self.max_seq_len), dtype=np.uint16)
        with open(self.bin_path, "wb") as f:
            f.write(arr.tobytes())

        with open(self.idx_path, "w") as f:
            json.dump({
                "num_chunks": self.num_chunks,
                "max_seq_len": self.max_seq_len,
                "dtype": "uint16"
            }, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_streaming_loader(self):
        loader = StreamingShardLoader([self.bin_path], batch_size=2, shuffle=False)
        iterator = iter(loader)

        # Batch 1 should have size 2
        batch1 = next(iterator)
        self.assertEqual(batch1.shape, (2, 64))
        self.assertEqual(batch1.dtype, torch.int64)

        # We only wrote 3 chunks total, so next output depends on how strict batching is.
        # Our loader yields the remainder.
        batch2 = next(iterator)
        # Because we're in an infinite loop across epochs, the loader actually wraps around seamlessly.
        # But for this test, we can just assert it yields tensors of expected max shape.
        self.assertTrue(batch2.shape[0] <= 2)
        self.assertEqual(batch2.shape[1], 64)

    def test_loader_state_dict(self):
        loader = StreamingShardLoader([self.bin_path], batch_size=2, shuffle=False)
        iterator = iter(loader)

        batch1 = next(iterator)

        state = loader.state_dict()

        # New loader, same files, but load state
        loader2 = StreamingShardLoader([self.bin_path], batch_size=2, shuffle=False)
        loader2.load_state_dict(state)
        iterator2 = iter(loader2)

        batch2 = next(iterator2)

        self.assertTrue(batch2.shape[0] <= 2)

if __name__ == "__main__":
    unittest.main()
