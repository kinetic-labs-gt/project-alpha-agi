import unittest
import os
import tempfile
import torch
from arch_a import ArchAConfig, ArchAForCausalLM
from arch_a.training.checkpointing import CheckpointManager

class TestCheckpointResume(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = ArchAConfig.for_debug()
        self.device = torch.device("cpu")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_manager_save_and_load(self):
        model1 = ArchAForCausalLM(self.config).to(self.device)
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
        manager = CheckpointManager(save_dir=self.temp_dir.name)

        # Modify weights slightly
        with torch.no_grad():
            for p in model1.parameters():
                p.add_(1.0)

        # Save
        manager.save_checkpoint(model1, optimizer1, None, global_step=10, config_dict=self.config.to_dict(), val_loss=0.5)

        # New model
        model2 = ArchAForCausalLM(self.config).to(self.device)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

        ckpt_path = os.path.join(self.temp_dir.name, "latest.pt")
        self.assertTrue(os.path.exists(ckpt_path))

        # Load
        state = manager.load_checkpoint(ckpt_path, model2, optimizer2, None)
        self.assertEqual(state["global_step"], 10)
        self.assertEqual(state["val_loss"], 0.5)

        # Verify weights match
        p1 = next(model1.parameters())
        p2 = next(model2.parameters())
        self.assertTrue(torch.allclose(p1, p2))

if __name__ == "__main__":
    unittest.main()
