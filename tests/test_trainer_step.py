import unittest
import torch
from arch_a import ArchAConfig, ArchAForCausalLM

class TestTrainerStep(unittest.TestCase):
    def test_single_forward_backward(self):
        config = ArchAConfig.for_debug()
        model = ArchAForCausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Batch simulation
        x = torch.randint(0, config.vocab_size, (2, 32))

        optimizer.zero_grad()
        out = model(x, labels=x)

        self.assertIsNotNone(out.loss)
        out.loss.backward()

        # Check that gradients flow to embeddings
        has_grad = False
        for p in model.parameters():
            if p.grad is not None:
                has_grad = True
                break

        self.assertTrue(has_grad, "No gradients were computed during the backward pass.")
        optimizer.step()

if __name__ == "__main__":
    unittest.main()
