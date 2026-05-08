import unittest
from arch_a import ArchAConfig

class TestConfig50m(unittest.TestCase):
    def test_for_50m_preset(self):
        config = ArchAConfig.for_50m()

        # Verify sizes generally fit the Phase 1 blueprint
        self.assertTrue(32000 <= config.vocab_size <= 50000)
        self.assertTrue(512 <= config.d_model <= 640)
        self.assertTrue(8 <= config.n_layers <= 10)
        self.assertTrue(8 <= config.n_heads <= 10)
        self.assertEqual(config.n_kv_heads, 4)

        # Internal logical constraints check
        self.assertEqual(config.d_model, config.n_heads * config.d_head, "d_model must equal n_heads * d_head")

if __name__ == "__main__":
    unittest.main()
