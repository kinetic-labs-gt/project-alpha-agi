import unittest
from arch_a import ArchAConfig

class TestConfigPresets(unittest.TestCase):
    def test_presets_exist(self):
        presets = ['for_debug', 'for_2gpu_demo', 'for_20m', 'for_50m', 'for_500m', 'for_1b']
        for preset in presets:
            self.assertTrue(hasattr(ArchAConfig, preset), f"Preset {preset} missing from ArchAConfig")
            config = getattr(ArchAConfig, preset)()
            self.assertIsInstance(config, ArchAConfig)
            self.assertEqual(config.n_heads * config.d_head, config.d_model, f"n_heads * d_head != d_model in preset {preset}")

if __name__ == '__main__':
    unittest.main()
