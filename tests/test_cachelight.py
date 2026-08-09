import tempfile
import unittest
from pathlib import Path

from cachelight import discover_scan_targets, resolve_hub_cache_path


class CachelightTests(unittest.TestCase):
    def test_nested_huggingface_hub_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "hub" / "models--owner--model"
            model.mkdir(parents=True)
            self.assertEqual(resolve_hub_cache_path(root), root / "hub")
            items, other_targets, _ = discover_scan_targets(root)
            self.assertEqual(len(items), 1)
            self.assertEqual(other_targets, [])
            self.assertEqual(items[0][2], "owner/model")

    def test_direct_hf_hub_cache_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "models--owner--model"
            model.mkdir()
            self.assertEqual(resolve_hub_cache_path(root), root)
            items, other_targets, _ = discover_scan_targets(root)
            self.assertEqual(len(items), 1)
            self.assertEqual(other_targets, [])
            self.assertEqual(items[0][2], "owner/model")


if __name__ == "__main__":
    unittest.main()
