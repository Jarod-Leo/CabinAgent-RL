import unittest

from scripts.launch_verl import ROOT
from scripts.merge_lora_parent import sha256_file
from scripts.validate_merged_parent import validate_inventory


class ValidateMergedParentTests(unittest.TestCase):
    def test_inventory_validates_exact_static_fixture(self) -> None:
        directory = ROOT / "tests/fixtures/merge_output"
        weights = directory / "weights.safetensors"
        manifest = {
            "inventory": [
                {
                    "path": "weights.safetensors",
                    "size_bytes": weights.stat().st_size,
                    "sha256": sha256_file(weights),
                }
            ]
        }
        verified = validate_inventory(directory, manifest)
        self.assertEqual(len(verified), 1)

    def test_inventory_rejects_wrong_digest(self) -> None:
        directory = ROOT / "tests/fixtures/merge_output"
        weights = directory / "weights.safetensors"
        manifest = {
            "inventory": [
                {
                    "path": "weights.safetensors",
                    "size_bytes": weights.stat().st_size,
                    "sha256": "0" * 64,
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_inventory(directory, manifest)


if __name__ == "__main__":
    unittest.main()
