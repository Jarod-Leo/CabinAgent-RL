import unittest

from scripts.merge_lora_parent import build_inventory, load_adapter_config
from scripts.launch_verl import ROOT


class MergeLoraParentTests(unittest.TestCase):
    def test_adapter_config_requires_lora(self) -> None:
        path = ROOT / "tests/fixtures/merge_adapter"
        self.assertEqual(load_adapter_config(path)["r"], 16)

    def test_inventory_hashes_files_and_excludes_manifest(self) -> None:
        path = ROOT / "tests/fixtures/merge_output"
        inventory = build_inventory(path)
        self.assertEqual([item["path"] for item in inventory], ["weights.safetensors"])
        self.assertEqual(inventory[0]["size_bytes"], len((path / "weights.safetensors").read_bytes()))


if __name__ == "__main__":
    unittest.main()
