import unittest

from scripts.export_verl_lora import ROOT, inventory


class ExportVerlLoraTests(unittest.TestCase):
    def test_inventory_hashes_adapter_files(self) -> None:
        rows = inventory(ROOT / "tests" / "fixtures" / "merge_output")
        self.assertTrue(rows)
        self.assertTrue(all(len(row["sha256"]) == 64 for row in rows))

    def test_export_job_is_single_gpu_and_non_overwriting(self) -> None:
        script = (ROOT / "scripts" / "slurm_export_f10_best_adapter.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --gres=gpu:pro6000:1", script)
        self.assertIn('SOURCE_STEP="${SOURCE_STEP:-50}"', script)
        self.assertIn('SELECTION_SCORE="${SELECTION_SCORE:-0.269231}"', script)
        self.assertIn('--step "$SOURCE_STEP"', script)
        self.assertIn('--metric-value "$SELECTION_SCORE"', script)
        exporter = (ROOT / "scripts" / "export_verl_lora.py").read_text(encoding="utf-8")
        self.assertIn("Refusing to overwrite adapter output", exporter)
        self.assertIn("validate_parent_adapter_generation", exporter)


if __name__ == "__main__":
    unittest.main()
