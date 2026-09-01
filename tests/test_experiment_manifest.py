import unittest

from scripts import update_experiment_manifest


class ExperimentManifestTests(unittest.TestCase):
    def test_lifecycle_metadata_preserves_reproducibility_fields(self) -> None:
        updated = update_experiment_manifest.with_lifecycle_metadata(
            {"run_id": "run-1", "source_sha256": "abc"},
            "running",
            slurm_job_id="123",
            node_list="node-[1-2]",
        )

        self.assertEqual(updated["status"], "running")
        self.assertEqual(updated["slurm_job_id"], "123")
        self.assertEqual(updated["slurm_node_list"], "node-[1-2]")
        self.assertEqual(updated["source_sha256"], "abc")
        self.assertIn("updated_at", updated)


if __name__ == "__main__":
    unittest.main()
