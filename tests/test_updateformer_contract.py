import unittest
from pathlib import Path

import torch

from mvtracker.profiling.updateformer_contract import (
    WORKLOADS,
    _close_mismatch,
    _exact_mismatch,
    tensor_record,
)


class UpdateFormerContractTests(unittest.TestCase):
    def test_tensor_record_detects_a_single_value_change(self):
        original = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        changed = original.clone()
        changed[1, 2] += 1

        self.assertNotEqual(
            tensor_record(original)["sha256"], tensor_record(changed)["sha256"]
        )

    def test_tensor_record_hashes_scalar_loss(self):
        record = tensor_record(torch.tensor(1.25))

        self.assertEqual(record["shape"], [])
        self.assertEqual(len(record["sha256"]), 64)

    def test_exact_mismatch_reports_nested_difference(self):
        expected = {"gradient": [{"sha256": "abc"}]}
        actual = {"gradient": [{"sha256": "abd"}]}

        self.assertEqual(
            _exact_mismatch(expected, actual),
            "root.gradient[0].sha256: 'abd' != 'abc'",
        )

    def test_float_contract_allows_only_tiny_kernel_noise(self):
        expected = torch.tensor([1.0, 0.0], dtype=torch.float32)

        self.assertIsNone(
            _close_mismatch(expected, expected + torch.tensor([1e-5, 1e-6]))
        )
        self.assertIsNotNone(
            _close_mismatch(expected, expected + torch.tensor([1e-2, 0.0]))
        )

    def test_bfloat16_forward_contract_is_exact(self):
        expected = torch.tensor([1.0], dtype=torch.bfloat16)
        actual = torch.tensor([1.01], dtype=torch.bfloat16)

        self.assertIsNotNone(_close_mismatch(expected, actual))

    def test_workloads_cover_single_and_padded_physical_batches(self):
        self.assertEqual({workload.batch_size for workload in WORKLOADS}, {1, 2, 4})
        for workload in WORKLOADS:
            self.assertEqual(len(workload.real_tracks), workload.batch_size)
            self.assertTrue(all(0 < count <= workload.tracks for count in workload.real_tracks))

    def test_modal_launcher_is_single_gpu_and_tagged(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "tools/modal_updateformer_autoresearch.py"
        ).read_text(encoding="utf-8")

        self.assertIn('gpu="H100!"', source)
        self.assertNotIn('gpu="H100!:2"', source)
        self.assertIn('"owner": "jeet"', source)
        self.assertIn('"project": "mvtracker"', source)
        self.assertIn('"purpose": "profiling"', source)


if __name__ == "__main__":
    unittest.main()
