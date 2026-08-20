import unittest
from pathlib import Path

import torch

from mvtracker.profiling.updateformer_contract import (
    WORKLOADS,
    _close_mismatch,
    _exact_mismatch,
    tensor_record,
)
from mvtracker.models.core.cotracker2.blocks import (
    EfficientUpdateFormer,
    FlashAttention,
    FusedFlashAttention,
    updateformer_track_capacity,
)


class UpdateFormerContractTests(unittest.TestCase):
    def test_fused_backend_uses_stable_capacity_shapes(self):
        self.assertEqual(updateformer_track_capacity(333), 512)
        self.assertEqual(updateformer_track_capacity(777), 1024)
        self.assertEqual(updateformer_track_capacity(1100), 1280)
        self.assertEqual(updateformer_track_capacity(1536), 2048)
        with self.assertRaises(ValueError):
            updateformer_track_capacity(2049)

    def test_fused_attention_loads_the_eager_state_dict(self):
        eager = FlashAttention(24, num_heads=3, dim_head=8, qkv_bias=True)
        fused = FusedFlashAttention(24, num_heads=3, dim_head=8, qkv_bias=True)
        fused.load_state_dict(eager.state_dict(), strict=True)

        eager_inputs = torch.randn(2, 7, 24, requires_grad=True)
        fused_inputs = eager_inputs.detach().clone().requires_grad_(True)
        eager_output = eager(eager_inputs)
        fused_output = fused(fused_inputs)
        torch.testing.assert_close(eager_output, fused_output, rtol=1e-5, atol=1e-6)
        weights = torch.randn_like(eager_output)
        (eager_output * weights).sum().backward()
        (fused_output * weights).sum().backward()
        torch.testing.assert_close(
            eager_inputs.grad, fused_inputs.grad, rtol=1e-5, atol=1e-6
        )
        for eager_parameter, fused_parameter in zip(
            eager.parameters(), fused.parameters()
        ):
            torch.testing.assert_close(
                eager_parameter.grad,
                fused_parameter.grad,
                rtol=1e-5,
                atol=1e-6,
            )

    def test_checkpointing_is_not_the_default_execution_path(self):
        model = EfficientUpdateFormer(
            space_depth=1,
            time_depth=1,
            input_dim=16,
            hidden_size=24,
            num_heads=3,
            output_dim=8,
            num_virtual_tracks=4,
            attn_class=FlashAttention,
        )

        self.assertFalse(model.checkpoint_updateformer)
        self.assertEqual(model.execution_backend, "eager")

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

    def test_bfloat16_forward_contract_allows_one_ulp(self):
        expected = torch.tensor([1.0], dtype=torch.bfloat16)
        one_ulp = torch.nextafter(expected, torch.tensor([2.0], dtype=torch.bfloat16))
        two_ulp = torch.nextafter(one_ulp, torch.tensor([2.0], dtype=torch.bfloat16))

        self.assertIsNone(_close_mismatch(expected, one_ulp))
        self.assertIsNotNone(_close_mismatch(expected, two_ulp))

    def test_workloads_cover_single_and_padded_physical_batches(self):
        self.assertEqual({workload.batch_size for workload in WORKLOADS}, {1, 2, 4})
        self.assertTrue(any(workload.tracks == 777 for workload in WORKLOADS))
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
