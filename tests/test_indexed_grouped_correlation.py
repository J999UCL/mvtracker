import importlib.util
import sys
import unittest
from pathlib import Path

import torch


def _load_harness():
    path = Path(__file__).resolve().parents[1] / "scripts" / "indexed_grouped_correlation_benchmark.py"
    spec = importlib.util.spec_from_file_location("indexed_grouped_correlation_harness", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


class IndexedGroupedCorrelationTests(unittest.TestCase):
    def _fixture(self, groups=2):
        torch.manual_seed(19)
        batch_size, points, queries, channels, neighbors = 2, 11, 5, 8, 3
        xyz = torch.randn(batch_size, points, 3, requires_grad=True)
        fvec = torch.randn(batch_size, points, channels, requires_grad=True)
        targets = torch.randn(batch_size, queries, channels, requires_grad=True)
        query_xyz = torch.randn(batch_size, queries, 3, requires_grad=True)
        indices = torch.tensor(
            [[[0, 4, 8], [1, 2, 9], [3, 5, 7], [0, 6, 10], [2, 4, 9]],
             [[1, 3, 6], [0, 5, 8], [2, 7, 10], [4, 6, 9], [1, 5, 10]]]
        )
        return targets, query_xyz, xyz, fvec, indices, groups

    def test_indexed_candidate_matches_eager_oracle_forward(self):
        values = self._fixture()
        oracle = harness.correlation_from_indices(*values, True, True)
        candidate = harness.indexed_grouped_candidate(*values, True, True)
        torch.testing.assert_close(candidate, oracle)

    def test_indexed_candidate_matches_eager_oracle_gradients(self):
        left = self._fixture()
        right = tuple(value.detach().clone().requires_grad_(value.is_floating_point()) if torch.is_tensor(value) else value
                      for value in left)
        oracle = harness.correlation_from_indices(*left, True, True)
        candidate = harness.indexed_grouped_candidate(*right, True, True)
        weights = torch.linspace(0.5, 1.5, oracle.numel()).reshape_as(oracle)
        (oracle * weights).sum().backward()
        (candidate * weights).sum().backward()
        for oracle_value, candidate_value in zip(left[:4], right[:4]):
            torch.testing.assert_close(oracle_value.grad, candidate_value.grad)

    def test_bad_grouping_self_test_fails(self):
        values = self._fixture()
        oracle = harness.correlation_from_indices(*values, True, True)
        bad = harness.deliberately_bad_grouping(*values, True, True)
        self.assertNotEqual(oracle.shape, bad.shape)

    def test_current_pointcloud_block_matches_oracle_with_mocked_knn(self):
        targets, query_xyz, xyz, fvec, indices, groups = self._fixture()
        block_class, namespace = harness.load_current_pointcloud_corr_block()
        namespace["knn"] = harness.mocked_knn(indices)
        block = block_class(indices.shape[-1], groups, xyz, fvec, True, True)
        actual = block.corr_sample(targets, query_xyz)
        expected = harness.correlation_from_indices(
            targets, query_xyz, xyz, fvec, indices, groups, True, True
        )
        torch.testing.assert_close(actual, expected)

    def test_current_pointcloud_block_matches_oracle_gradients(self):
        values = self._fixture(groups=1)
        targets, query_xyz, xyz, fvec, indices, groups = values
        block_class, namespace = harness.load_current_pointcloud_corr_block()
        namespace["knn"] = harness.mocked_knn(indices)
        block = block_class(indices.shape[-1], groups, xyz, fvec, True, True)
        actual = block.corr_sample(targets, query_xyz)
        expected = harness.correlation_from_indices(
            targets, query_xyz, xyz, fvec, indices, groups, True, True
        )
        weights = torch.linspace(0.5, 1.5, actual.numel()).reshape_as(actual)
        (actual * weights).sum().backward()
        grad_actual = [value.grad.detach().clone() for value in (targets, query_xyz, xyz, fvec)]

        targets2, query_xyz2, xyz2, fvec2, _, _ = self._fixture(groups=1)
        expected2 = harness.correlation_from_indices(
            targets2, query_xyz2, xyz2, fvec2, indices, groups, True, True
        )
        (expected2 * weights).sum().backward()
        for actual_grad, expected_value in zip(
            grad_actual, (targets2, query_xyz2, xyz2, fvec2)
        ):
            torch.testing.assert_close(actual_grad, expected_value.grad)

    def test_bad_batch_indexing_self_test_fails(self):
        values = self._fixture()
        oracle = harness.correlation_from_indices(*values, True, True)
        bad = harness.deliberately_bad_batch_indexing(
            *values, True, True
        )
        self.assertFalse(torch.allclose(oracle, bad))

    def test_detached_candidate_passes_forward_but_fails_gradient_self_test(self):
        left = self._fixture()
        right = tuple(
            value.detach().clone().requires_grad_(value.is_floating_point())
            if torch.is_tensor(value) else value
            for value in left
        )
        oracle = harness.correlation_from_indices(*left, True, True)
        bad = harness.deliberately_bad_detached_backward(*right, True, True)
        torch.testing.assert_close(oracle, bad)
        oracle.sum().backward()
        bad.sum().backward()
        self.assertIsNotNone(left[3].grad)
        self.assertIsNone(right[3].grad)

    def test_duplicate_neighbors_accumulate_source_gradients(self):
        torch.manual_seed(23)
        targets = torch.ones(1, 2, 4, requires_grad=True)
        query_xyz = torch.zeros(1, 2, 3, requires_grad=True)
        xyz = torch.randn(1, 5, 3, requires_grad=True)
        fvec = torch.randn(1, 5, 4, requires_grad=True)
        indices = torch.tensor([[[2, 2, 2], [4, 4, 1]]])
        output = harness.indexed_grouped_candidate(
            targets, query_xyz, xyz, fvec, indices, 1, False, False
        )
        output.sum().backward()
        self.assertEqual(fvec.grad[0, 2].abs().sum().item(), 6.0)
        self.assertEqual(fvec.grad[0, 4].abs().sum().item(), 4.0)

    def test_production_operator_matches_oracle_forward_and_gradients(self):
        from mvtracker.models.core.mvtracker.indexed_correlation import (
            indexed_grouped_correlation,
        )

        left = self._fixture()
        targets, _, _, fvec, indices, groups = left
        targets_candidate = targets.detach().clone().requires_grad_()
        fvec_candidate = fvec.detach().clone().requires_grad_()

        expected = harness.correlation_from_indices(
            *left, False, False
        )
        actual = indexed_grouped_correlation(
            targets_candidate, fvec_candidate, indices, groups
        )
        torch.testing.assert_close(actual, expected)

        weights = torch.linspace(0.5, 1.5, expected.numel()).reshape_as(expected)
        (expected * weights).sum().backward()
        (actual * weights).sum().backward()
        torch.testing.assert_close(targets_candidate.grad, targets.grad)
        torch.testing.assert_close(fvec_candidate.grad, fvec.grad)


if __name__ == "__main__":
    unittest.main()
