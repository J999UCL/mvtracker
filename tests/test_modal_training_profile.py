import unittest
from pathlib import Path

import torch

from mvtracker.models.core.embeddings import get_3d_sincos_pos_embed_from_grid
from mvtracker.profiling.modal_training import (
    BATCH_CANDIDATES,
    FRONTIER_BATCH_CANDIDATES,
    FRONTIER_CASES,
    FRONTIER_TRAJECTORY_CANDIDATES,
    HIGH_TRAJECTORY_CANDIDATES,
    TRAJECTORY_CANDIDATES,
    PROFILE_CASES,
    SearchResult,
    TrialResult,
    find_largest_safe,
    find_largest_safe_batch,
    is_memory_safe,
    validate_gpu_request,
    validate_view_count,
)


class ModalTrainingProfileTests(unittest.TestCase):
    def test_dependency_layers_precede_commit_specific_source_checkout(self):
        source = (
            Path(__file__).resolve().parents[1] / "tools/modal_training_profile.py"
        ).read_text(encoding="utf-8")
        dependency_start = source.index("def _dependency_image()")
        source_start = source.index("def _source_image(")
        runtime_start = source.index("def _runtime_image()")
        dependency = source[dependency_start:source_start]
        source_layer = source[source_start:runtime_start]
        runtime = source[runtime_start:source.index("\n\napp = modal.App", runtime_start)]

        self.assertIn("pip_install_from_requirements", dependency)
        self.assertIn("flash-attn==2.8.3.post1", dependency)
        self.assertIn("pointops", dependency)
        self.assertIn("build_indexed_correlation_extension.py", dependency)
        self.assertNotIn("_source_commit()", dependency)
        self.assertIn("_source_image(_dependency_image())", runtime)
        self.assertIn(".run_commands(clone)", source_layer)
        self.assertIn("checkout --detach FETCH_HEAD", source_layer)
        self.assertIn("rev-parse HEAD", source_layer)
        self.assertIn("cp /opt/mvtracker-extension/", source_layer)
        self.assertNotIn("build_indexed_correlation_extension.py", source_layer)

    def test_runtime_installs_lossless_gpu_image_codecs_and_zstd(self):
        source = (
            Path(__file__).resolve().parents[1] / "tools/modal_training_profile.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"zstd"', source)
        self.assertIn('"nvidia-nvimgcodec-cu12[nvtiff]==0.9.0.20"', source)
        self.assertIn('"nvidia-libnvcomp-cu12==5.3.0.16"', source)

    def test_3d_embedding_accepts_bfloat16_coordinates(self):
        grid = torch.tensor(
            [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]],
            dtype=torch.bfloat16,
        )

        actual = get_3d_sincos_pos_embed_from_grid(12, grid)
        expected = get_3d_sincos_pos_embed_from_grid(12, grid.float())

        torch.testing.assert_close(actual, expected)

    def test_trajectory_candidates_cover_planned_search_space(self):
        self.assertEqual(
            TRAJECTORY_CANDIDATES,
            (1024, 2048),
        )

    def test_profile_matrix_is_four_views_by_two_trajectory_targets(self):
        self.assertEqual(len(PROFILE_CASES), 8)
        self.assertEqual(
            {(case.views, case.trajectories) for case in PROFILE_CASES},
            {(views, trajectories) for views in range(1, 5) for trajectories in (1024, 2048)},
        )
        self.assertEqual(BATCH_CANDIDATES, tuple(range(1, 9)))

    def test_five_six_view_frontier_search_space(self):
        self.assertEqual(
            {(case.views, case.trajectories) for case in FRONTIER_CASES},
            {(views, trajectories) for views in (5, 6) for trajectories in (1024, 2048)},
        )
        self.assertEqual(FRONTIER_BATCH_CANDIDATES, tuple(range(1, 13)))
        self.assertEqual(FRONTIER_TRAJECTORY_CANDIDATES[0], 1024)
        self.assertEqual(FRONTIER_TRAJECTORY_CANDIDATES[-1], 6144)
        self.assertTrue(
            all(
                right - left == 512
                for left, right in zip(
                    FRONTIER_TRAJECTORY_CANDIDATES,
                    FRONTIER_TRAJECTORY_CANDIDATES[1:],
                )
            )
        )
        self.assertEqual(HIGH_TRAJECTORY_CANDIDATES[0], 6144)
        self.assertEqual(HIGH_TRAJECTORY_CANDIDATES[-1], 8192)

    def test_memory_safety_includes_ninety_percent_boundary(self):
        total_bytes = 80_000

        self.assertTrue(is_memory_safe(72_000, total_bytes))
        self.assertFalse(is_memory_safe(72_001, total_bytes))

    def test_search_probes_ceiling_first_and_selects_largest_safe_candidate(self):
        probed = []

        def probe(trajectories):
            probed.append(trajectories)
            status = "safe" if trajectories <= 1024 else "unsafe"
            return TrialResult(requested=trajectories, status=status)

        result = find_largest_safe(probe)

        self.assertEqual(result.selected_trajectories, 1024)
        self.assertEqual(probed, [2048, 1024])
        self.assertEqual(
            tuple(trial.requested for trial in result.trials),
            tuple(probed),
        )

    def test_search_stops_after_safe_ceiling(self):
        result = find_largest_safe(
            lambda trajectories: TrialResult(trajectories, "safe")
        )

        self.assertEqual(result.selected_trajectories, 2048)
        self.assertEqual(result.trials, (TrialResult(2048, "safe"),))

    def test_batch_search_uses_eight_as_ceiling_and_continues_after_oom(self):
        def probe(trajectories):
            if trajectories == 6:
                return TrialResult(
                    requested=trajectories,
                    status="oom",
                    peak_memory_bytes=80_000,
                    total_memory_bytes=80_000,
                    result_path="trials/1536.json",
                )
            status = "safe" if trajectories <= 4 else "unsafe"
            return TrialResult(requested=trajectories, status=status)

        result = find_largest_safe_batch(probe)

        self.assertIsInstance(result, SearchResult)
        self.assertEqual(result.selected_trajectories, 4)
        self.assertEqual(
            [(trial.requested, trial.status) for trial in result.trials],
            [
                (8, "unsafe"),
                (4, "safe"),
                (6, "oom"),
                (5, "unsafe"),
            ],
        )
        oom_trial = result.trials[2]
        self.assertEqual(oom_trial.peak_memory_bytes, 80_000)
        self.assertEqual(oom_trial.total_memory_bytes, 80_000)
        self.assertEqual(oom_trial.result_path, "trials/1536.json")

    def test_search_returns_none_when_no_candidate_is_safe(self):
        result = find_largest_safe(
            lambda trajectories: TrialResult(trajectories, "oom"),
            candidates=BATCH_CANDIDATES,
        )

        self.assertIsNone(result.selected_trajectories)
        self.assertTrue(result.trials)
        self.assertTrue(all(trial.status == "oom" for trial in result.trials))

    def test_gpu_request_parser_enforces_one_gpu_hard_limit(self):
        self.assertEqual(validate_gpu_request("H100!"), 1)

        with self.assertRaisesRegex(ValueError, "hard limit of 1"):
            validate_gpu_request("H100!:2")
        with self.assertRaisesRegex(ValueError, "hard limit of 1"):
            validate_gpu_request("H100!:8")

    def test_gpu_request_cap_is_configurable_but_never_implicit(self):
        self.assertEqual(validate_gpu_request("H100!:1", max_gpus=1), 1)
        with self.assertRaisesRegex(ValueError, "hard limit of 1"):
            validate_gpu_request("H100!:2", max_gpus=1)

    def test_profile_view_count_accepts_full_mv_kubric_range(self):
        validate_view_count(1)
        validate_view_count(6)
        with self.assertRaisesRegex(ValueError, "between one and six"):
            validate_view_count(7)


if __name__ == "__main__":
    unittest.main()
