import unittest

from mvtracker.profiling.modal_training import (
    TRAJECTORY_CANDIDATES,
    SearchResult,
    TrialResult,
    find_largest_safe,
    is_memory_safe,
    validate_gpu_request,
)


class ModalTrainingProfileTests(unittest.TestCase):
    def test_trajectory_candidates_cover_planned_search_space(self):
        self.assertEqual(
            TRAJECTORY_CANDIDATES,
            (256, 512, 768, 1024, 1280, 1536, 1792, 2048),
        )

    def test_memory_safety_includes_ninety_percent_boundary(self):
        total_bytes = 80_000

        self.assertTrue(is_memory_safe(72_000, total_bytes))
        self.assertFalse(is_memory_safe(72_001, total_bytes))

    def test_search_probes_ceiling_first_and_selects_largest_safe_candidate(self):
        probed = []

        def probe(trajectories):
            probed.append(trajectories)
            status = "safe" if trajectories <= 1280 else "unsafe"
            return TrialResult(trajectories=trajectories, status=status)

        result = find_largest_safe(probe)

        self.assertEqual(result.selected_trajectories, 1280)
        self.assertEqual(probed, [2048, 1024, 1536, 1280])
        self.assertEqual(
            tuple(trial.trajectories for trial in result.trials),
            tuple(probed),
        )

    def test_search_stops_after_safe_ceiling(self):
        result = find_largest_safe(
            lambda trajectories: TrialResult(trajectories, "safe")
        )

        self.assertEqual(result.selected_trajectories, 2048)
        self.assertEqual(result.trials, (TrialResult(2048, "safe"),))

    def test_oom_trial_is_recorded_and_search_continues(self):
        def probe(trajectories):
            if trajectories == 1536:
                return TrialResult(
                    trajectories=trajectories,
                    status="oom",
                    peak_memory_bytes=80_000,
                    total_memory_bytes=80_000,
                    result_path="trials/1536.json",
                )
            status = "safe" if trajectories <= 1280 else "unsafe"
            return TrialResult(trajectories=trajectories, status=status)

        result = find_largest_safe(probe)

        self.assertIsInstance(result, SearchResult)
        self.assertEqual(result.selected_trajectories, 1280)
        self.assertEqual(
            [(trial.trajectories, trial.status) for trial in result.trials],
            [
                (2048, "unsafe"),
                (1024, "safe"),
                (1536, "oom"),
                (1280, "safe"),
            ],
        )
        oom_trial = result.trials[2]
        self.assertEqual(oom_trial.peak_memory_bytes, 80_000)
        self.assertEqual(oom_trial.total_memory_bytes, 80_000)
        self.assertEqual(oom_trial.result_path, "trials/1536.json")

    def test_search_returns_none_when_no_candidate_is_safe(self):
        result = find_largest_safe(
            lambda trajectories: TrialResult(trajectories, "oom")
        )

        self.assertIsNone(result.selected_trajectories)
        self.assertTrue(result.trials)
        self.assertTrue(all(trial.status == "oom" for trial in result.trials))

    def test_gpu_request_parser_enforces_two_gpu_hard_limit(self):
        self.assertEqual(validate_gpu_request("H100!"), 1)
        self.assertEqual(validate_gpu_request("H100!:2"), 2)

        with self.assertRaisesRegex(ValueError, "hard limit of 2"):
            validate_gpu_request("H100!:3")
        with self.assertRaisesRegex(ValueError, "hard limit of 2"):
            validate_gpu_request("H100!:8")

    def test_gpu_request_cap_is_configurable_but_never_implicit(self):
        self.assertEqual(validate_gpu_request("H100!:1", max_gpus=1), 1)
        with self.assertRaisesRegex(ValueError, "hard limit of 1"):
            validate_gpu_request("H100!:2", max_gpus=1)


if __name__ == "__main__":
    unittest.main()
