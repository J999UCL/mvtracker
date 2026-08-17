import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


TRAIN_PATH = Path(__file__).resolve().parents[1] / "mvtracker/cli/train.py"
TRAIN_TREE = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"), filename=str(TRAIN_PATH))


def _node(name, node_type=ast.FunctionDef):
    return next(
        node for node in TRAIN_TREE.body
        if isinstance(node, node_type) and node.name == name
    )


def _load(name, namespace):
    node = next(
        node for node in TRAIN_TREE.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    values = dict(namespace)
    exec(compile(ast.fix_missing_locations(module), str(TRAIN_PATH), "exec"), values)
    return values[name]


class AttrDict(dict):
    __getattr__ = dict.__getitem__


class MixedTrainingIntegrationTests(unittest.TestCase):
    def test_all_worker_pools_start_before_cuda_producers(self):
        events = []

        class RawLoader:
            def __init__(self, source):
                self.source = source

            def __iter__(self):
                events.append(f"workers:{self.source}")
                return iter(())

        class PrefetchLoader:
            def __init__(self, source):
                self.source = source
                self.loader = RawLoader(source)

            def iter_from(self, iterator):
                events.append(f"producer:{self.source}")
                return iterator

        start = _load("_start_mixed_source_iterators", {})
        start({source: PrefetchLoader(source) for source in ("diegesis", "mvkubric")})

        self.assertEqual(
            events,
            [
                "workers:diegesis",
                "workers:mvkubric",
                "producer:diegesis",
                "producer:mvkubric",
            ],
        )

    def test_source_shape_metrics_accept_float_padding_masks(self):
        metrics = _load(
            "_source_batch_shape_metrics",
            {"torch": torch},
        )
        batch = SimpleNamespace(
            video=torch.empty(2, 4, 3),
            trajectory=torch.empty(2, 3, 5, 3),
            track_padding_mask=torch.tensor(
                [[0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0]]
            ),
        )

        self.assertEqual(metrics(batch), (4, 3.5))

    def test_source_sampler_uses_persisted_rank_local_cursor(self):
        sampler_class = _load(
            "_ScheduledSourceSampler",
            {"torch": torch},
        )

        class Schedule:
            def sample_source(self, source, cursor, rank):
                return SimpleNamespace(request=(source, cursor, rank))

        sampler = sampler_class(Schedule(), "diegesis", rank=1, request_count=3)
        sampler.set_start_cursor(7)
        self.assertEqual(
            list(sampler),
            [("diegesis", 7, 1), ("diegesis", 8, 1), ("diegesis", 9, 1)],
        )

    def test_source_dataset_factory_honors_roots_allowlists_and_tap_probabilities(self):
        calls = []

        class FakeKubric:
            @staticmethod
            def from_name(*args, **kwargs):
                calls.append(("kubric", args, kwargs))
                return "kubric-dataset"

        class FakeTap:
            @staticmethod
            def from_name(*args, **kwargs):
                calls.append(("tap", args, kwargs))
                return {"marker": True}

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        build = _load(
            "_build_training_dataset",
            {
                "KubricMultiViewDataset": FakeKubric,
                "PointOdysseyMultiViewDataset": object,
                "TapVid3DMultiViewDataset": FakeTap,
            },
        )
        source = AttrDict(
            include_scene_ids=["a", "b"],
            exclude_scene_ids=[],
            view_count_probabilities=[0.25] * 4,
        )
        tap = build("tapvid3d-multiview-training", "/diegesis", object(), object(), source)
        self.assertEqual(tap.kwargs["view_count_probabilities"], (0.25,) * 4)
        self.assertEqual(calls[0][1][1], "/diegesis")
        self.assertEqual(calls[0][2]["include_scene_ids"], ["a", "b"])

        result = build(
            "kubric-multiview-v3-training",
            "/mvkubric",
            object(),
            object(),
            AttrDict(include_scene_ids=["900", "997"], exclude_scene_ids=[]),
        )
        self.assertEqual(result, "kubric-dataset")
        self.assertEqual(calls[1][1][1], "/mvkubric")
        self.assertEqual(calls[1][2]["include_scene_ids"], ["900", "997"])

    def test_configured_wandb_id_is_persisted_and_mismatch_rejected(self):
        helper = _load(
            "_get_or_create_wandb_run_id",
            {
                "Path": Path,
                "WANDB_RUN_ID_FILE": "wandb_run_id.txt",
                "wandb": SimpleNamespace(
                    util=SimpleNamespace(generate_id=lambda: "generated")
                ),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(helper(directory, "configured"), "configured")
            self.assertEqual(helper(directory, "configured"), "configured")
            with self.assertRaisesRegex(ValueError, "does not match"):
                helper(directory, "different")

    def test_main_encodes_mixed_retry_resume_metrics_and_single_eval_contract(self):
        source = ast.unparse(_node("main"))
        self.assertIn("('diegesis', 'mvkubric', 'diegesis', 'mvkubric')", source)
        self.assertIn("source_cursors[current_source] += 1", source)
        self.assertIn("if not _all_ranks_succeeded", source)
        self.assertLess(
            source.index("source_cursors[current_source] += 1"),
            source.index("if not _all_ranks_succeeded"),
        )
        self.assertIn("state.mixed_schedule_state", source)
        self.assertIn("state.source_cursors", source)
        self.assertIn("checkpoint_source_cursors = dict(source_cursors)", source)
        self.assertIn("state.source_cursors = dict(checkpoint_source_cursors)", source)
        self.assertIn("source/{source}/component/{component}", source)
        self.assertIn("source/{source}/sample_count", source)
        self.assertIn("if last_eval_step != total_steps", source)
        self.assertIn("cfg.logging.get('wandb_run_name')", source)
        self.assertIn("cfg.logging.get('wandb_run_id')", source)

    def test_checkpoint_state_carries_schedule_and_source_cursors(self):
        source = ast.unparse(_node("_save_training_checkpoint"))
        self.assertIn("state.mixed_schedule_state = mixed_schedule_state", source)
        self.assertIn("state.source_cursors = dict(source_cursors)", source)


if __name__ == "__main__":
    unittest.main()
