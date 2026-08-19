import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


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


class _DeferredFuture:
    def __init__(self, executor, function, args, kwargs):
        self.executor = executor
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def result(self):
        self.executor.events.append(("run", self.executor.source_for(self.args)))
        return self.function(*self.args, **self.kwargs)


class _RecordingExecutor:
    """Deterministic executor that exposes submission before execution."""

    instances = []

    def __init__(self, max_workers=None):
        self.events = []
        self.max_workers = max_workers
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    @staticmethod
    def source_for(args):
        return next(
            (
                argument.source if hasattr(argument, "source") else argument
                for argument in args
                if hasattr(argument, "source")
                or argument in ("diegesis", "mvkubric")
            ),
            None,
        )

    def submit(self, function, *args, **kwargs):
        self.events.append(("submit", self.source_for(args)))
        return _DeferredFuture(self, function, args, kwargs)


class _SourceIterator:
    def __init__(self, source, candidates):
        self.source = source
        self._candidates = iter(candidates)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._candidates)


class _FabricSuccesses:
    def __init__(self, successes):
        self.successes = iter(successes)
        self.local_successes = []


class MixedTrainingIntegrationTests(unittest.TestCase):
    def test_mvkubric_recipe_uses_two_thousand_scene_training_split(self):
        config_path = Path(__file__).resolve().parents[1] / (
            "configs/experiment/diegesis_mvkubric_gt_ddp.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        mvkubric = config["datasets"]["train"]["sources"]["mvkubric"]
        self.assertNotIn("include_scene_ids", mvkubric)
        self.assertNotIn("exclude_scene_ids", mvkubric)

    def test_mvkubric_validation_schedule_keeps_datasets_separate(self):
        config_path = Path(__file__).resolve().parents[1] / (
            "configs/experiment/diegesis_mvkubric_gt_ddp.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            config["datasets"]["eval"]["schedule"],
            [
                {
                    "steps": [0, 1000],
                    "names": [
                        "tapvid3d-multiview-validation",
                        "kubric-multiview-v3-validation-full",
                    ],
                },
                {
                    "steps": [250, 500, 750],
                    "names": [
                        "tapvid3d-multiview-validation",
                        "kubric-multiview-v3-validation-subset",
                    ],
                },
            ],
        )

    def test_eval_schedule_selects_configured_names(self):
        select = _load(
            "_eval_dataset_names_for_step",
            {
                "ValueError": ValueError,
            },
        )
        cfg = SimpleNamespace(
            datasets=SimpleNamespace(
                eval=AttrDict(
                    names=["default"],
                    schedule=[
                        AttrDict(steps=[0, 1000], names=["full"]),
                        AttrDict(steps=[250, 500, 750], names=["subset"]),
                    ],
                )
            )
        )
        self.assertEqual(select(cfg, 0), ("full",))
        self.assertEqual(select(cfg, 500), ("subset",))
        with self.assertRaisesRegex(ValueError, "no evaluation dataset schedule"):
            select(cfg, 1250)

    def test_cuda_prefetch_workers_use_thread_safe_spawn_context(self):
        source = ast.unparse(_node("_build_source_train_loader"))
        self.assertIn("loader_kwargs['multiprocessing_context'] = 'spawn'", source)

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
        loader_source = ast.unparse(_node("_load_mixed_step"))
        self.assertIn("('diegesis', 'mvkubric', 'diegesis', 'mvkubric')", source)
        self.assertIn("_load_mixed_step", source)
        self.assertIn("source_cursors[source] += 1", loader_source)
        self.assertIn("_all_ranks_succeeded", loader_source)
        self.assertLess(
            loader_source.index("source_cursors[source] += 1"),
            loader_source.index("_all_ranks_succeeded"),
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


class MixedWholeStepLoaderTests(unittest.TestCase):
    pattern = ("diegesis", "mvkubric", "diegesis", "mvkubric")

    @staticmethod
    def _all_ranks_succeeded(fabric, local_success):
        fabric.local_successes.append(local_success)
        return next(fabric.successes)

    def setUp(self):
        self.load_step = _load(
            "_load_mixed_step",
            {
                "ThreadPoolExecutor": _RecordingExecutor,
                "_all_ranks_succeeded": self._all_ranks_succeeded,
            },
        )

    @staticmethod
    def _iterators(rank, start=0, count=8):
        starts = (
            start
            if isinstance(start, dict)
            else {source: start for source in ("diegesis", "mvkubric")}
        )
        return {
            source: _SourceIterator(
                source,
                [
                    (
                        SimpleNamespace(
                            label=f"{source[0]}{cursor}-r{rank}",
                            sample_metadata=[{}],
                        ),
                        [True],
                    )
                    for cursor in range(starts[source], starts[source] + count)
                ],
            )
            for source in ("diegesis", "mvkubric")
        }

    def _invoke(self, data_iters, cursors, successes):
        result = self.load_step(
            fabric=_FabricSuccesses(successes),
            source_pattern=self.pattern,
            data_iters=data_iters,
            source_samplers={},
            train_loaders={},
            source_cursors=cursors,
        )
        return (*result, _RecordingExecutor.instances[-1])

    def _load(self, rank, cursors, successes, *, start=0):
        return self._invoke(
            self._iterators(rank, start=start), cursors, successes
        )

    def test_two_ranks_load_exactly_eight_requests_in_local_dkdk_order(self):
        global_microbatches = []
        for rank in range(2):
            cursors = {"diegesis": 0, "mvkubric": 0}
            microbatches, loaded, failed, _ = self._load(
                rank, cursors, [True] * 4
            )
            self.assertEqual(len(microbatches), 4)
            self.assertEqual(
                [source for source, _ in microbatches], list(self.pattern)
            )
            self.assertEqual(loaded, 4)
            self.assertEqual(failed, 0)
            self.assertEqual(cursors, {"diegesis": 2, "mvkubric": 2})
            global_microbatches.extend((rank, *item) for item in microbatches)

        self.assertEqual(len(global_microbatches), 8)
        self.assertEqual(
            [batch.label for _, _, batch in global_microbatches],
            [
                "d0-r0", "m0-r0", "d1-r0", "m1-r0",
                "d0-r1", "m0-r1", "d1-r1", "m1-r1",
            ],
        )

    def test_submits_both_sources_before_waiting_for_either(self):
        _, _, _, executor = self._load(
            rank=0,
            cursors={"diegesis": 0, "mvkubric": 0},
            successes=[True] * 4,
        )
        first_run = next(
            index for index, event in enumerate(executor.events) if event[0] == "run"
        )
        submitted_sources = {
            source for event, source in executor.events[:first_run]
            if event == "submit"
        }
        self.assertEqual(submitted_sources, {"diegesis", "mvkubric"})

    def test_global_pair_retry_advances_cursor_and_resume_is_exact(self):
        cursors = {"diegesis": 0, "mvkubric": 0}
        uninterrupted_iters = self._iterators(rank=0)
        microbatches, loaded, failed, _ = self._invoke(
            uninterrupted_iters,
            cursors,
            # Every local sample succeeds, but the peer rank rejects the first slot.
            [False, True, True, True, True],
        )
        self.assertEqual(
            [(source, batch.label) for source, batch in microbatches],
            [
                ("diegesis", "d1-r0"),
                ("mvkubric", "m0-r0"),
                ("diegesis", "d2-r0"),
                ("mvkubric", "m1-r0"),
            ],
        )
        self.assertEqual((loaded, failed), (5, 1))
        self.assertEqual(cursors, {"diegesis": 3, "mvkubric": 2})

        saved_cursors = dict(cursors)
        uninterrupted, _, _, _ = self._invoke(
            uninterrupted_iters, cursors, [True] * 4
        )
        resumed, _, _, _ = self._invoke(
            self._iterators(rank=0, start=saved_cursors),
            saved_cursors,
            [True] * 4,
        )
        self.assertEqual(resumed, uninterrupted)


if __name__ == "__main__":
    unittest.main()
