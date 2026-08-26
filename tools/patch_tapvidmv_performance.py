"""Apply the local throughput fixes to the pinned TAPVidMV checkout."""

from pathlib import Path
import sys


def replace_once(path: Path, before: str, after: str) -> None:
    text = path.read_text()
    assert text.count(before) == 1, f"expected one patch target in {path}"
    path.write_text(text.replace(before, after, 1))
    print(f"patched {path}", flush=True)


def main(root: Path) -> None:
    dataloading = root / "dataloading.py"
    replace_once(
        dataloading,
        '''def load_sequences_prefetched(
        sequence_dirs: list[Path],
        *,
        require_depth: bool = False,
        resolution: int | None = 512,
):
    """Yield loaded sequences while loading one sequence ahead on a background thread."""
    sequence_dirs = [Path(sequence_dir) for sequence_dir in sequence_dirs]
    if not sequence_dirs:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as loader:
        future = loader.submit(load_sequence, sequence_dirs[0],
                               require_depth=require_depth, resolution=resolution)
        for idx in range(len(sequence_dirs)):
            next_future = None
            if idx + 1 < len(sequence_dirs):
                next_future = loader.submit(load_sequence, sequence_dirs[idx + 1],
                                            require_depth=require_depth, resolution=resolution)
            yield future.result()
            future = next_future
''',
        '''def load_sequences_prefetched(
        sequence_dirs: list[Path],
        *,
        require_depth: bool = False,
        resolution: int | None = 512,
        max_workers: int = 4,
):
    """Yield loaded sequences in order while loading a bounded queue in parallel."""
    sequence_dirs = [Path(sequence_dir) for sequence_dir in sequence_dirs]
    if not sequence_dirs:
        return
    worker_count = min(max_workers, len(sequence_dirs))
    assert worker_count > 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as loader:
        futures = {
            idx: loader.submit(
                load_sequence,
                sequence_dirs[idx],
                require_depth=require_depth,
                resolution=resolution,
            )
            for idx in range(worker_count)
        }
        next_submit = worker_count
        for idx in range(len(sequence_dirs)):
            yield futures.pop(idx).result()
            if next_submit < len(sequence_dirs):
                futures[next_submit] = loader.submit(
                    load_sequence,
                    sequence_dirs[next_submit],
                    require_depth=require_depth,
                    resolution=resolution,
                )
                next_submit += 1
''',
    )

    reconstruction = root / "predictors" / "reconstruction_conditioned.py"
    replace_once(
        reconstruction,
        "from __future__ import annotations\n\nfrom pathlib import Path",
        "from __future__ import annotations\n\nfrom functools import lru_cache\nfrom pathlib import Path",
    )
    replace_once(
        reconstruction,
        '''def _build_model(model_name: str, device: str):
    """Instantiate a configured tracker and restore its checkpoint."""
''',
        '''@lru_cache(maxsize=None)
def _build_model(model_name: str, device: str):
    """Instantiate a configured tracker once per process and restore its checkpoint."""
    print(f"loading {model_name} checkpoint on {device}", flush=True)
''',
    )

    runner = root / "run_predictor.py"
    replace_once(runner, "from pathlib import Path\n", "from pathlib import Path\nimport time\n")
    replace_once(
        runner,
        '    parser.add_argument("--seed", type=int, default=72)\n',
        '    parser.add_argument("--seed", type=int, default=72)\n'
        '    parser.add_argument("--loader_workers", type=int, default=4)\n',
    )
    replace_once(
        runner,
        '    print(f"Prediction root: {args.tapvidmv_predictions}", flush=True)\n',
        '    print(f"Prediction root: {args.tapvidmv_predictions}", flush=True)\n'
        '    print(f"Loader workers: {args.loader_workers}", flush=True)\n',
    )
    replace_once(
        runner,
        '''        sequences = dataloading.load_sequences_prefetched(
            [sequence_dir for _sequence_idx, _sequence_name, sequence_dir, _output_dir in sequence_jobs],
            require_depth=predictor.requires_depth,
            resolution=dataloading.parse_resolution(args.resolution),
        )
        for (sequence_idx, sequence_name, _sequence_dir, output_dir), sequence in zip(
                tqdm.tqdm(sequence_jobs, desc=data_source, unit="seq"),
                sequences,
        ):
''',
        '''        sequences = dataloading.load_sequences_prefetched(
            [sequence_dir for _sequence_idx, _sequence_name, sequence_dir, _output_dir in sequence_jobs],
            require_depth=predictor.requires_depth,
            resolution=dataloading.parse_resolution(args.resolution),
            max_workers=args.loader_workers,
        )
        sequence_iterator = iter(sequences)
        for sequence_idx, sequence_name, _sequence_dir, output_dir in tqdm.tqdm(
                sequence_jobs, desc=data_source, unit="seq"):
            load_wait_started = time.perf_counter()
            sequence = next(sequence_iterator)
            load_wait_seconds = time.perf_counter() - load_wait_started
''',
    )
    replace_once(
        runner,
        '''            prediction = predictor.predict_sequence(sequence, debug_logs_path=debug_logs_path)
            save_prediction(output_dir, prediction)
''',
        '''            inference_started = time.perf_counter()
            prediction = predictor.predict_sequence(sequence, debug_logs_path=debug_logs_path)
            inference_seconds = time.perf_counter() - inference_started
            save_started = time.perf_counter()
            save_prediction(output_dir, prediction)
''',
    )
    replace_once(
        runner,
        '''            total_sequences += 1
            tqdm.tqdm.write(f"wrote {output_dir}")
''',
        '''            save_seconds = time.perf_counter() - save_started
            tqdm.tqdm.write(
                f"timing {data_source}/{sequence_name}: "
                f"load_wait={load_wait_seconds:.2f}s "
                f"inference={inference_seconds:.2f}s save={save_seconds:.2f}s"
            )
            total_sequences += 1
            tqdm.tqdm.write(f"wrote {output_dir}")
''',
    )
    print("TAPVidMV performance patch complete", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
