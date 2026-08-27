"""Apply cache-warming throughput fixes to the pinned TAPVidMV checkout."""

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
        "import io\nfrom dataclasses import dataclass",
        "import io\nimport json\nimport time\nfrom dataclasses import dataclass",
    )
    replace_once(
        dataloading,
        "def load_sequence(\n",
        '''def _decode_resize_jpegs_cuda(
        encoded: np.ndarray,
        *,
        sequence_name: str,
        view_id: int,
        square_crop: bool,
        resolution: int | None,
        chunk_size: int = 32,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Decode bounded batches with nvJPEG, then run the exact endpoint resize."""
    import torch
    from torchvision.io import ImageReadMode, decode_jpeg

    started = time.perf_counter()
    stream = torch.cuda.Stream()
    output = None
    source_hw = None
    target_hw = None
    cv2.setNumThreads(1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as resize_pool:
        for start in range(0, len(encoded), chunk_size):
            cpu_bytes = []
            for raw in encoded[start:start + chunk_size]:
                array = raw if isinstance(raw, np.ndarray) else np.frombuffer(raw, dtype=np.uint8)
                cpu_bytes.append(torch.from_numpy(np.asarray(array, dtype=np.uint8).reshape(-1)))
            with torch.cuda.stream(stream):
                decoded = decode_jpeg(cpu_bytes, mode=ImageReadMode.RGB, device="cuda")
                shapes = {tuple(image.shape[1:]) for image in decoded}
                assert len(shapes) == 1
                chunk_source_hw = next(iter(shapes))
                if source_hw is None:
                    source_hw = chunk_source_hw
                assert source_hw == chunk_source_hw
                batch = torch.stack(decoded)
                height, width = source_hw
                if square_crop and width > height:
                    left = (width - height) // 2
                    batch = batch[:, :, :, left:left + height]
                processed_hw = tuple(batch.shape[-2:])
                target_hw = resolution_hw(processed_hw, resolution) if resolution is not None else processed_hw
                batch_np = batch.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            if processed_hw != target_hw:
                frames = list(resize_pool.map(
                    lambda frame: endpoint_resize_image(frame, target_hw), batch_np))
                batch_np = np.stack(frames).astype(np.uint8, copy=False)
            if output is None:
                output = np.empty((len(encoded), *batch_np.shape[1:]), dtype=np.uint8)
            output[start:start + len(batch_np)] = batch_np
    stream.synchronize()
    assert output is not None and source_hw is not None and target_hw is not None
    print(json.dumps({
        "event": "image_conversion_completed",
        "sequence": sequence_name,
        "view": view_id,
        "frames": len(encoded),
        "source_hw": source_hw,
        "output_hw": output.shape[1:3],
        "seconds": time.perf_counter() - started,
        "backend": "torchvision_nvjpeg_cuda_endpoint_cpu",
    }), flush=True)
    return output, source_hw


def load_sequence(
''',
    )
    replace_once(
        dataloading,
        '''        require_depth: bool = False,
        resolution: int | None = 512,
) -> SequenceData:
''',
        '''        require_depth: bool = False,
        load_depth: bool = True,
        resolution: int | None = 512,
        image_backend: str = "cpu",
) -> SequenceData:
''',
    )
    replace_once(
        dataloading,
        '''    sequence_dir = Path(sequence_dir)
    tracks_xyz = np.load(sequence_dir / "tracks_xyz.npy")
''',
        '''    sequence_started = time.perf_counter()
    sequence_dir = Path(sequence_dir)
    assert image_backend in ("cpu", "cuda")
    assert load_depth or not require_depth
    tracks_xyz = np.load(sequence_dir / "tracks_xyz.npy")
''',
    )
    replace_once(
        dataloading,
        '''        depth = None
        if depth_path.exists():
''',
        '''        depth = None
        if load_depth and depth_path.exists():
''',
    )
    replace_once(
        dataloading,
        '''        # Decode frame by frame into one array instead of stacking a list of
        # frames, which would hold the video twice.
        first = decode_jpeg(images_jpeg_bytes[0])
        images = np.empty((num_frames, *first.shape), dtype=np.uint8)
        images[0] = first
        for frame_idx in range(1, num_frames):
            images[frame_idx] = decode_jpeg(images_jpeg_bytes[frame_idx])
        extrinsics_w2c = np.load(view_dir / "extrinsics_w2c.npy")
''',
        '''        if image_backend == "cuda":
            images, image_source_hw = _decode_resize_jpegs_cuda(
                images_jpeg_bytes,
                sequence_name=sequence_dir.name,
                view_id=int(view_dir.name),
                square_crop=square_protocol,
                resolution=resolution,
            )
        else:
            # Decode frame by frame into one array instead of stacking a list of
            # frames, which would hold the video twice.
            first = decode_jpeg(images_jpeg_bytes[0])
            images = np.empty((num_frames, *first.shape), dtype=np.uint8)
            images[0] = first
            for frame_idx in range(1, num_frames):
                images[frame_idx] = decode_jpeg(images_jpeg_bytes[frame_idx])
            image_source_hw = images.shape[1:3]
        extrinsics_w2c = np.load(view_dir / "extrinsics_w2c.npy")
''',
    )
    replace_once(
        dataloading,
        '''        view_id = int(view_dir.name)
        if square_protocol and images.shape[2] > images.shape[1]:
            images, depth, foreground_mask, intrinsics, _ = _center_square_crop_view(
                images, depth, foreground_mask, intrinsics, queries_xytv, query_v, view_id)

        # Resize inside the loop so each view's native arrays free before the
        # next view decodes; a 4K sequence otherwise peaks near 100 GiB.
        if resolution is not None:
            target_hw = resolution_hw(images.shape[1:3], resolution)
            images, depth, foreground_mask, intrinsics = _resize_view_to_resolution(
                images, depth, foreground_mask, intrinsics,
                queries_xytv, query_v, view_id, target_hw)
''',
        '''        view_id = int(view_dir.name)
        metadata_hw = image_source_hw
        if square_protocol and metadata_hw[1] > metadata_hw[0]:
            if image_backend == "cuda":
                height, width = metadata_hw
                left = (width - height) // 2
                if depth is not None:
                    depth = depth[:, :, left:left + height]
                if foreground_mask is not None:
                    foreground_mask = foreground_mask[:, :, left:left + height]
                intrinsics = intrinsics.astype(np.float32, copy=True)
                intrinsics[2] -= left
                queries_xytv[query_v == view_id, 0] -= left
                metadata_hw = (height, height)
            else:
                images, depth, foreground_mask, intrinsics, _ = _center_square_crop_view(
                    images, depth, foreground_mask, intrinsics, queries_xytv, query_v, view_id)
                metadata_hw = images.shape[1:3]

        # Resize inside the loop so each view's native arrays free before the
        # next view decodes; a 4K sequence otherwise peaks near 100 GiB.
        if resolution is not None:
            target_hw = resolution_hw(metadata_hw, resolution)
            if image_backend == "cuda":
                row_index = endpoint_nearest_indices(metadata_hw[0], target_hw[0])
                column_index = endpoint_nearest_indices(metadata_hw[1], target_hw[1])
                if depth is not None:
                    depth = depth[:, row_index[:, None], column_index[None, :]]
                if foreground_mask is not None:
                    foreground_mask = foreground_mask[:, row_index[:, None], column_index[None, :]]
                scale_x = (target_hw[1] - 1) / (metadata_hw[1] - 1)
                scale_y = (target_hw[0] - 1) / (metadata_hw[0] - 1)
                intrinsics = intrinsics.astype(np.float32, copy=True)
                intrinsics *= np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
                queries_xytv[query_v == view_id, 0] *= scale_x
                queries_xytv[query_v == view_id, 1] *= scale_y
                assert images.shape[1:3] == target_hw
            else:
                images, depth, foreground_mask, intrinsics = _resize_view_to_resolution(
                    images, depth, foreground_mask, intrinsics,
                    queries_xytv, query_v, view_id, target_hw)
''',
    )
    replace_once(
        dataloading,
        '''    sequence = SequenceData(
        root=sequence_dir,
        tracks_xyz=tracks_xyz,
        queries_xytv=queries_xytv,
        views=views,
    )
    return sequence
''',
        '''    sequence = SequenceData(
        root=sequence_dir,
        tracks_xyz=tracks_xyz,
        queries_xytv=queries_xytv,
        views=views,
    )
    print(json.dumps({
        "event": "sequence_load_completed",
        "source": sequence_dir.parent.name,
        "sequence": sequence_dir.name,
        "frames": sequence.num_frames,
        "views": sequence.num_views,
        "seconds": time.perf_counter() - sequence_started,
        "image_backend": image_backend,
        "loaded_depth": load_depth,
    }), flush=True)
    return sequence
''',
    )
    replace_once(
        dataloading,
        '''        require_depth: bool = False,
        resolution: int | None = 512,
):
    """Yield loaded sequences while loading one sequence ahead on a background thread."""
''',
        '''        require_depth: bool = False,
        load_depth: bool = True,
        resolution: int | None = 512,
        image_backend: str = "cpu",
):
    """Yield loaded sequences while loading one sequence ahead on a background thread."""
''',
    )
    replace_once(
        dataloading,
        '''        future = loader.submit(load_sequence, sequence_dirs[0],
                               require_depth=require_depth, resolution=resolution)
        for idx in range(len(sequence_dirs)):
            next_future = None
            if idx + 1 < len(sequence_dirs):
                next_future = loader.submit(load_sequence, sequence_dirs[idx + 1],
                                            require_depth=require_depth, resolution=resolution)
''',
        '''        future = loader.submit(
            load_sequence, sequence_dirs[0], require_depth=require_depth,
            load_depth=load_depth, resolution=resolution, image_backend=image_backend)
        for idx in range(len(sequence_dirs)):
            next_future = None
            if idx + 1 < len(sequence_dirs):
                next_future = loader.submit(
                    load_sequence, sequence_dirs[idx + 1], require_depth=require_depth,
                    load_depth=load_depth, resolution=resolution, image_backend=image_backend)
''',
    )

    runner = root / "run_predictor.py"
    replace_once(runner, "import argparse\n", "import argparse\nimport json\nimport time\n")
    replace_once(
        runner,
        '    parser.add_argument("--seed", type=int, default=72)\n',
        '    parser.add_argument("--seed", type=int, default=72)\n'
        '    parser.add_argument("--skip_optional_depth", action="store_true")\n'
        '    parser.add_argument("--image_backend", choices=("cpu", "cuda"), default="cpu")\n',
    )
    replace_once(
        runner,
        '    print(f"Prediction root: {args.tapvidmv_predictions}", flush=True)\n',
        '    print(f"Prediction root: {args.tapvidmv_predictions}", flush=True)\n'
        '    print(f"Image backend: {args.image_backend}", flush=True)\n'
        '    print(f"Load optional depth: {not args.skip_optional_depth}", flush=True)\n'
        '    assert not args.skip_optional_depth or not predictor.requires_depth\n',
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
            load_depth=not args.skip_optional_depth,
            resolution=dataloading.parse_resolution(args.resolution),
            image_backend=args.image_backend,
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
            save_seconds = time.perf_counter() - save_started
            tqdm.tqdm.write(json.dumps({
                "event": "sequence_prediction_completed",
                "source": data_source,
                "sequence": sequence_name,
                "load_wait_seconds": load_wait_seconds,
                "inference_seconds": inference_seconds,
                "save_seconds": save_seconds,
            }))
''',
    )
    replace_once(
        runner,
        '''                                "seed": args.seed,
                                "data_source": data_source,
''',
        '''                                "seed": args.seed,
                                "image_backend": args.image_backend,
                                "load_optional_depth": not args.skip_optional_depth,
                                "data_source": data_source,
''',
    )

    omega = root / "predictors" / "reconstruction_sources" / "vggt_omega.py"
    release = '''        _load_vggt_omega_model.cache_clear()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
'''
    replace_once(
        omega,
        release,
        '''        if os.environ.get("TAPVIDMV_KEEP_VGGT_MODEL") != "1":
            _load_vggt_omega_model.cache_clear()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
''',
    )
    release = '''    _load_vggt_omega_model.cache_clear()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
'''
    text = omega.read_text()
    assert text.count(release) == 2
    omega.write_text(text.replace(
        release,
        '''    if os.environ.get("TAPVIDMV_KEEP_VGGT_MODEL") != "1":
        _load_vggt_omega_model.cache_clear()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
''',
    ))
    print(f"patched {omega}", flush=True)
    print("TAPVidMV cache-warming performance patch complete", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
