import io
import inspect
import os
import shutil
import tarfile
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from mvtracker.datasets import kubric_dali_dataset
from mvtracker.datasets.kubric_dali_dataset import _IndexedRecordStore
from mvtracker.preprocessing.mvkubric_webdataset import build_record_locator


def _write_shard(root: Path, name: str, samples: list[tuple[str, dict[str, bytes]]]):
    tar_path = root / f"{name}.tar"
    with tarfile.open(tar_path, "w", format=tarfile.USTAR_FORMAT) as archive:
        for key, components in samples:
            for extension, payload in components.items():
                info = tarfile.TarInfo(f"{key}.{extension}")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

    bundles: list[list[tuple[str, int, int, str]]] = []
    with tarfile.open(tar_path, "r") as archive:
        last_key = None
        for member in archive:
            key, extension = member.name.split(".", 1)
            if key != last_key:
                bundles.append([])
                last_key = key
            bundles[-1].append((extension, member.offset_data, member.size, member.name))

    index_path = tar_path.with_suffix(".idx")
    lines = [f"v1.2 {len(bundles)}"]
    lines.extend(" ".join(" ".join(map(str, component)) for component in bundle) for bundle in bundles)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "tar": tar_path.name,
        "nsamples": len(samples),
        "bytes": tar_path.stat().st_size,
    }


def _bytes(component) -> bytes:
    if isinstance(component, bytes):
        return component
    if hasattr(component, "getvalue"):
        return component.getvalue()
    return bytes(component)


class KubricIndexedTarLoaderTests(unittest.TestCase):
    def _fixture(self, root: Path):
        first = _write_shard(
            root,
            "mvkubric-00001",
            [
                ("scene-20", {"meta.npz": b"metadata-20"}),
                (
                    "scene-20-view-00",
                    {"rgb.npz": b"rgb-20-0", "depth.npz": b"depth-20-0"},
                ),
            ],
        )
        second = _write_shard(
            root,
            "mvkubric-00000",
            [
                ("scene-10", {"meta.npz": b"metadata-10"}),
                (
                    "scene-10-view-00",
                    {"rgb.npz": b"rgb-10-0", "depth.npz": b"depth-10-0"},
                ),
            ],
        )
        locator = build_record_locator([first, second], root / "record-locator.npz")
        return _IndexedRecordStore(locator)

    def test_reads_exact_components_by_arbitrary_global_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self._fixture(Path(temporary))

            scene_10_view = store.get(3)
            scene_20_meta = store.get(0)
            scene_10_meta = store.get(2)
            scene_20_view = store.get(1)

            self.assertEqual(scene_10_view["__key__"], "scene-10-view-00")
            self.assertEqual(_bytes(scene_10_view[".rgb.npz"]), b"rgb-10-0")
            self.assertEqual(_bytes(scene_10_view[".depth.npz"]), b"depth-10-0")
            self.assertEqual(scene_20_meta["__key__"], "scene-20")
            self.assertEqual(_bytes(scene_20_meta[".meta.npz"]), b"metadata-20")
            self.assertEqual(_bytes(scene_10_meta[".meta.npz"]), b"metadata-10")
            self.assertEqual(_bytes(scene_20_view[".rgb.npz"]), b"rgb-20-0")
            self.assertEqual(_bytes(scene_20_view[".depth.npz"]), b"depth-20-0")

    def test_global_order_follows_catalog_shard_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self._fixture(Path(temporary))
            self.assertEqual(
                [store.get(index)["__key__"] for index in range(4)],
                ["scene-20", "scene-20-view-00", "scene-10", "scene-10-view-00"],
            )

    def test_concurrent_reads_from_shared_shards_are_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self._fixture(Path(temporary))
            indices = [3, 0, 1, 2] * 32

            with ThreadPoolExecutor(max_workers=16) as executor:
                records = list(executor.map(store.get, indices))

            self.assertEqual(
                [record["__key__"] for record in records],
                [
                    ["scene-10-view-00", "scene-20", "scene-20-view-00", "scene-10"][i % 4]
                    for i in range(len(indices))
                ],
            )

    def test_reads_do_not_copy_or_cache_tar_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._fixture(root)
            before = {
                path.relative_to(root): path.stat().st_size
                for path in root.rglob("*")
                if path.is_file()
            }

            with patch.object(shutil, "copyfile", side_effect=AssertionError("whole-TAR copy")), patch.object(
                shutil, "copy2", side_effect=AssertionError("whole-TAR copy")
            ), patch.object(os, "pread", wraps=os.pread) as pread:
                records, stats = store.read_many([3, 0, 1, 2])

            self.assertEqual(
                [record["__key__"] for record in records],
                ["scene-10-view-00", "scene-20", "scene-20-view-00", "scene-10"],
            )
            self.assertEqual(stats.requested_bytes, 58)
            self.assertEqual(stats.read_bytes, 58)
            self.assertEqual(stats.record_count, 4)
            self.assertEqual(sum(call.args[1] for call in pread.call_args_list), 58)
            after = {
                path.relative_to(root): path.stat().st_size
                for path in root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after, before)
            self.assertFalse((root / "_wids_cache").exists())

    def test_live_loader_has_no_wids_or_whole_tar_copy_path(self):
        source = inspect.getsource(kubric_dali_dataset)
        self.assertIn("os.pread", source)
        self.assertIn("_IndexedRecordStore(self.catalog.record_locator_path)", source)
        self.assertNotIn("_WidsRecordStore", source)
        self.assertNotIn("ShardListDataset", source)
        self.assertNotIn("shutil.copyfile", source)


if __name__ == "__main__":
    unittest.main()
