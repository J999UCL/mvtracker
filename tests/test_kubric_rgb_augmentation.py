import ast
import types
import unittest
from pathlib import Path

import numpy as np


def _load_photometric_augmentation_method():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "mvtracker"
        / "datasets"
        / "kubric_multiview_dataset.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    dataset_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KubricMultiViewDataset"
    )
    method = next(
        node
        for node in dataset_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_add_photometric_augs"
    )
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {
        "np": np,
        "Image": types.SimpleNamespace(fromarray=lambda image: image.copy()),
        "F_torchvision": types.SimpleNamespace(
            gaussian_blur=lambda image, _kernel_size, _sigma: image + 1,
        ),
    }
    exec(
        compile(ast.fix_missing_locations(module), str(source_path), "exec"),
        namespace,
    )
    return namespace[method.name]


_add_photometric_augs = _load_photometric_augmentation_method()


class _BlurAugmentation:
    sigma = (0.1, 2.0)
    kernel_size = [11, 11]

    @staticmethod
    def get_params(_sigma_min, _sigma_max):
        return 1.0


class KubricRgbAugmentationTests(unittest.TestCase):
    def test_gaussian_blur_return_value_is_saved(self):
        dataset = types.SimpleNamespace(
            color_aug_prob=0.0,
            blur_aug_prob=1.0,
            blur_aug=_BlurAugmentation(),
        )
        rgbs = np.zeros((2, 3, 4, 5, 3), dtype=np.uint8)
        trajectories = np.zeros((2, 3, 1, 3), dtype=np.float32)
        visibility = np.ones((2, 3, 1), dtype=bool)

        augmented, augmented_visibility = _add_photometric_augs(
            dataset,
            rgbs,
            trajectories,
            visibility,
            np.random.RandomState(0),
            eraser=False,
            replace=False,
        )

        np.testing.assert_array_equal(augmented, np.ones_like(rgbs))
        np.testing.assert_array_equal(augmented_visibility, visibility)


if __name__ == "__main__":
    unittest.main()
