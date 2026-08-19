import ast
import unittest
from pathlib import Path

import torch


TRAIN_PATH = Path(__file__).resolve().parents[1] / "mvtracker" / "cli" / "train.py"
TRAIN_TREE = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"), filename=str(TRAIN_PATH))


def _load_visibility_check():
    function = next(
        node
        for node in TRAIN_TREE.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_assert_real_tracks_visible"
    )
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module([function], [])), str(TRAIN_PATH), "exec"), namespace)
    return namespace[function.name]


_assert_real_tracks_visible = _load_visibility_check()


class TrainingVisibilityTests(unittest.TestCase):
    def test_padding_slots_are_exempt(self):
        visible = torch.tensor([[[True, False], [False, False]]])
        padding = torch.tensor([[False, True]])
        _assert_real_tracks_visible(visible, padding)

    def test_real_invisible_track_still_fails(self):
        visible = torch.tensor([[[True, False], [False, False]]])
        padding = torch.tensor([[False, False]])
        with self.assertRaisesRegex(AssertionError, "real points"):
            _assert_real_tracks_visible(visible, padding)


if __name__ == "__main__":
    unittest.main()
