from pathlib import Path
import unittest


SOURCE = (Path(__file__).parents[1] / "tools/modal_waymo_rerun.py").read_text()


class ModalWaymoRerunTest(unittest.TestCase):
    def test_cpu_tags_and_resources(self):
        for value in ('"owner": "jeet"', '"project": "mvtracker"', '"purpose": "profiling"'):
            self.assertIn(value, SOURCE)
        self.assertIn('"gpu": "cpu"', SOURCE)
        self.assertNotIn('gpu="', SOURCE)
        self.assertIn("cpu=8", SOURCE)

    def test_pinned_dependencies_and_paths(self):
        self.assertIn('"waymo-open-dataset-tf-2-12-0==1.6.6"', SOURCE)
        self.assertIn('"rerun-sdk==0.21.0"', SOURCE)
        self.assertIn("datasets/waymo-visualization/source", SOURCE)
        self.assertIn("waymo-visualization", SOURCE)


if __name__ == "__main__":
    unittest.main()
