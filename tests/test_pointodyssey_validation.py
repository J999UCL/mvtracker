import unittest

from mvtracker.evaluation.evaluator_3dpt import _evaluation_setting_for_dataset


class PointOdysseyValidationTests(unittest.TestCase):
    def test_pointodyssey_uses_the_existing_kubric_multiview_metric_policy(self):
        self.assertEqual(
            _evaluation_setting_for_dataset(
                "pointodyssey-multiview-validation",
                no_tracking_labels=False,
            ),
            "kubric-multiview",
        )

    def test_existing_dataset_mappings_are_unchanged(self):
        expected = {
            "kubric-multiview-v3": "kubric-multiview",
            "panoptic-multiview": "panoptic-multiview",
            "dex-ycb-multiview": "dexycb-multiview",
            "tapvid2d-davis-first": "tapvid2d",
        }
        for dataset_name, evaluation_setting in expected.items():
            with self.subTest(dataset_name=dataset_name):
                self.assertEqual(
                    _evaluation_setting_for_dataset(
                        dataset_name,
                        no_tracking_labels=False,
                    ),
                    evaluation_setting,
                )

    def test_unlabelled_generic_data_retains_its_existing_policy(self):
        self.assertEqual(
            _evaluation_setting_for_dataset("generic", no_tracking_labels=True),
            "no-tracking-labels",
        )


if __name__ == "__main__":
    unittest.main()
