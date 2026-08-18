import unittest

from tools.profile_mixed_physical_loader import parse_device_ids


class MixedPhysicalLoaderProfileTests(unittest.TestCase):
    def test_loader_profile_is_pinned_to_titans_one_and_two(self):
        self.assertEqual(parse_device_ids("1,2"), (1, 2))

    def test_loader_profile_rejects_other_device_pairs(self):
        with self.assertRaisesRegex(ValueError, "physical devices 1,2"):
            parse_device_ids("0,1")


if __name__ == "__main__":
    unittest.main()
