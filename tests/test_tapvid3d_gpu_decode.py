import io
import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for nvJPEG")
class TapVid3DGpuDecodeTests(unittest.TestCase):
    def test_decodes_the_whole_microbatch_with_one_nvjpeg_call(self):
        from mvtracker.datasets import tapvid3d_multiview_dataset as loader

        encoded = []
        for value in (20, 40):
            output = io.BytesIO()
            Image.fromarray(np.full((8, 10, 3), value, dtype=np.uint8)).save(output, "JPEG")
            encoded.append(torch.frombuffer(bytearray(output.getvalue()), dtype=torch.uint8))
        theta = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]])
        sample = loader.EncodedTapVid3DSample(
            jpeg_bytes=tuple(encoded),
            depth=torch.ones((1, 2, 1, 8, 10)),
            theta=theta,
            intrs=torch.eye(3).repeat(1, 2, 1, 1),
            extrs=torch.eye(4)[:3].repeat(1, 2, 1, 1),
            trajectory=torch.zeros((1, 2, 1, 3)),
            trajectory_3d=torch.zeros((2, 1, 3)),
            visibility=torch.ones((1, 2, 1), dtype=torch.bool),
            valid=torch.ones((2, 1)),
            query_points_3d=torch.zeros((1, 4)),
            seq_name="fixture",
            metadata={},
            output_size=(8, 10),
            apply_depth_aug=False,
            augmentation_seed=1,
            depth_scale=1.0,
            max_depth=1000.0,
            depth_patch_operations=(),
        )
        batch = loader.EncodedTapVid3DBatch([sample, sample])
        with mock.patch.object(loader, "decode_jpeg", wraps=loader.decode_jpeg) as decoder:
            decoded = loader.decode_tapvid3d_batch(batch, torch.device("cuda"))
        self.assertEqual(decoder.call_count, 1)
        self.assertEqual(len(decoder.call_args.args[0]), 4)
        self.assertEqual(decoded.video.shape, (2, 1, 2, 3, 8, 10))
        self.assertTrue(decoded.video.is_cuda)


if __name__ == "__main__":
    unittest.main()
