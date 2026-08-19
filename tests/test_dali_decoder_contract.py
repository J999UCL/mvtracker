import inspect
import unittest
from pathlib import Path

import torch

from mvtracker.datasets.tapvid3d_multiview_dataset import DaliEncodedImageDecoder


class _TensorOutput:
    def __init__(self, tensor):
        self.tensor = tensor

    def as_tensor(self):
        return self.tensor


class DaliDecoderContractTests(unittest.TestCase):
    def test_gpu_output_conversion_preserves_batch_shape_and_dtype(self):
        decoder = object.__new__(DaliEncodedImageDecoder)
        decoder._device = torch.device("cpu")
        source = torch.arange(24, dtype=torch.uint16).reshape(2, 3, 4)
        result = decoder._as_torch(_TensorOutput(source))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].dtype, torch.uint16)
        torch.testing.assert_close(result[1], source[1])

    def test_decoder_uses_mixed_rgb_and_native_cpu_depth(self):
        source = inspect.getsource(DaliEncodedImageDecoder)
        self.assertIn('fn.external_source', source)
        self.assertIn('device="mixed"', source)
        self.assertIn('device="cpu"', source)
        self.assertIn('output_type=types.ANY_DATA', source)
        self.assertIn('exec_dynamic=False', source)
        self.assertIn('max_encoded_images: int = 288', source)
        self.assertIn('batch_size=max_encoded_images', source)

    def test_modal_dependency_image_uses_one_dali_codec_abi(self):
        source = (Path(__file__).parents[1] / "tools/modal_training_profile.py").read_text()
        self.assertIn('"nvidia-dali-cuda120==1.53.0"', source)
        self.assertIn('"nvidia-nvimgcodec-cu12[nvtiff]==0.7.0.11"', source)
        self.assertIn('"nvidia-libnvcomp-cu12==5.1.0.21"', source)
        self.assertNotIn('"nvidia-nvimgcodec-cu12[nvtiff]==0.9.0.20"', source)
        self.assertNotIn('"nvidia-libnvcomp-cu12==5.3.0.16"', source)


if __name__ == "__main__":
    unittest.main()
