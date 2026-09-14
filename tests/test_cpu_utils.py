"""Small CPU-only regressions; no checkpoints, datasets, training or downloads."""
import math
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from deblur_utils import (
    charbonnier, fft_l1, load_image_tensor, luma,
    psnr8, psnr8_signed, scene_disjoint_split,
)


class CPUUtilsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_luma_primary_colours_and_dtype(self):
        colours = torch.eye(3, dtype=torch.float64).reshape(3, 3, 1, 1)
        actual = luma(colours)
        self.assertEqual(actual.shape, (3, 1, 1, 1))
        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(actual.flatten(), torch.tensor([0.299, 0.587, 0.114], dtype=torch.float64))

    def test_psnr_identity_and_full_contrast(self):
        black = torch.zeros(1, 3, 4, 4)
        white = torch.ones_like(black)
        self.assertTrue(math.isinf(psnr8(black, black)))
        self.assertAlmostEqual(psnr8(black, white), 0.0)
        self.assertAlmostEqual(psnr8_signed(-white, white), 0.0)

    def test_psnr_does_not_modify_inputs(self):
        left = torch.tensor([0.2, 0.5, 0.8])
        right = torch.tensor([0.1, 0.4, 0.9])
        before = left.clone(), right.clone()
        self.assertTrue(math.isfinite(psnr8(left, right)))
        torch.testing.assert_close(left, before[0])
        torch.testing.assert_close(right, before[1])

    def test_charbonnier_identity_and_gradient(self):
        source = torch.ones(1, 1, 4, 4, requires_grad=True)
        self.assertAlmostEqual(charbonnier(source, source).item(), 0.001, places=6)
        loss = charbonnier(source, torch.zeros_like(source))
        loss.backward()
        self.assertTrue(torch.isfinite(source.grad).all().item())
        self.assertGreater(source.grad.min().item(), 0)

    def test_fft_identity_and_difference(self):
        source = torch.zeros(1, 1, 4, 4)
        self.assertEqual(fft_l1(source, source).item(), 0)
        self.assertGreater(fft_l1(source, torch.ones_like(source)).item(), 0)

    def test_whole_scene_holdout(self):
        names = [f'scene{scene}-{frame:03d}.png' for scene in (1, 2) for frame in range(8)]
        train, validation, scene = scene_disjoint_split(names, 3)
        self.assertEqual(scene, 'scene2')
        self.assertEqual(len(validation), 3)
        self.assertEqual(len(train), 8)
        self.assertTrue(all(name.split('-')[0] != scene for name in train))
        self.assertTrue(all(name.split('-')[0] == scene for name in validation))
        self.assertFalse(set(train) & set(validation))

    def test_tiny_image_loading_rgb_and_gray(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'synthetic.png'
            Image.new('RGB', (3, 2), (255, 0, 0)).save(path)
            rgb = load_image_tensor(path, 'cpu')
            gray = load_image_tensor(path, 'cpu', gray=True)
        self.assertEqual(rgb.shape, (1, 3, 2, 3))
        self.assertEqual(gray.shape, (1, 1, 2, 3))
        self.assertEqual(rgb.device.type, 'cpu')
        self.assertEqual(rgb.dtype, torch.float32)
        torch.testing.assert_close(rgb[:, 0], torch.ones(1, 2, 3))
        self.assertEqual(rgb[:, 1:].abs().sum().item(), 0)
        self.assertGreaterEqual(gray.min().item(), 0)
        self.assertLessEqual(gray.max().item(), 1)


if __name__ == '__main__':
    unittest.main()
