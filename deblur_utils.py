"""Shared helpers for the training and benchmark scripts."""

import io
import math
import os

import numpy as np
import torch
from PIL import Image

# ITU-R 601-2 luma, same coefficients PIL uses for convert('L')
LUMA = (0.299, 0.587, 0.114)


def luma(x):
    """(B,3,H,W) -> (B,1,H,W) luma, any float dtype/device."""
    lw = torch.tensor(LUMA, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * lw).sum(1, keepdim=True)


def psnr8(a, b):
    """PSNR after uint8 quantization for tensors in [0,1]."""
    a = a.float().mul(255).round_().clamp_(0, 255)
    b = b.float().mul(255).round_().clamp_(0, 255)
    mse = (a - b).pow(2).mean().item()
    return float('inf') if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))


def psnr8_signed(a, b):
    """Same, for tensors in [-1,1]."""
    return psnr8((a.float() + 1) / 2, (b.float() + 1) / 2)


def charbonnier(a, b, eps=1e-3):
    return torch.sqrt((a - b) ** 2 + eps * eps).mean()


def fft_l1(a, b):
    fa, fb = torch.fft.rfft2(a), torch.fft.rfft2(b)
    return (torch.view_as_real(fa) - torch.view_as_real(fb)).abs().mean()


def scene_disjoint_split(names, val_count):
    """Hold out the whole last scene (GoPro-style 'SCENE-frame.png' names),
    subsampled to val_count frames, so no validation video leaks into
    training. Returns (train_names, val_names, val_scene)."""
    scenes = sorted({n.split('-')[0] for n in names})
    val_scene = scenes[-1]
    val_names = [n for n in names if n.startswith(val_scene)]
    val_names = val_names[::max(1, len(val_names) // val_count)][:val_count]
    train_names = [n for n in names if not n.startswith(val_scene)]
    return train_names, val_names, val_scene


def load_image_tensor(path, device, gray=False):
    """Image file -> (1,C,H,W) float tensor in [0,1]."""
    a = np.asarray(Image.open(path).convert('L' if gray else 'RGB'), dtype=np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(device) / 255.0


class GoProTestLMDB:
    """BasicSR-style GoPro test set: <root>/input.lmdb + <root>/target.lmdb."""

    def __init__(self, root, limit=0):
        import lmdb  # optional dependency, only for benchmarks
        with open(os.path.join(root, 'input.lmdb', 'meta_info.txt')) as f:
            self.keys = [ln.split()[0].removesuffix('.png') for ln in f if ln.strip()]
        if limit:
            self.keys = self.keys[:limit]
        opts = dict(readonly=True, lock=False, readahead=False, meminit=False)
        self.env_in = lmdb.open(os.path.join(root, 'input.lmdb'), **opts)
        self.env_gt = lmdb.open(os.path.join(root, 'target.lmdb'), **opts)

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def _decode(env, key, device):
        with env.begin(write=False) as txn:
            data = txn.get(key.encode())
        a = np.asarray(Image.open(io.BytesIO(data)).convert('RGB'), dtype=np.float32)
        return torch.from_numpy(a).permute(2, 0, 1)[None].to(device) / 255.0

    def pair(self, key, device):
        """(blurry, sharp) tensors in [0,1]."""
        return (self._decode(self.env_in, key, device),
                self._decode(self.env_gt, key, device))
