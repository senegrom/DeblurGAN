"""Fast DeblurGAN inference.

Standalone, modern-PyTorch replacement for the test.py pipeline. Loads the
generator weights trained by this repo (resnet_9blocks, instance norm) and runs
them at full image resolution with:

  * FP16 (default on CUDA) / BF16 / FP32 precision
  * cuDNN autotune, inference_mode
  * multi-worker prefetching + pinned memory + async H2D copies
  * threaded image encoding so PNG/JPEG writes overlap GPU compute
  * size-bucketed batching (same-size images run as one batch)
  * optional --gray mode (1-channel input, folded first conv, luma output)
  * optional --compile (torch.compile; needs Triton, falls back gracefully)

Unlike test.py, this processes images at native resolution (padded to a
multiple of 4) instead of random 256x256 crops, and it runs dropout in eval
mode (test.py accidentally leaves dropout active at inference, which makes
outputs stochastic and noisy).

Usage:
  python deblur_fast.py --input path/to/blurry_dir --output path/to/out
  python deblur_fast.py --input img.jpg --output out/ --precision fp32
  python deblur_fast.py --input dir/ --output out/ --gray --batch 4
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.webp', '.tif', '.tiff'}

# ITU-R 601-2 luma, same coefficients PIL uses for convert('L')
LUMA = (0.299, 0.587, 0.114)


# ---------------------------------------------------------------------------
# Model: exact same architecture/state-dict layout as networks.ResnetGenerator
# (including its one-conv-per-ResnetBlock quirk), minus training-only pieces.
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Matches the trained checkpoint: pad -> conv -> IN -> ReLU, residual add.

    (networks.py's ResnetBlock builds only one conv per block due to a ternary
    precedence bug; the shipped checkpoint is trained that way, so we replicate
    it. The Dropout that followed ReLU is parameterless and inference-off.)
    """

    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=True),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
            nn.ReLU(True),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class FastGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=9, residual=True):
        super().__init__()
        self.residual = residual

        def norm(c):
            return nn.InstanceNorm2d(c, affine=False, track_running_stats=False)

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, kernel_size=7, bias=True),
            norm(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=True),
            norm(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=True),
            norm(ngf * 4),
            nn.ReLU(True),
        ]
        layers += [ResBlock(ngf * 4) for _ in range(n_blocks)]
        layers += [
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            norm(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            norm(ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, kernel_size=7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        y = self.model(x)
        if self.residual:
            # 1-ch gray input broadcasts against 3-ch output, which is exactly
            # the RGB net fed with a replicated-gray image.
            y = torch.clamp(x + y, min=-1, max=1)
        return y


def load_generator(checkpoint, device, precision='fp16', gray=False,
                   residual=True, channels_last=False, compile_model=False):
    sd = torch.load(checkpoint, map_location='cpu', weights_only=True)
    # Strip DataParallel prefix and InstanceNorm running stats. The original
    # pipeline never called eval(), so it always normalized with per-instance
    # statistics; dropping the running stats reproduces that (deterministically).
    sd = {k.removeprefix('module.'): v for k, v in sd.items()
          if not k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}

    if gray:
        # Fold RGB->gray into the first conv: for a replicated-gray input,
        # sum(W_r + W_g + W_b) * g is exact.
        w = sd['model.1.weight']
        sd = dict(sd)
        sd['model.1.weight'] = w.sum(dim=1, keepdim=True)

    net = FastGenerator(in_ch=1 if gray else 3, residual=residual)
    net.load_state_dict(sd, strict=True)

    net.eval().requires_grad_(False)
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16,
             'fp32': torch.float32}[precision]
    net = net.to(device=device, dtype=dtype)
    if channels_last:
        net = net.to(memory_format=torch.channels_last)
    if compile_model:
        try:
            net = torch.compile(net, mode='max-autotune', dynamic=False)
        except Exception as e:  # e.g. no Triton on Windows
            print(f'[warn] torch.compile unavailable ({e}); running eager')
    return net, dtype


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def list_images(path):
    if os.path.isfile(path):
        return [path]
    files = []
    for root, _, names in sorted(os.walk(path)):
        for n in sorted(names):
            if os.path.splitext(n)[1].lower() in IMG_EXTENSIONS:
                files.append(os.path.join(root, n))
    if not files:
        sys.exit(f'no images found under {path}')
    return files


def bucket_by_size(paths, batch):
    """Group paths into batches of identical (W,H). PIL reads only the header."""
    by_size = {}
    for p in paths:
        try:
            with Image.open(p) as im:
                by_size.setdefault(im.size, []).append(p)
        except Exception as e:
            print(f'[warn] skipping unreadable {p}: {e}')
    chunks = []
    for _, group in sorted(by_size.items()):
        for i in range(0, len(group), batch):
            chunks.append(group[i:i + batch])
    return chunks


class ChunkDataset(torch.utils.data.Dataset):
    """Each item is one same-size batch, already stacked and normalized."""

    def __init__(self, chunks, gray):
        self.chunks = chunks
        self.gray = gray

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        paths = self.chunks[i]
        arrs = []
        for p in paths:
            img = Image.open(p).convert('L' if self.gray else 'RGB')
            a = np.asarray(img, dtype=np.float32)
            if a.ndim == 2:
                a = a[:, :, None]
            arrs.append(a)
        x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
        x = x.div_(127.5).sub_(1.0)  # [0,255] -> [-1,1]
        return x, list(paths)


def collate_one(items):
    return items[0]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def pad_to_multiple(x, m=4):
    h, w = x.shape[-2:]
    ph, pw = (m - h % m) % m, (m - w % m) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    return x, h, w


def to_uint8_images(y, gray_out):
    """[-1,1] float batch -> list of HxWx{3|1} uint8 arrays (on CPU)."""
    if gray_out:
        lw = torch.tensor(LUMA, device=y.device, dtype=y.dtype).view(1, 3, 1, 1)
        y = (y * lw).sum(dim=1, keepdim=True)
    y = (y.float() + 1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
    return list(y.permute(0, 2, 3, 1).cpu().numpy())


def save_image(arr, path, quality):
    if arr.shape[2] == 1:
        img = Image.fromarray(arr[:, :, 0], 'L')
    else:
        img = Image.fromarray(arr)
    if os.path.splitext(path)[1].lower() in ('.jpg', '.jpeg'):
        img.save(path, quality=quality)
    else:
        img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='image file or directory')
    ap.add_argument('--output', required=True, help='output directory')
    ap.add_argument('--checkpoint',
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'checkpoints', 'experiment_name',
                                         'latest_net_G.pth'))
    ap.add_argument('--precision', choices=['fp16', 'bf16', 'fp32'], default=None,
                    help='default: fp16 on CUDA, fp32 on CPU')
    ap.add_argument('--gray', action='store_true',
                    help='grayscale in/out (folds the first conv; output is luma)')
    ap.add_argument('--no-residual', action='store_true',
                    help='for checkpoints trained without --learn_residual')
    ap.add_argument('--batch', type=int, default=1,
                    help='batch size (same-size images are bucketed together)')
    ap.add_argument('--workers', type=int, default=4, help='data-loading workers')
    ap.add_argument('--compile', action='store_true',
                    help='torch.compile the generator (needs Triton; compiles '
                         'once per distinct image size, so best for many '
                         'same-size images)')
    ap.add_argument('--channels-last', action='store_true',
                    help='NHWC memory format (measured slower on this net; '
                         'mainly useful to cut VRAM at very high resolutions)')
    ap.add_argument('--ext', default=None,
                    help="output extension, e.g. png or jpg (default: keep source's)")
    ap.add_argument('--jpeg-quality', type=int, default=95)
    ap.add_argument('--suffix', default='', help='appended to output file stem')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.precision is None:
        args.precision = 'fp16' if device.type == 'cuda' else 'fp32'
    if device.type == 'cpu' and args.precision == 'fp16':
        print('[warn] fp16 on CPU is slow; consider --precision fp32')
    torch.backends.cudnn.benchmark = True

    net, dtype = load_generator(
        args.checkpoint, device, precision=args.precision, gray=args.gray,
        residual=not args.no_residual,
        channels_last=args.channels_last, compile_model=args.compile)

    paths = list_images(args.input)
    chunks = bucket_by_size(paths, args.batch)
    os.makedirs(args.output, exist_ok=True)

    ds = ChunkDataset(chunks, gray=args.gray)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=collate_one,
        num_workers=min(args.workers, len(chunks)),
        pin_memory=(device.type == 'cuda'),
        persistent_workers=False)

    n_done = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        futures = []
        for x, batch_paths in loader:
            x = x.to(device, dtype=dtype, non_blocking=True)
            if args.channels_last:
                x = x.to(memory_format=torch.channels_last)
            x, h, w = pad_to_multiple(x, 4)
            y = net(x)[:, :, :h, :w]
            for arr, src in zip(to_uint8_images(y, args.gray), batch_paths):
                stem, src_ext = os.path.splitext(os.path.basename(src))
                ext = ('.' + args.ext.lstrip('.')) if args.ext else src_ext
                dst = os.path.join(args.output, stem + args.suffix + ext)
                futures.append(pool.submit(save_image, arr, dst, args.jpeg_quality))
            n_done += len(batch_paths)
            print(f'\r{n_done}/{len(paths)} images', end='', flush=True)
        for f in futures:
            f.result()
    dt = time.perf_counter() - t0
    print(f'\ndone: {n_done} images in {dt:.2f}s ({n_done / dt:.2f} img/s) '
          f'[{device.type}, {args.precision}'
          f'{", gray" if args.gray else ""}{", compiled" if args.compile else ""}]')


if __name__ == '__main__':
    main()
