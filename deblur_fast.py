"""Fast deblurring inference (DeblurGAN generator or NAFNet).

Standalone, modern-PyTorch replacement for the test.py pipeline with:

  * FP16 (default on CUDA) / BF16 / FP32 precision
  * cuDNN autotune, inference_mode
  * multi-worker prefetching + pinned memory + async H2D copies
  * threaded image encoding so PNG/JPEG writes overlap GPU compute
  * size-bucketed batching (same-size images run as one batch)
  * optional --gray mode (grayscale in/out)
  * optional --compile (torch.compile; needs Triton, falls back gracefully)

Architectures (--arch):
  deblurgan  this repo's generator. Auto-detects one-conv-per-block legacy
             checkpoints (trained before the ResnetBlock fix) as well as
             paper-correct two-conv checkpoints, with or without dropout.
  nafnet     NAFNet (e.g. NAFNet-GoPro-width64.pth) loaded via `spandrel`
             (pip install spandrel). ~+5 dB over DeblurGAN on GoPro.
  student    slim FastGenerator checkpoints produced by train_student.py
             (self-describing .pth with arch_config). Default weights are the
             NAFNet-distilled student; --arch student-gt selects the
             GT-supervised variant.

Unlike test.py, this processes images at native resolution (padded to the
network's required multiple) instead of random 256x256 crops, and it runs
dropout in eval mode (test.py accidentally leaves dropout active at
inference, which makes outputs stochastic and noisy).

Usage:
  python deblur_fast.py --input blurry_dir --output out_dir
  python deblur_fast.py --arch nafnet --input blurry_dir --output out_dir
  python deblur_fast.py --input dir/ --output out/ --gray --batch 4 --compile
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.webp', '.tif', '.tiff'}

# ITU-R 601-2 luma, same coefficients PIL uses for convert('L')
LUMA = (0.299, 0.587, 0.114)

_REPO = os.path.dirname(os.path.abspath(__file__))


def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[-1]


# deblurgan: prefer the recovered official (two-conv) weights over the
# upstream checkpoint (trained on the one-conv ResnetBlock bug, much weaker).
# student: prefer the NAFNet-distilled variant (29.29 dB) over GT-trained.
DEFAULT_CKPT = {
    'deblurgan': _first_existing(
        os.path.join(_REPO, 'checkpoints', 'official', 'latest_net_G.pth'),
        os.path.join(_REPO, 'checkpoints', 'experiment_name', 'latest_net_G.pth')),
    'nafnet': os.path.join(_REPO, 'checkpoints', 'NAFNet-GoPro-width64.pth'),
    'student': _first_existing(
        os.path.join(_REPO, 'checkpoints', 'student_nafnet', 'student_best.pth'),
        os.path.join(_REPO, 'checkpoints', 'student', 'student_best.pth'),
        os.path.join(_REPO, 'checkpoints', 'student', 'student_latest.pth')),
    # the GT-supervised (non-distilled) student, 29.07 dB vs the distilled 29.29
    'student-gt': _first_existing(
        os.path.join(_REPO, 'checkpoints', 'student', 'student_best.pth'),
        os.path.join(_REPO, 'checkpoints', 'student', 'student_latest.pth')),
}


# ---------------------------------------------------------------------------
# DeblurGAN generator, state-dict compatible with models/networks.py
# ---------------------------------------------------------------------------

def _norm(c):
    return nn.InstanceNorm2d(c, affine=False, track_running_stats=False)


class ResBlock(nn.Module):
    """Residual block matching networks.py's ResnetBlock state-dict layout.

    block_layout:
      'legacy'       pad-conv-IN-ReLU (+residual). Produced by the pre-fix
                     ResnetBlock (an operator-precedence bug collapsed it to a
                     single conv); the shipped experiment_name checkpoint is
                     this variant. Conv key: conv_block.1
      'paper'        pad-conv-IN-ReLU-[drop]-pad-conv-IN, the two-conv block
                     of the paper as built by the fixed ResnetBlock (dropout
                     slot always present). Conv keys: conv_block.1 / .6
      'paper_nodrop' same but without the dropout slot (e.g. checkpoints from
                     other DeblurGAN forks). Conv keys: conv_block.1 / .5
    """

    def __init__(self, dim, block_layout='legacy', dropout=0.0):
        super().__init__()
        first = [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, bias=True),
                 _norm(dim), nn.ReLU(True)]
        if block_layout == 'legacy':
            layers = first
        elif block_layout == 'paper':
            # slot 4 is the paper's dropout position; parameterless either way,
            # so checkpoint keys are identical with or without dropout
            mid = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            layers = first + [mid,
                              nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, bias=True),
                              _norm(dim)]
        elif block_layout == 'paper_nodrop':
            layers = first + [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, bias=True),
                              _norm(dim)]
        else:
            raise ValueError(f'unknown block_layout {block_layout!r}')
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv_block(x)


class FastGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=9, residual=True,
                 trunk=None, block_layout='legacy', dropout=0.0):
        super().__init__()
        self.residual = residual
        trunk = trunk or ngf * 4   # width of the residual-block stream

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, kernel_size=7, bias=True),
            _norm(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=True),
            _norm(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, trunk, kernel_size=3, stride=2, padding=1, bias=True),
            _norm(trunk),
            nn.ReLU(True),
        ]
        layers += [ResBlock(trunk, block_layout, dropout) for _ in range(n_blocks)]
        layers += [
            nn.ConvTranspose2d(trunk, ngf * 2, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            _norm(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            _norm(ngf),
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


def _finalize(net, device, precision, channels_last, compile_model):
    net.eval().requires_grad_(False)
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16,
             'fp32': torch.float32}[precision]
    net = net.to(device=device, dtype=dtype)
    if channels_last:
        net = net.to(memory_format=torch.channels_last)
    if compile_model:
        try:
            net = torch.compile(net, mode='max-autotune', dynamic=False)
        except Exception as e:  # e.g. no Triton
            print(f'[warn] torch.compile unavailable ({e}); running eager')
    return net, dtype


def load_generator(checkpoint, device, precision='fp16', gray=False,
                   residual=True, channels_last=False, compile_model=False):
    """Load a DeblurGAN generator checkpoint (any block layout)."""
    sd = torch.load(checkpoint, map_location='cpu', weights_only=True)
    # Strip DataParallel prefix and InstanceNorm running stats. The original
    # pipeline never called eval(), so it always normalized with per-instance
    # statistics; dropping the running stats reproduces that (deterministically).
    sd = {k.removeprefix('module.'): v for k, v in sd.items()
          if not k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}

    if 'model.10.conv_block.6.weight' in sd:
        block_layout = 'paper'
    elif 'model.10.conv_block.5.weight' in sd:
        block_layout = 'paper_nodrop'
    else:
        block_layout = 'legacy'

    if gray:
        # Fold RGB->gray into the first conv: for a replicated-gray input,
        # sum(W_r + W_g + W_b) * g is exact.
        sd = dict(sd)
        sd['model.1.weight'] = sd['model.1.weight'].sum(dim=1, keepdim=True)

    net = FastGenerator(in_ch=1 if gray else 3, residual=residual,
                        block_layout=block_layout)
    net.load_state_dict(sd, strict=True)
    return _finalize(net, device, precision, channels_last, compile_model)


def load_student(checkpoint, device, precision='fp16', channels_last=False,
                 compile_model=False):
    """Load a distilled slim FastGenerator saved by train_student.py."""
    raw = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = raw['arch_config']
    net = FastGenerator(**cfg)
    net.load_state_dict(raw['state_dict'], strict=True)
    net, dtype = _finalize(net, device, precision, channels_last, compile_model)
    return net, dtype, cfg


class _TLCAvgPool2d(nn.Module):
    """Test-time Local Converter pooling (NAFNetLocal, from megvii's
    local_arch.py). Replaces the global AdaptiveAvgPool2d(1) inside NAFNet's
    channel attention: at test resolutions above the 256x256 training crops,
    global pooling mismatches training statistics; a local window whose extent
    matches base_size (1.5x the train crop) in input pixels restores them.
    The kernel size per layer is fixed by a dry forward at train size.
    """

    def __init__(self, base_size=384, train_size=256):
        super().__init__()
        self.base_size = base_size
        self.train_size = train_size
        self.kernel_size = None

    def forward(self, x):
        if self.kernel_size is None:
            # dry forward at train resolution: feature size * base / train
            self.kernel_size = (x.shape[-2] * self.base_size // self.train_size,
                                x.shape[-1] * self.base_size // self.train_size)
        k1, k2 = self.kernel_size
        h, w = x.shape[-2:]
        if k1 >= h and k2 >= w:
            return F.adaptive_avg_pool2d(x, 1)
        k1, k2 = min(k1, h), min(k2, w)
        s = x.cumsum(-1).cumsum(-2)
        s = F.pad(s, (1, 0, 1, 0))                    # zero row/col for windows
        out = (s[..., k1:, k2:] - s[..., :-k1, k2:]
               - s[..., k1:, :-k2] + s[..., :-k1, :-k2]) / (k1 * k2)
        pt, pl = k1 - 1 - (k1 - 1) // 2, k2 - 1 - (k2 - 1) // 2
        return F.pad(out, (pl, k2 - 1 - pl, pt, k1 - 1 - pt), mode='replicate')


def _apply_tlc(net, device, base_size=384, train_size=256):
    """Swap NAFNet's global pools for TLC pools and fix their kernels."""
    swapped = 0
    for module in net.modules():
        for name, child in module.named_children():
            if isinstance(child, nn.AdaptiveAvgPool2d) and child.output_size in (1, (1, 1)):
                setattr(module, name, _TLCAvgPool2d(base_size, train_size))
                swapped += 1
    if swapped:
        with torch.inference_mode():
            net(torch.zeros(1, 3, train_size, train_size, device=device))
    return swapped


def load_nafnet(checkpoint, device, precision='fp16', channels_last=False,
                compile_model=False, tlc=True):
    """Load a NAFNet checkpoint via spandrel. Input/output range is [0,1].

    NAFNet's LayerNorms are numerically fragile in half precision (full fp16
    cast produces garbage, full bf16 visible artifacts — measured), so fp16 /
    bf16 here mean *autocast* mixed precision: weights stay fp32, convs run on
    tensor cores, norms/reductions stay fp32. Returns (net, autocast_dtype).
    tlc=True applies the official test-time local-pooling conversion
    (NAFNetLocal), which the official GoPro evaluation uses.
    """
    try:
        from spandrel import ModelLoader
    except ImportError:
        sys.exit('--arch nafnet needs spandrel: pip install spandrel')
    desc = ModelLoader().load_from_file(checkpoint)
    net = desc.model.eval().requires_grad_(False).to(device)
    if tlc:
        _apply_tlc(net, device)
    net, _ = _finalize(net, device, 'fp32', channels_last, compile_model)
    ac_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16,
                'fp32': None}[precision]
    mult = getattr(desc.size_requirements, 'multiple_of', 1) or 1
    return net, ac_dtype, max(mult, 16)


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
    """Each item is one same-size batch, stacked, in [0,1]."""

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
        x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).div_(255.0)
        return x, list(paths)


def collate_one(items):
    return items[0]


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def pad_to_multiple(x, m=4):
    h, w = x.shape[-2:]
    ph, pw = (m - h % m) % m, (m - w % m) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    return x, h, w


def to_uint8_images(y, luma_out, lo, hi):
    """[lo,hi] float batch -> list of HxWx{3|1} uint8 arrays (on CPU)."""
    if luma_out and y.shape[1] == 3:
        lw = torch.tensor(LUMA, device=y.device, dtype=y.dtype).view(1, 3, 1, 1)
        y = (y * lw).sum(dim=1, keepdim=True)
    y = (y.float() - lo).mul_(255.0 / (hi - lo)).round_().clamp_(0, 255)
    return list(y.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())


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
    ap.add_argument('--arch',
                    choices=['deblurgan', 'nafnet', 'student', 'student-gt'],
                    default='deblurgan',
                    help='student = NAFNet-distilled slim model (default '
                         'weights); student-gt = the GT-supervised variant')
    ap.add_argument('--checkpoint', default=None,
                    help='weights path (default depends on --arch)')
    ap.add_argument('--precision', choices=['fp16', 'bf16', 'fp32'], default=None,
                    help='default: fp16 on CUDA, fp32 on CPU')
    ap.add_argument('--gray', action='store_true',
                    help='grayscale in/out (deblurgan: folds the first conv; '
                         'nafnet: replicated input, luma output)')
    ap.add_argument('--no-residual', action='store_true',
                    help='deblurgan checkpoints trained without --learn_residual')
    ap.add_argument('--no-tlc', action='store_true',
                    help='nafnet: disable test-time local pooling (NAFNetLocal)')
    ap.add_argument('--batch', type=int, default=1,
                    help='batch size (same-size images are bucketed together)')
    ap.add_argument('--workers', type=int, default=4, help='data-loading workers')
    ap.add_argument('--compile', action='store_true',
                    help='torch.compile the model (needs Triton; compiles once '
                         'per distinct image size, so best for many same-size '
                         'images)')
    ap.add_argument('--channels-last', action='store_true',
                    help='NHWC memory format (measured slower on the DeblurGAN '
                         'net; mainly useful to cut VRAM at very high resolutions)')
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
    checkpoint = args.checkpoint or DEFAULT_CKPT[args.arch]

    # per-arch pipeline config
    expand3 = False           # replicate 1-ch gray to 3-ch before the net
    luma_out = args.gray      # collapse 3-ch output to luma
    ac_dtype = None           # autocast dtype (nafnet); others fully cast
    if args.arch == 'deblurgan':
        net, dtype = load_generator(
            checkpoint, device, precision=args.precision, gray=args.gray,
            residual=not args.no_residual,
            channels_last=args.channels_last, compile_model=args.compile)
        lo, hi, mult = -1.0, 1.0, 4
    elif args.arch == 'nafnet':
        net, ac_dtype, mult = load_nafnet(
            checkpoint, device, precision=args.precision,
            channels_last=args.channels_last, compile_model=args.compile,
            tlc=not args.no_tlc)
        dtype = torch.float32   # model and inputs stay fp32; autocast inside
        lo, hi = 0.0, 1.0
        expand3 = args.gray
    else:  # student
        net, dtype, cfg = load_student(
            checkpoint, device, precision=args.precision,
            channels_last=args.channels_last, compile_model=args.compile)
        lo, hi, mult = -1.0, 1.0, 4
        args.gray = cfg.get('in_ch', 3) == 1   # dataset side follows the arch
        luma_out = args.gray and cfg.get('out_ch', 3) == 3

    paths = list_images(args.input)
    chunks = bucket_by_size(paths, args.batch)
    os.makedirs(args.output, exist_ok=True)

    ds = ChunkDataset(chunks, gray=args.gray)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=collate_one,
        num_workers=min(args.workers, len(chunks)),
        pin_memory=(device.type == 'cuda'),
        persistent_workers=False)

    ac = (torch.autocast(device.type, dtype=ac_dtype) if ac_dtype is not None
          else nullcontext())

    n_done = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        futures = []
        for x, batch_paths in loader:
            x = x.to(device, dtype=dtype, non_blocking=True)
            if lo == -1.0:
                x = x * 2 - 1
            if expand3 and x.shape[1] == 1:
                x = x.expand(-1, 3, -1, -1)
            if args.channels_last:
                x = x.to(memory_format=torch.channels_last)
            x, h, w = pad_to_multiple(x, mult)
            with ac:
                y = net(x)
            y = y[:, :, :h, :w]
            for arr, src in zip(to_uint8_images(y, luma_out, lo, hi), batch_paths):
                stem, src_ext = os.path.splitext(os.path.basename(src))
                ext = ('.' + args.ext.lstrip('.')) if args.ext else src_ext
                dst = os.path.join(args.output, stem + args.suffix + ext)
                futures.append(pool.submit(save_image, arr, dst, args.jpeg_quality))
            n_done += len(batch_paths)
            print(f'\r{n_done}/{len(paths)} images', end='', flush=True)
        for f in futures:
            f.result()
    dt = time.perf_counter() - t0
    prec = (f'autocast-{args.precision}' if ac_dtype is not None
            else args.precision)
    print(f'\ndone: {n_done} images in {dt:.2f}s ({n_done / dt:.2f} img/s) '
          f'[{args.arch}, {device.type}, {prec}'
          f'{", gray" if args.gray else ""}{", compiled" if args.compile else ""}]')


if __name__ == '__main__':
    main()
