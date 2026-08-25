"""Benchmark DeblurGAN generator inference configurations.

Measures ms/image and PSNR-vs-FP32 for precision / memory-format / grayscale /
compile variants at several resolutions, using the real trained checkpoint.

  python benchmark_deblur.py                 # standard matrix
  python benchmark_deblur.py --compile       # also try torch.compile (Triton)
  python benchmark_deblur.py --iters 50
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F

from deblur_fast import FastGenerator, load_generator, pad_to_multiple, LUMA

SIZES = [(256, 256), (1280, 720), (1920, 1080)]


def make_input(w, h, gray=False, device='cuda'):
    """Deterministic natural-ish test image in [-1,1]: upsampled low-freq noise."""
    g = torch.Generator(device='cpu').manual_seed(1234)
    base = torch.rand((1, 3, h // 16 + 1, w // 16 + 1), generator=g) * 2 - 1
    img = F.interpolate(base, size=(h, w), mode='bicubic',
                        align_corners=False).clamp_(-1, 1)
    if gray:
        lw = torch.tensor(LUMA).view(1, 3, 1, 1)
        img = (img * lw).sum(dim=1, keepdim=True)
    return img.to(device)


def to_uint8(y):
    return (y.float() + 1).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)


def psnr_uint8(a, b):
    mse = (a.float() - b.float()).pow(2).mean().item()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def bench_one(net, x, iters, warmup=5):
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(warmup):
            y = net(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            y = net(x)
        end.record()
        torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return ms, peak_gb, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint',
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'checkpoints', 'experiment_name',
                                         'latest_net_G.pth'))
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--compile', action='store_true')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU required for this benchmark')
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    print(f'GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}\n')

    configs = [
        # name,                precision, channels_last, gray, compile
        ('fp32 (baseline)',    'fp32', False, False, False),
        ('fp32 + ch-last',     'fp32', True,  False, False),
        ('fp16',               'fp16', False, False, False),
        ('fp16 + ch-last',     'fp16', True,  False, False),
        ('bf16 + ch-last',     'bf16', True,  False, False),
        ('fp16 + ch-last GRAY', 'fp16', True,  True,  False),
    ]
    if args.compile:
        configs.append(('fp16 + ch-last + compile', 'fp16', True, False, True))

    results = {}   # (config, size) -> (ms, peak, psnr)
    refs = {}      # size -> fp32 uint8 output

    for name, prec, cl, gray, comp in configs:
        net, dtype = load_generator(args.checkpoint, device, precision=prec,
                                    gray=gray, residual=True, channels_last=cl,
                                    compile_model=comp)
        for (w, h) in SIZES:
            x = make_input(w, h, gray=gray, device=device).to(dtype=dtype)
            if cl:
                x = x.to(memory_format=torch.channels_last)
            x, _, _ = pad_to_multiple(x, 4)
            try:
                ms, peak, y = bench_one(net, x, args.iters)
            except torch.cuda.OutOfMemoryError:
                results[(name, (w, h))] = None
                torch.cuda.empty_cache()
                continue
            if gray:
                lw = torch.tensor(LUMA, device=device, dtype=y.dtype).view(1, 3, 1, 1)
                y = (y * lw).sum(dim=1, keepdim=True).expand(-1, 3, -1, -1)
            out8 = to_uint8(y)
            key = (w, h)
            if name == 'fp32 (baseline)':
                refs[key] = out8
            if gray:
                # compare against the fp32 RGB pipeline's luma
                ref = refs[key].float()
                lw = torch.tensor(LUMA, device=device).view(1, 3, 1, 1)
                ref = (ref * lw).sum(1, keepdim=True).round().expand(-1, 3, -1, -1)
                p = psnr_uint8(out8.float(), ref)
            else:
                p = psnr_uint8(out8, refs[key])
            results[(name, key)] = (ms, peak, p)
        del net
        torch.cuda.empty_cache()
        print(f'measured: {name}')

    print('\n| config | ' + ' | '.join(f'{w}x{h} ms (img/s)' for w, h in SIZES) +
          ' | PSNR vs fp32 @1080p | peak VRAM @1080p |')
    print('|---' * (len(SIZES) + 3) + '|')
    for name, *_ in configs:
        cells = []
        for s in SIZES:
            r = results.get((name, s))
            cells.append('OOM' if r is None else f'{r[0]:.1f} ({1000 / r[0]:.1f})')
        r1080 = results.get((name, SIZES[-1]))
        psnr = ('-' if r1080 is None else
                ('ref' if name == 'fp32 (baseline)' else f'{r1080[2]:.1f} dB'))
        vram = '-' if r1080 is None else f'{r1080[1]:.2f} GB'
        print(f'| {name} | ' + ' | '.join(cells) + f' | {psnr} | {vram} |')


if __name__ == '__main__':
    main()
