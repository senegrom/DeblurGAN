"""Zero-retrain trunk-channel pruning probe (see SPEED.md for results).

1. Ranks the 256 trunk (residual-stream) channels by activation importance on
   the example images.
2. Slices the checkpoint to keep the top {224,192,160,128} channels everywhere
   the stream is touched (down-conv out, block convs in+out, up-conv in).
3. Reports output fidelity (PSNR vs the full model, fp32) and fp16 speed,
   plus the stream's channel-correlation spectrum (merge headroom).

Run from the repo root: python experiments/prune_probe.py
"""
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from benchmark_deblur import bench_one, make_input
from deblur_fast import FastGenerator, pad_to_multiple

CKPT = os.path.join(REPO, 'checkpoints', 'experiment_name', 'latest_net_G.pth')
IMAGES = [os.path.join(REPO, 'examples', 'blurry', f)
          for f in ('animation3_blurry.png', 'animation4_blurry.png')]
dev = 'cuda'
torch.backends.cudnn.benchmark = True

STRIP = ('running_mean', 'running_var', 'num_batches_tracked')
sd = {k: v for k, v in
      torch.load(CKPT, map_location='cpu', weights_only=True).items()
      if not k.endswith(STRIP)}


def load_img(p):
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32)
    return (torch.from_numpy(a).permute(2, 0, 1)[None] / 127.5 - 1).to(dev)


def to8(y):
    return (y.float() + 1).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)


def psnr(a, b):
    mse = (a.float() - b.float()).pow(2).mean().item()
    return float('inf') if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))


# --- full model + reference outputs + stream activation stats ---------------
full = FastGenerator()
full.load_state_dict(sd)
full.to(dev).eval().requires_grad_(False)

taps = []             # per-tap (C,) mean-|activation| of the trunk stream
stream_samples = []   # raw mid-trunk tensors for the correlation spectrum

def hook(_m, _i, out):
    taps.append(out.detach().abs().mean(dim=(0, 2, 3)).float())

hooks = [full.model[9].register_forward_hook(hook)]            # trunk entry
hooks += [full.model[i].register_forward_hook(hook) for i in range(10, 19)]

def mid_hook(_m, _i, out):
    stream_samples.append(out.detach().float())

hooks.append(full.model[14].register_forward_hook(mid_hook))

refs = []
with torch.inference_mode():
    for p in IMAGES:
        x, h, w = pad_to_multiple(load_img(p))
        refs.append(to8(full(x)[:, :, :h, :w]))
importance = torch.stack(taps).mean(0)
for h_ in hooks:
    h_.remove()

# --- correlation spectrum of the mid-trunk stream ---------------------------
xs = torch.cat([s.flatten(2).squeeze(0) for s in stream_samples], dim=1)
xs = xs - xs.mean(1, keepdim=True)
xs = xs / (xs.std(1, keepdim=True) + 1e-8)
corr = (xs @ xs.T) / xs.shape[1]
off = corr - torch.eye(256, device=dev)
eig = torch.linalg.eigvalsh(corr).flip(0)
cum = eig.cumsum(0) / eig.sum()
print(f'stream channel correlation: max |rho| = {off.abs().max():.3f}, '
      f'pairs |rho|>0.95: {(off.abs() > 0.95).sum().item() // 2}, '
      f'>0.90: {(off.abs() > 0.90).sum().item() // 2}')
print(f'effective rank: {int((cum < 0.95).sum()) + 1}/256 channels explain 95% '
      f'of stream variance, {int((cum < 0.99).sum()) + 1}/256 explain 99%')
print(f'importance spread: min {importance.min():.4f}, median '
      f'{importance.median():.4f}, max {importance.max():.4f}')

# --- pruned variants --------------------------------------------------------
order = torch.argsort(importance, descending=True).cpu()
print('\n| trunk width | PSNR vs full (img1/img2) | fp16 720p ms | fp16 1080p ms |')
print('|---|---|---|---|')

for keep_n in (256, 224, 192, 160, 128):
    keep = order[:keep_n].sort().values
    sd2 = dict(sd)
    sd2['model.7.weight'] = sd['model.7.weight'][keep]
    sd2['model.7.bias'] = sd['model.7.bias'][keep]
    for i in range(10, 19):
        w = sd[f'model.{i}.conv_block.1.weight']
        sd2[f'model.{i}.conv_block.1.weight'] = w[keep][:, keep]
        sd2[f'model.{i}.conv_block.1.bias'] = sd[f'model.{i}.conv_block.1.bias'][keep]
    sd2['model.19.weight'] = sd['model.19.weight'][keep]

    net = FastGenerator(trunk=keep_n)
    net.load_state_dict(sd2)
    net.to(dev).eval().requires_grad_(False)

    ps = []
    with torch.inference_mode():
        for p, ref in zip(IMAGES, refs):
            x, h, w_ = pad_to_multiple(load_img(p))
            ps.append(psnr(to8(net(x)[:, :, :h, :w_]), ref))

    net_h = net.half()
    times = []
    for (W, H) in ((1280, 720), (1920, 1080)):
        x = make_input(W, H, device=dev).half()
        ms, _, _ = bench_one(net_h, x, iters=15)
        times.append(ms)
    print(f'| {keep_n} | {ps[0]:.1f} / {ps[1]:.1f} dB | {times[0]:.1f} | '
          f'{times[1]:.1f} |', flush=True)
    del net, net_h
    torch.cuda.empty_cache()
