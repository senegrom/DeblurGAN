"""Zero-retrain trunk-channel pruning probe (see SPEED.md for the verdict).

1. Ranks the 256 trunk (residual-stream) channels by activation importance on
   the example images.
2. Slices the checkpoint to keep the top {224,192,160,128} channels everywhere
   the stream is touched (down-conv out, every block conv in+out, up-conv in).
3. Reports fidelity (PSNR vs the full model) and fp16 speed, plus the
   stream's channel-correlation spectrum (merge headroom).

Works with any DeblurGAN generator checkpoint (one- or two-conv blocks).

  python experiments/prune_probe.py [--checkpoint <path>]
"""
import argparse
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from benchmark_deblur import bench_one, make_input
from deblur_fast import DEFAULT_CKPT, FastGenerator, pad_to_multiple
from deblur_utils import load_image_tensor, psnr8_signed

IMAGES = [os.path.join(REPO, 'examples', 'gopro', 'input', f)
          for f in ('GOPR0869_11_00-000034.png', 'GOPR0881_11_01-000210.png')]
STRIP = ('running_mean', 'running_var', 'num_batches_tracked')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT['deblurgan'])
    args = ap.parse_args()
    dev = 'cuda'
    torch.backends.cudnn.benchmark = True

    sd = {k.removeprefix('module.'): v for k, v in
          torch.load(args.checkpoint, map_location='cpu', weights_only=True).items()
          if not k.endswith(STRIP)}
    # block layout: which conv_block indices hold the (1 or 2) convs
    if 'model.10.conv_block.6.weight' in sd:
        layout, conv_idx = 'paper', (1, 6)
    elif 'model.10.conv_block.5.weight' in sd:
        layout, conv_idx = 'paper_nodrop', (1, 5)
    else:
        layout, conv_idx = 'legacy', (1,)
    print(f'checkpoint layout: {layout} (block convs at {conv_idx})')

    full = FastGenerator(block_layout=layout)
    full.load_state_dict(sd)
    full.to(dev).eval().requires_grad_(False)

    taps, stream_samples = [], []
    hooks = [full.model[9].register_forward_hook(
        lambda m, i, o: taps.append(o.detach().abs().mean(dim=(0, 2, 3)).float()))]
    hooks += [full.model[i].register_forward_hook(
        lambda m, i_, o: taps.append(o.detach().abs().mean(dim=(0, 2, 3)).float()))
        for i in range(10, 19)]
    hooks.append(full.model[14].register_forward_hook(
        lambda m, i, o: stream_samples.append(o.detach().float())))

    refs = []
    with torch.inference_mode():
        for p in IMAGES:
            x = load_image_tensor(p, dev) * 2 - 1
            xp, h, w = pad_to_multiple(x, 4)
            refs.append(full(xp)[:, :, :h, :w])
    importance = torch.stack(taps).mean(0)
    for h_ in hooks:
        h_.remove()

    xs = torch.cat([s.flatten(2).squeeze(0) for s in stream_samples], dim=1)
    xs = (xs - xs.mean(1, keepdim=True)) / (xs.std(1, keepdim=True) + 1e-8)
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

    order = torch.argsort(importance, descending=True).cpu()
    print('\n| trunk width | PSNR vs full (img1/img2) | fp16 720p ms | fp16 1080p ms |')
    print('|---|---|---|---|')
    for keep_n in (256, 224, 192, 160, 128):
        keep = order[:keep_n].sort().values
        sd2 = dict(sd)
        sd2['model.7.weight'] = sd['model.7.weight'][keep]
        sd2['model.7.bias'] = sd['model.7.bias'][keep]
        for i in range(10, 19):
            for c in conv_idx:
                k = f'model.{i}.conv_block.{c}'
                sd2[f'{k}.weight'] = sd[f'{k}.weight'][keep][:, keep]
                sd2[f'{k}.bias'] = sd[f'{k}.bias'][keep]
        sd2['model.19.weight'] = sd['model.19.weight'][keep]

        net = FastGenerator(trunk=keep_n, block_layout=layout)
        net.load_state_dict(sd2)
        net.to(dev).eval().requires_grad_(False)
        ps = []
        with torch.inference_mode():
            for p, ref in zip(IMAGES, refs):
                x = load_image_tensor(p, dev) * 2 - 1
                xp, h, w = pad_to_multiple(x, 4)
                ps.append(psnr8_signed(net(xp)[:, :, :h, :w], ref))
        net_h = net.half()
        times = [bench_one(net_h, make_input(W, H, device=dev).half(), iters=15)[0]
                 for (W, H) in ((1280, 720), (1920, 1080))]
        print(f'| {keep_n} | {ps[0]:.1f} / {ps[1]:.1f} dB | {times[0]:.1f} | '
              f'{times[1]:.1f} |', flush=True)
        del net, net_h
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
