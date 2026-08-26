"""GoPro test-set (LMDB) PSNR benchmark: input vs DeblurGAN checkpoints vs NAFNet.

  python experiments/eval_gopro_test.py --data D:/Photos/TrainingData/GoPro/test
"""
import argparse
import io
import math
import os
import sys

import lmdb
import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from deblur_fast import DEFAULT_CKPT, load_generator, load_nafnet, pad_to_multiple


def get(env, key, dev):
    with env.begin(write=False) as txn:
        data = txn.get(key.encode())
    a = np.asarray(Image.open(io.BytesIO(data)).convert('RGB'), dtype=np.float32)
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(dev) / 255.0


def psnr8(a, b):
    a = a.float().mul(255).round_().clamp_(0, 255)
    b = b.float().mul(255).round_().clamp_(0, 255)
    mse = (a - b).pow(2).mean().item()
    return float('inf') if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'D:/Photos/TrainingData/GoPro/test',
                    help='dir with input.lmdb and target.lmdb')
    ap.add_argument('--limit', type=int, default=0, help='0 = all images')
    args = ap.parse_args()
    dev = 'cuda'
    torch.backends.cudnn.benchmark = True

    with open(os.path.join(args.data, 'input.lmdb', 'meta_info.txt')) as f:
        keys = [line.split()[0].removesuffix('.png') for line in f if line.strip()]
    if args.limit:
        keys = keys[:args.limit]
    print(f'{len(keys)} test pairs')

    env_in = lmdb.open(os.path.join(args.data, 'input.lmdb'), readonly=True,
                       lock=False, readahead=False, meminit=False)
    env_gt = lmdb.open(os.path.join(args.data, 'target.lmdb'), readonly=True,
                       lock=False, readahead=False, meminit=False)

    models = {}
    official = os.path.join(REPO, 'checkpoints', 'official', 'latest_net_G.pth')
    if os.path.exists(official):
        models['official DeblurGAN'] = ('gan', *load_generator(official, dev, 'fp16'))
    buggy = os.path.join(REPO, 'checkpoints', 'experiment_name', 'latest_net_G.pth')
    if os.path.exists(buggy):
        models['repo (buggy) ckpt'] = ('gan', *load_generator(buggy, dev, 'fp16'))
    if os.path.exists(DEFAULT_CKPT['nafnet']):
        net, ac, mult = load_nafnet(DEFAULT_CKPT['nafnet'], dev, 'fp16')
        models['NAFNet-w64'] = ('nafnet', net, ac, mult)

    sums = {'input': 0.0, **dict.fromkeys(models, 0.0)}
    with torch.inference_mode():
        for i, key in enumerate(keys):
            x = get(env_in, key, dev)
            gt = get(env_gt, key, dev)
            sums['input'] += psnr8(x, gt)
            for name, spec in models.items():
                if spec[0] == 'gan':
                    xp, h, w = pad_to_multiple((x * 2 - 1).half(), 4)
                    y = spec[1](xp)[:, :, :h, :w]
                    sums[name] += psnr8((y.float() + 1) / 2, gt)
                else:
                    _, net, ac, mult = spec
                    xp, h, w = pad_to_multiple(x, mult)
                    with torch.autocast('cuda', dtype=ac):
                        y = net(xp)
                    sums[name] += psnr8(y[:, :, :h, :w], gt)
            if (i + 1) % 200 == 0:
                print(f'{i + 1}/{len(keys)}: ' +
                      ', '.join(f'{k} {v / (i + 1):.2f}' for k, v in sums.items()),
                      flush=True)

    print(f'\n=== GoPro test ({len(keys)} images), average PSNR ===')
    for k, v in sums.items():
        print(f'{k}: {v / len(keys):.2f} dB')


if __name__ == '__main__':
    main()
