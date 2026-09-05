"""Precompute NAFNet teacher labels for distillation.

Runs NAFNet-w64 (autocast-fp16) over every image in <data>/input and writes
the outputs as PNGs to <data>/<out-sub>, skipping files that already exist.
Full-resolution inference gives the teacher full context (better labels than
per-crop teachering) and removes all teacher cost from the training loop.

  python experiments/make_teacher_labels.py --data D:/Photos/TrainingData/GoPro/train
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from deblur_fast import DEFAULT_CKPT, load_nafnet, pad_to_multiple


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'D:/Photos/TrainingData/GoPro/train')
    ap.add_argument('--out-sub', default='teacher_nafnet_tlc',
                    help='output subdir under --data (default reflects the '
                         'TLC teacher; older teacher_nafnet/ dirs are non-TLC)')
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT['nafnet'])
    ap.add_argument('--no-tlc', action='store_true')
    args = ap.parse_args()
    dev = 'cuda'
    torch.backends.cudnn.benchmark = True

    in_dir = os.path.join(args.data, 'input')
    out_dir = os.path.join(args.data, args.out_sub)
    os.makedirs(out_dir, exist_ok=True)
    # provenance marker so a label dir can never be mistaken for another
    # teacher configuration (skip-existing would otherwise mix them silently)
    marker = os.path.join(out_dir, 'TEACHER.json')
    config = {'checkpoint': os.path.basename(args.checkpoint),
              'tlc': not args.no_tlc, 'precision': 'autocast-fp16'}
    if os.path.exists(marker):
        with open(marker) as f:
            prev = json.load(f)
        if prev != config:
            sys.exit(f'{out_dir} was generated with a different teacher config '
                     f'{prev}; use another --out-sub')
    else:
        with open(marker, 'w') as f:
            json.dump(config, f)
    names = sorted(n for n in os.listdir(in_dir)
                   if not os.path.exists(os.path.join(out_dir, n)))
    print(f'{len(names)} images to label -> {out_dir}', flush=True)
    if not names:
        return

    net, ac_dtype, mult = load_nafnet(args.checkpoint, dev, precision='fp16')

    def save(arr, path):
        Image.fromarray(arr).save(path)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        futures = []
        for i, n in enumerate(names):
            a = np.asarray(Image.open(os.path.join(in_dir, n)).convert('RGB'),
                           dtype=np.float32)
            x = torch.from_numpy(a).permute(2, 0, 1)[None].to(dev) / 255.0
            xp, h, w = pad_to_multiple(x, mult)
            with torch.autocast('cuda', dtype=ac_dtype):
                y = net(xp)
            y = (y[:, :, :h, :w].float() * 255).round_().clamp_(0, 255)
            arr = y[0].to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            futures.append(pool.submit(save, arr, os.path.join(out_dir, n)))
            if (i + 1) % 200 == 0:
                print(f'{i + 1}/{len(names)} ({(i + 1) / (time.time() - t0):.1f} img/s)',
                      flush=True)
        for f in futures:
            f.result()
    print(f'done: {len(names)} teacher labels in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
