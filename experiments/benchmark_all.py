"""Full benchmark of every available model: quality (GoPro test LMDB, all
1111 pairs) and speed (720p/1080p, eager + compiled), on an idle GPU.

  python experiments/benchmark_all.py --data D:/Photos/TrainingData/GoPro/test
"""
import argparse
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from deblur_fast import (DEFAULT_CKPT, load_generator, load_nafnet,
                         load_student, pad_to_multiple)
from deblur_utils import GoProTestLMDB, luma, psnr8

SIZES = ((1280, 720), (1920, 1080))


def time_model(fn, x, iters=15, warmup=5):
    with torch.inference_mode():
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn(x)
        e.record()
        torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'D:/Photos/TrainingData/GoPro/test')
    ap.add_argument('--out', default=os.path.join(REPO, 'experiments', 'results',
                                                  'benchmark_all.md'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--skip-compile', action='store_true',
                    help='eager speed only (for quick smoke tests)')
    args = ap.parse_args()
    dev = 'cuda'
    torch.backends.cudnn.benchmark = True
    lines = [f'# Benchmark {time.strftime("%Y-%m-%d %H:%M")} '
             f'({torch.cuda.get_device_name(0)}, torch {torch.__version__})', '']

    official_ck = os.path.join(REPO, 'checkpoints', 'official', 'latest_net_G.pth')
    student_cks = []          # every FastGenerator-container checkpoint present
    for label, p in (
            ('student GT-trained', os.path.join(REPO, 'checkpoints', 'student',
                                                'student_best.pth')),
            ('student NAFNet-distilled', os.path.join(REPO, 'checkpoints',
                                                      'student_nafnet',
                                                      'student_best.pth')),
            ('student NAFNet-TLC-distilled', os.path.join(REPO, 'checkpoints',
                                                          'student_nafnet_tlc',
                                                          'student_best.pth')),
            ('DeblurGAN retrained (modern recipe)',
             os.path.join(REPO, 'checkpoints', 'retrain', 'deblurgan_best.pth')),
            ('DeblurGAN retrained v2 (dropout 0.5)',
             os.path.join(REPO, 'checkpoints', 'retrain_v2', 'deblurgan_best.pth'))):
        if os.path.exists(p):
            student_cks.append((label, p))

    # ---------------- quality: full GoPro test set --------------------------
    test = GoProTestLMDB(args.data, limit=args.limit)
    keys = test.keys

    official, _ = load_generator(official_ck, dev, precision='fp16')
    naf, naf_ac, naf_mult = load_nafnet(DEFAULT_CKPT['nafnet'], dev, 'fp16')
    containers = []
    for label, p in student_cks:
        net, dt, cfg = load_student(p, dev, precision='fp16')
        it = torch.load(p, map_location='cpu', weights_only=True).get('iter', '?')
        rgb = cfg.get('in_ch', 3) == 3
        containers.append((f'{label}@{it} ({"RGB" if rgb else "gray"})',
                           net, dt, rgb))

    sums = {'input (RGB)': 0.0, 'official DeblurGAN (RGB)': 0.0,
            'NAFNet-w64 (RGB)': 0.0, 'input (gray)': 0.0}
    sums.update({label: 0.0 for label, _, _, _ in containers})
    with torch.inference_mode():
        for i, key in enumerate(keys):
            x, gt = test.pair(key, dev)
            sums['input (RGB)'] += psnr8(x, gt)

            xp, h, w = pad_to_multiple((x * 2 - 1).half(), 4)
            y = official(xp)[:, :, :h, :w]
            sums['official DeblurGAN (RGB)'] += psnr8((y.float() + 1) / 2, gt)

            xp, h, w = pad_to_multiple(x, naf_mult)
            with torch.autocast('cuda', dtype=naf_ac):
                y = naf(xp)
            sums['NAFNet-w64 (RGB)'] += psnr8(y[:, :, :h, :w], gt)

            xg, gtg = luma(x), luma(gt)
            sums['input (gray)'] += psnr8(xg, gtg)
            for label, stu, dt, rgb in containers:
                src, ref = (x, gt) if rgb else (xg, gtg)
                xp, h, w = pad_to_multiple((src * 2 - 1).to(dt), 4)
                y = stu(xp)[:, :, :h, :w]
                sums[label] += psnr8((y.float() + 1) / 2, ref)

            if (i + 1) % 200 == 0:
                print(f'{i + 1}/{len(keys)}', flush=True)

    n = len(keys)
    lines += [f'## Quality: GoPro test set ({n} images), average PSNR', '',
              '| model | PSNR |', '|---|---|']
    for k, v in sums.items():
        lines.append(f'| {k} | {v / n:.2f} dB |')
    lines.append('')
    print('\n'.join(lines[-(len(sums) + 4):]), flush=True)

    del official, naf, containers
    torch.cuda.empty_cache()

    # ---------------- speed --------------------------------------------------
    lines += ['## Speed (batch 1, ms/img; eager then torch.compile)', '',
              '| model | ' + ' | '.join(f'{w}x{h}' for w, h in SIZES) +
              ' | compiled ' +
              ' | compiled '.join(f'{w}x{h}' for w, h in SIZES) + ' |',
              '|---' * (2 * len(SIZES) + 1) + '|']

    def speed_row(name, make_net, run, pad_mult, in_ch=3, dtype=torch.float16):
        row = [name]
        net = make_net()
        for (w, h) in SIZES:
            x = torch.rand(1, in_ch, h, w, device=dev, dtype=dtype) * 2 - 1
            x, _, _ = pad_to_multiple(x, pad_mult)
            row.append(f'{time_model(lambda t: run(net, t), x):.1f}')
        if args.skip_compile:
            row += ['-'] * len(SIZES)
        else:
            cnet = torch.compile(net, mode='max-autotune', dynamic=False)
            for (w, h) in SIZES:
                x = torch.rand(1, in_ch, h, w, device=dev, dtype=dtype) * 2 - 1
                x, _, _ = pad_to_multiple(x, pad_mult)
                row.append(f'{time_model(lambda t: run(cnet, t), x):.1f}')
            del cnet
        del net
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        return '| ' + ' | '.join(row) + ' |'

    lines.append(speed_row(
        'official DeblurGAN fp16',
        lambda: load_generator(official_ck, dev, precision='fp16')[0],
        lambda n_, t: n_(t), 4))
    print('speed: official done', flush=True)

    def run_naf(n_, t):
        with torch.autocast('cuda', dtype=torch.float16):
            return n_(t)
    lines.append(speed_row(
        'NAFNet-w64 autocast-fp16',
        lambda: load_nafnet(DEFAULT_CKPT['nafnet'], dev, 'fp16')[0],
        run_naf, 16, dtype=torch.float32))
    print('speed: nafnet done', flush=True)

    # speed of the slim student architecture (all student variants share it):
    # the first container whose arch is grayscale/slim
    for label, p in student_cks:
        cfg = torch.load(p, map_location='cpu', weights_only=True)['arch_config']
        if cfg.get('in_ch', 3) == 1:
            lines.append(speed_row(
                f'{label} fp16 (gray)',
                lambda p=p: load_student(p, dev, precision='fp16')[0],
                lambda n_, t: n_(t), 4, in_ch=1))
            print('speed: student done', flush=True)
            break

    text = '\n'.join(lines) + '\n'
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(text)
    print('\n' + text)


if __name__ == '__main__':
    main()
