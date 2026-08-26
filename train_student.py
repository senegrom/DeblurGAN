"""Train a slim grayscale deblurring student (see SPEED.md).

Student: FastGenerator(in_ch=1, out_ch=1, ngf=32, trunk=128) — the fixed
two-conv residual block, ~4x fewer FLOPs than the original RGB generator.
Supervised on GoPro train pairs (grayscale), Charbonnier + FFT loss, bf16
autocast, cosine LR. Checkpoints are self-describing and load directly with
`deblur_fast.py --arch student`.

  python train_student.py --data D:/Photos/TrainingData/GoPro/train
  python train_student.py --resume   # continue from the saved training state
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from deblur_fast import FastGenerator, pad_to_multiple

ARCH_CONFIG = dict(in_ch=1, out_ch=1, ngf=32, n_blocks=9, residual=True,
                   trunk=128, block_layout='paper')


class GrayPairs(torch.utils.data.Dataset):
    """Random same-position crops from GoPro-style input/ + target/ pairs."""

    def __init__(self, root, names, crop):
        self.root = root
        self.names = names
        self.crop = crop

    def __len__(self):
        return len(self.names)

    def _load(self, sub, name):
        a = np.asarray(Image.open(os.path.join(self.root, sub, name)).convert('L'),
                       dtype=np.float32)
        return a / 127.5 - 1.0

    def __getitem__(self, i):
        name = self.names[i]
        blur, sharp = self._load('input', name), self._load('target', name)
        c = self.crop
        y = random.randint(0, blur.shape[0] - c)
        x = random.randint(0, blur.shape[1] - c)
        blur, sharp = blur[y:y + c, x:x + c], sharp[y:y + c, x:x + c]
        if random.random() < 0.5:
            blur, sharp = blur[:, ::-1], sharp[:, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(blur))[None],
                torch.from_numpy(np.ascontiguousarray(sharp))[None])


def charbonnier(a, b, eps=1e-3):
    return torch.sqrt((a - b) ** 2 + eps * eps).mean()


def fft_l1(a, b):
    fa, fb = torch.fft.rfft2(a), torch.fft.rfft2(b)
    return (torch.view_as_real(fa) - torch.view_as_real(fb)).abs().mean()


def psnr_pair(pred, gt):
    """uint8-domain PSNR for tensors in [-1,1]."""
    p = (pred.float() + 1).mul(127.5).round_().clamp_(0, 255)
    g = (gt.float() + 1).mul(127.5).round_().clamp_(0, 255)
    mse = (p - g).pow(2).mean().item()
    return float('inf') if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))


@torch.inference_mode()
def evaluate(net, root, names, device, log):
    net.eval()
    tot, base = 0.0, 0.0
    for name in names:
        blur = np.asarray(Image.open(os.path.join(root, 'input', name)).convert('L'),
                          dtype=np.float32) / 127.5 - 1
        sharp = np.asarray(Image.open(os.path.join(root, 'target', name)).convert('L'),
                           dtype=np.float32) / 127.5 - 1
        x = torch.from_numpy(blur)[None, None].to(device)
        g = torch.from_numpy(sharp)[None, None].to(device)
        xp, h, w = pad_to_multiple(x, 4)
        y = net(xp)[:, :, :h, :w]
        tot += psnr_pair(y, g)
        base += psnr_pair(x, g)
    net.train()
    return tot / len(names), base / len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'D:/Photos/TrainingData/GoPro/train',
                    help='dir with input/ and target/ subdirs of paired images')
    ap.add_argument('--out', default=os.path.join('checkpoints', 'student'))
    ap.add_argument('--iters', type=int, default=100000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--crop', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--val-count', type=int, default=20)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, 'train_log.txt')

    def log(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    names = sorted(os.listdir(os.path.join(args.data, 'input')))
    val_names = names[-args.val_count:]
    train_names = names[:-args.val_count]
    log(f'{len(train_names)} train / {len(val_names)} val pairs from {args.data}')

    net = FastGenerator(**ARCH_CONFIG).to(device).train()
    n_params = sum(p.numel() for p in net.parameters())
    log(f'student params: {n_params / 1e6:.2f}M (config {ARCH_CONFIG})')

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.9),
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters, eta_min=1e-6)

    start_iter = 0
    state_path = os.path.join(args.out, 'train_state.pth')
    best_psnr = 0.0
    if args.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=True)
        net.load_state_dict(st['state_dict'])
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        start_iter, best_psnr = st['iter'], st['best_psnr']
        log(f'resumed from iter {start_iter} (best {best_psnr:.2f} dB)')

    ds = GrayPairs(args.data, train_names, args.crop)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=True)

    def save(tag, it, psnr):
        torch.save({'arch_config': ARCH_CONFIG, 'state_dict': net.state_dict(),
                    'iter': it, 'psnr': psnr},
                   os.path.join(args.out, f'student_{tag}.pth'))

    it = start_iter
    t0 = time.time()
    loss_acc, n_acc = 0.0, 0
    while it < args.iters:
        for blur, sharp in loader:
            if it >= args.iters:
                break
            blur = blur.to(device, non_blocking=True)
            sharp = sharp.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = net(blur)
            pred = pred.float()
            loss = charbonnier(pred, sharp) + 0.05 * fft_l1(pred, sharp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_acc += loss.item()
            n_acc += 1
            it += 1

            if it % 100 == 0:
                rate = (it - start_iter) / (time.time() - t0)
                eta_h = (args.iters - it) / rate / 3600 if rate > 0 else 0
                log(f'iter {it}/{args.iters} loss {loss_acc / n_acc:.4f} '
                    f'lr {sched.get_last_lr()[0]:.2e} '
                    f'{rate:.1f} it/s eta {eta_h:.1f}h')
                loss_acc, n_acc = 0.0, 0

            if it % args.eval_every == 0:
                psnr, base = evaluate(net, args.data, val_names, device, log)
                log(f'iter {it}: val PSNR {psnr:.2f} dB (blurry input: {base:.2f})')
                save('latest', it, psnr)
                torch.save({'state_dict': net.state_dict(), 'opt': opt.state_dict(),
                            'sched': sched.state_dict(), 'iter': it,
                            'best_psnr': best_psnr}, state_path)
                if psnr > best_psnr:
                    best_psnr = psnr
                    save('best', it, psnr)
                    log(f'new best: {psnr:.2f} dB')

    psnr, base = evaluate(net, args.data, val_names, device, log)
    save('latest', it, psnr)
    if psnr > best_psnr:
        save('best', it, psnr)
    log(f'done at iter {it}: final val PSNR {psnr:.2f} dB (best {max(best_psnr, psnr):.2f})')


if __name__ == '__main__':
    main()
