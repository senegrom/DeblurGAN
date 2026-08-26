"""Retrain the real (two-conv) DeblurGAN generator on GoPro pairs.

The official pretrained weights are gone from every source (see SPEED.md), and
every surviving checkpoint was trained on the one-conv ResnetBlock bug. This
script retrains the paper architecture — FastGenerator block_layout='paper',
identical state-dict layout to the original released weights (issue #145) —
with the paper's losses, modernized:

  * content: VGG19 conv3_3 feature MSE x100 (with proper ImageNet norm)
  * adversarial: WGAN-GP (5 critic iters, lambda 10) or --gan-type lsgan
  * optional Charbonnier pixel anchor (--pixel-weight, default 10; 0 = paper-pure)

Checkpoints are self-describing and load with:
  deblur_fast.py --arch student --checkpoint checkpoints/retrain/deblurgan_best.pth

  python train_deblurgan.py --data D:/Photos/TrainingData/GoPro/train
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from deblur_fast import FastGenerator, pad_to_multiple

ARCH_CONFIG = dict(in_ch=3, out_ch=3, ngf=64, n_blocks=9, residual=True,
                   trunk=256, block_layout='paper')


class RGBPairs(torch.utils.data.Dataset):
    def __init__(self, root, names, crop):
        self.root, self.names, self.crop = root, names, crop

    def __len__(self):
        return len(self.names)

    def _load(self, sub, name):
        a = np.asarray(Image.open(os.path.join(self.root, sub, name)).convert('RGB'),
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
        blur = np.ascontiguousarray(blur.transpose(2, 0, 1))
        sharp = np.ascontiguousarray(sharp.transpose(2, 0, 1))
        return torch.from_numpy(blur), torch.from_numpy(sharp)


class NLayerDiscriminator(nn.Module):
    """PatchGAN critic (3 layers, instance norm), as in the paper."""

    def __init__(self, ndf=64):
        super().__init__()
        def block(cin, cout, stride):
            return [nn.Conv2d(cin, cout, 4, stride, 2),
                    nn.InstanceNorm2d(cout), nn.LeakyReLU(0.2, True)]
        layers = [nn.Conv2d(3, ndf, 4, 2, 2), nn.LeakyReLU(0.2, True)]
        layers += block(ndf, ndf * 2, 2) + block(ndf * 2, ndf * 4, 2)
        layers += block(ndf * 4, ndf * 8, 1)
        layers += [nn.Conv2d(ndf * 8, 1, 4, 1, 2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VGGContent(nn.Module):
    """VGG19 up to conv3_3 (paper's content layer), ImageNet-normalized."""

    def __init__(self, device):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19
        feats = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:15]
        self.feats = feats.to(device).eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406],
                                                  device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225],
                                                 device=device).view(1, 3, 1, 1))

    def forward(self, x):                       # x in [-1,1]
        x = ((x + 1) / 2 - self.mean) / self.std
        return self.feats(x)


def charbonnier(a, b, eps=1e-3):
    return torch.sqrt((a - b) ** 2 + eps * eps).mean()


def gradient_penalty(critic, real, fake):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    mix = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = critic(mix)
    grad = torch.autograd.grad(score.sum(), mix, create_graph=True)[0]
    return ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def psnr_pair(pred, gt):
    p = (pred.float() + 1).mul(127.5).round_().clamp_(0, 255)
    g = (gt.float() + 1).mul(127.5).round_().clamp_(0, 255)
    mse = (p - g).pow(2).mean().item()
    return float('inf') if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))


@torch.inference_mode()
def evaluate(net, root, names, device):
    net.eval()
    tot, base = 0.0, 0.0
    for name in names:
        blur = np.asarray(Image.open(os.path.join(root, 'input', name)).convert('RGB'),
                          dtype=np.float32) / 127.5 - 1
        sharp = np.asarray(Image.open(os.path.join(root, 'target', name)).convert('RGB'),
                           dtype=np.float32) / 127.5 - 1
        x = torch.from_numpy(blur.transpose(2, 0, 1))[None].to(device)
        g = torch.from_numpy(sharp.transpose(2, 0, 1))[None].to(device)
        xp, h, w = pad_to_multiple(x, 4)
        y = net(xp)[:, :, :h, :w]
        tot += psnr_pair(y, g)
        base += psnr_pair(x, g)
    net.train()
    return tot / len(names), base / len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'D:/Photos/TrainingData/GoPro/train')
    ap.add_argument('--out', default=os.path.join('checkpoints', 'retrain'))
    ap.add_argument('--iters', type=int, default=100000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--crop', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--gan-type', choices=['wgan-gp', 'lsgan'], default='wgan-gp')
    ap.add_argument('--critic-iters', type=int, default=5,
                    help='critic updates per G update (wgan-gp only)')
    ap.add_argument('--content-weight', type=float, default=100.0)
    ap.add_argument('--pixel-weight', type=float, default=10.0,
                    help='Charbonnier anchor; 0 = paper-pure recipe')
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
    log(f'{len(train_names)} train / {len(val_names)} val pairs; '
        f'gan={args.gan_type} content_w={args.content_weight} '
        f'pixel_w={args.pixel_weight}')

    netG = FastGenerator(**ARCH_CONFIG).to(device).train()
    netD = NLayerDiscriminator().to(device).train()
    vgg = VGGContent(device)
    log(f'G params: {sum(p.numel() for p in netG.parameters()) / 1e6:.2f}M, '
        f'D params: {sum(p.numel() for p in netD.parameters()) / 1e6:.2f}M')

    optG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    schedG = torch.optim.lr_scheduler.CosineAnnealingLR(optG, args.iters, 1e-6)
    schedD = torch.optim.lr_scheduler.CosineAnnealingLR(optD, args.iters, 1e-6)

    start_iter, best_psnr = 0, 0.0
    state_path = os.path.join(args.out, 'train_state.pth')
    if args.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=True)
        netG.load_state_dict(st['G'])
        netD.load_state_dict(st['D'])
        optG.load_state_dict(st['optG'])
        optD.load_state_dict(st['optD'])
        schedG.load_state_dict(st['schedG'])
        schedD.load_state_dict(st['schedD'])
        start_iter, best_psnr = st['iter'], st['best_psnr']
        log(f'resumed from iter {start_iter} (best {best_psnr:.2f} dB)')

    loader = torch.utils.data.DataLoader(
        RGBPairs(args.data, train_names, args.crop), batch_size=args.batch,
        shuffle=True, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=True)

    def save(tag, it, psnr):
        torch.save({'arch_config': ARCH_CONFIG, 'state_dict': netG.state_dict(),
                    'iter': it, 'psnr': psnr},
                   os.path.join(args.out, f'deblurgan_{tag}.pth'))

    it = start_iter
    t0 = time.time()
    acc = {'g': 0.0, 'd': 0.0, 'content': 0.0, 'pixel': 0.0}
    n_acc = 0
    data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        try:
            b = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            b = next(data_iter)
        return (t.to(device, non_blocking=True) for t in b)

    while it < args.iters:
        # --- critic ---------------------------------------------------------
        d_steps = args.critic_iters if args.gan_type == 'wgan-gp' else 1
        for _ in range(d_steps):
            blur, sharp = next_batch()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                fake = netG(blur)
            fake = fake.float()
            optD.zero_grad(set_to_none=True)
            if args.gan_type == 'wgan-gp':
                lossD = (netD(fake).mean() - netD(sharp).mean()
                         + 10.0 * gradient_penalty(netD, sharp, fake))
            else:
                lossD = 0.5 * (F.mse_loss(netD(sharp), torch.ones_like(netD(sharp)))
                               + F.mse_loss(netD(fake), torch.zeros_like(netD(fake))))
            lossD.backward()
            optD.step()
        schedD.step()

        # --- generator ------------------------------------------------------
        blur, sharp = next_batch()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            fake = netG(blur)
        fake = fake.float()
        content = F.mse_loss(vgg(fake), vgg(sharp))
        pixel = charbonnier(fake, sharp) if args.pixel_weight else fake.new_zeros(())
        if args.gan_type == 'wgan-gp':
            adv = -netD(fake).mean()
        else:
            pred = netD(fake)
            adv = F.mse_loss(pred, torch.ones_like(pred))
        lossG = (args.content_weight * content + args.pixel_weight * pixel + adv)
        optG.zero_grad(set_to_none=True)
        lossG.backward()
        optG.step()
        schedG.step()

        acc['g'] += lossG.item()
        acc['d'] += lossD.item()
        acc['content'] += content.item()
        acc['pixel'] += float(pixel.detach())
        n_acc += 1
        it += 1

        if it % 100 == 0:
            rate = (it - start_iter) / (time.time() - t0)
            eta_h = (args.iters - it) / rate / 3600 if rate > 0 else 0
            log(f'iter {it}/{args.iters} G {acc["g"] / n_acc:.3f} '
                f'D {acc["d"] / n_acc:.3f} content {acc["content"] / n_acc:.4f} '
                f'pixel {acc["pixel"] / n_acc:.4f} '
                f'{rate:.2f} it/s eta {eta_h:.1f}h')
            acc = dict.fromkeys(acc, 0.0)
            n_acc = 0

        if it % args.eval_every == 0:
            psnr, base = evaluate(netG, args.data, val_names, device)
            log(f'iter {it}: val PSNR {psnr:.2f} dB (blurry input: {base:.2f})')
            save('latest', it, psnr)
            torch.save({'G': netG.state_dict(), 'D': netD.state_dict(),
                        'optG': optG.state_dict(), 'optD': optD.state_dict(),
                        'schedG': schedG.state_dict(), 'schedD': schedD.state_dict(),
                        'iter': it, 'best_psnr': best_psnr}, state_path)
            if psnr > best_psnr:
                best_psnr = psnr
                save('best', it, psnr)
                log(f'new best: {psnr:.2f} dB')

    psnr, _ = evaluate(netG, args.data, val_names, device)
    save('latest', it, psnr)
    if psnr > best_psnr:
        save('best', it, psnr)
    log(f'done at iter {it}: final val PSNR {psnr:.2f} dB '
        f'(best {max(best_psnr, psnr):.2f})')


if __name__ == '__main__':
    main()
