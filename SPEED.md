# DeblurGAN: models, results, how to run

RTX 5070 Ti (16 GB), PyTorch 2.13.0+cu130, Python 3.14, venv `D:\PyEnv\torch`.

## Results — GoPro test set (1111 images), 2026-08-26

| model | avg PSNR | 720p ms (compiled) | 1080p ms (compiled) |
|---|---|---|---|
| blurry input (RGB / gray) | 25.64 / 25.69 | — | — |
| official DeblurGAN, recovered (11.4M, RGB) | 27.30 | 33.4 (19.6) | 81.2 (45.9) |
| **student, GT-trained (2.84M, gray)** | **29.07** | 13.0 (**6.6**) | 32.6 (**14.5**) |
| NAFNet-w64 (68M, RGB) | 33.08 | 123.8 (42.5) | 327.4 (102.2) |

Reproduce: `python experiments/benchmark_all.py`. Paper reports 28.7 dB for
DeblurGAN (Y-channel vs RGB PSNR accounts for the offset).

## Tutorial

Python needs: `torch`/`torchvision` (cu130), `numpy`, `pillow`; `spandrel`
(NAFNet), `triton-windows` (`--compile`), `lmdb` (benchmarks). All installed
in `D:\PyEnv\torch`; `py` below means `D:\PyEnv\torch\Scripts\python.exe`.

### Deblur images

```powershell
py deblur_fast.py --input <img-or-dir> --output <dir>                  # official DeblurGAN
py deblur_fast.py --arch nafnet  --input <dir> --output <dir>          # best quality
py deblur_fast.py --arch student --input <dir> --output <dir>          # real-time, grayscale
py deblur_fast.py --input examples\gopro\input --output examples\gopro\deblurred   # demo
py deblur_fast.py --input D:\frames --output D:\out --batch 4 --compile --ext jpg  # bulk
```

| flag | effect |
|---|---|
| `--arch deblurgan\|nafnet\|student` | model (defaults: `checkpoints/official/`, `checkpoints/NAFNet-GoPro-width64.pth`, `checkpoints/student/student_latest.pth`) |
| `--checkpoint <path>` | other weights; DeblurGAN block layout auto-detected (legacy one-conv / paper two-conv) |
| `--compile` | torch.compile max-autotune; minutes-long first compile per image size, then fastest — for many same-size images |
| `--gray` | 1-channel in/out (exact first-conv fold for deblurgan) |
| `--batch N` | same-size images batched together |
| `--precision fp16\|bf16\|fp32` | default fp16 (nafnet: autocast — full half casts break its LayerNorms) |
| `--ext`, `--jpeg-quality`, `--suffix`, `--workers` | output format / naming / loader threads |
| `--no-residual` | for DeblurGAN checkpoints trained without `--learn_residual` |

### Benchmark

```powershell
py experiments/benchmark_all.py                    # quality (full GoPro test) + speed, all models
py experiments/benchmark_all.py --skip-compile     # quick pass
py benchmark_deblur.py --compile                   # precision/layout matrix, one checkpoint
```

### Train

Data layout: `<data>/input/*.png` + `<data>/target/*.png` (GoPro train at
`D:/Photos/TrainingData/GoPro/train`).

```powershell
py train_student.py                                # slim gray student, 100k iters (~7 h)
py train_student.py --resume                       # continue after a crash
py experiments/make_teacher_labels.py              # precompute NAFNet outputs (once, ~6 min)
py train_student.py --teacher-sub teacher_nafnet --init checkpoints/student/student_best.pth --iters 60000 --lr 1.5e-4 --out checkpoints/student_nafnet   # distill from NAFNet
py train_deblurgan.py --iters 60000                # full paper retrain, modernized (~1 day)
```

`--teacher-alpha` blends teacher vs ground truth (default 1.0 = pure teacher);
validation always scores vs GT. Checkpoints are self-describing: load any of
them with `--arch student --checkpoint <file>`.

Long runs at lowest CPU priority, detached:

```powershell
$p = Start-Process D:\PyEnv\torch\Scripts\python.exe -ArgumentList 'train_student.py' `
  -WorkingDirectory E:\OneDrive\Coding\DeblurGAN -RedirectStandardOutput out.log `
  -RedirectStandardError err.log -WindowStyle Hidden -PassThru
$p.PriorityClass = 'Idle'
```

## Findings

- `deblur_fast.py` replaces `test.py`: full-resolution inference (reflect-pad,
  not 256² crops), dropout off (`test.py` left it active — stochastic output),
  per-instance norm stats, prefetching, threaded encoding. Verified
  bit-identical to the original network.
- fp16 is safe (66 dB vs fp32) and default. torch.compile is the big lever:
  3.5x total on the legacy model (113 -> 31.9 ms/1080p), similar ratio on all
  archs; max-autotune > default > reduce-overhead.
- channels_last is slower on this InstanceNorm/ReflectionPad net; only cuts
  VRAM. bf16 has no advantage here.
- `--gray` on the RGB net saves ~2% (channel widths are internal); the real
  gray speedup is the slim student (trained, 29.07 dB, table above).
- FP8 tested via TensorRT 11 + ModelOpt PTQ: 12.7 vs 13.7 ms @720p (+7%),
  58.5 dB — net is norm-bound, not worth a second runtime. Stock PyTorch has
  no FP8 conv kernels at all. TRT fp16 ties torch.compile.
- Zero-retrain channel pruning/merging fails (`experiments/prune_probe.py`):
  dropout training left no dead channels; dropping 12.5% already costs ~7 dB.
- Legacy code quirks: upstream `ResnetBlock` had an operator-precedence bug
  (one conv per block instead of two) — fixed here; old one-conv checkpoints
  still load via auto-detection. `train.py` writes `opt.txt` before its
  hardcoded overrides, so old `opt.txt` files lie.

## Weights

Official weights vanished (README Drive link deleted, Dropbox tombstone; only
buggy one-conv retrains circulated). Recovered by scanning all 531 upstream
forks for a divergent committed checkpoint: one hit, `haozhe15/DeblurGAN`
(committed 2018-11-12, "download weights"), 45.6 MB two-conv generator
matching the official key layout from upstream issue #145. Committed at
`checkpoints/official/latest_net_G.pth` (default). The buggy upstream
checkpoint (25.3 dB — below the blurry input) is deleted from this fork.

## Better models (Aug 2026)

| model | GoPro PSNR | notes |
|---|---|---|
| [NAFNet-w64](https://github.com/megvii-research/NAFNet) | 33.7 (33.08 here, RGB) | MIT; integrated as `--arch nafnet` |
| [FFTformer](https://github.com/kkkls/FFTformer) | 34.2 | MIT, weights in repo, ~2-3 s/1080p |
| [MIMO-UNet+](https://github.com/chosj95/MIMO-UNet) | 32.5 | video-rate; no license file |
| [EVSSM](https://github.com/kkkls/EVSSM) | 34.5 | MIT; mamba-ssm needs WSL2 |

For real handheld photos prefer RealBlur-trained checkpoints (FFTformer ships
one). Loadable via `spandrel` on this venv.
