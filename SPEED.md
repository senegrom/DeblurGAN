# DeblurGAN inference speedups

Measured on RTX 5070 Ti (16 GB, Blackwell sm_120), PyTorch 2.13.0+cu130,
Python 3.14, venv `D:\PyEnv\torch`, checkpoint
`checkpoints/experiment_name/latest_net_G.pth` (resnet_9blocks, instance norm,
learn_residual).

## TL;DR

Use `deblur_fast.py` instead of `test.py`:

```
D:/PyEnv/torch/Scripts/python.exe deblur_fast.py --input <blurry dir> --output <out dir>
```

Defaults (FP16, cuDNN autotune, prefetching, threaded encoding) are the
measured-fastest eager settings; add `--compile` for large same-size batches.
`benchmark_deblur.py` reproduces the numbers below.

## Example job

Deblur the original full-resolution GoPro test frames used by the demo GIFs:

```powershell
D:\PyEnv\torch\Scripts\python.exe deblur_fast.py --input examples\gopro\input --output examples\gopro\deblurred
```

Their paired sharp ground truth is in `examples/gopro/target/`. The older
640x360 GIF-extracted inputs and committed reference outputs remain in
`examples/blurry/` and `examples/deblurred/` for comparison. A realistic bulk
job -- a folder of same-size frames, batched, compiled, saved as JPEG:

```powershell
D:\PyEnv\torch\Scripts\python.exe deblur_fast.py --input D:\frames --output D:\frames_deblurred --batch 4 --compile --ext jpg
```

Grayscale variant (1-channel processing, luma output): add `--gray`.
Reproduce the benchmark table: `benchmark_deblur.py --compile`.

## Benchmark (batch 1, ms per image / img/s)

| config | 256x256 | 1280x720 | 1920x1080 | PSNR vs fp32 @1080p | peak VRAM @1080p |
|---|---|---|---|---|---|
| fp32 (baseline)      | 3.6 (280)  | 50.2 (19.9) | 113.1 (8.8) | ref     | 10.1 GB |
| fp32 + channels_last | 4.0 (250)  | 66.2 (15.1) | 146.4 (6.8) | 69.3 dB | 2.5 GB  |
| fp16                 | 4.4 (226)  | 25.2 (39.7) | 62.5 (16.0) | 66.1 dB | 9.9 GB  |
| fp16 + channels_last | 6.1 (163)  | 35.4 (28.3) | 90.3 (11.1) | 66.1 dB | 1.7 GB  |
| bf16 + channels_last | 3.8 (265)  | 38.0 (26.3) | 88.9 (11.3) | 57.5 dB | 1.7 GB  |
| fp16 GRAY (folded)   | 3.8 (265)  | 34.6 (28.9) | 91.2 (11.0) | n/a¹    | 1.4 GB  |
| **fp16 + torch.compile** | **1.1 (915)** | **15.0 (66.6)** | **31.9 (31.4)** | 66.2 dB | **0.9 GB** |

torch.compile (Inductor via `triton-windows`) fuses the 21 InstanceNorm/ReLU/pad
kernels this net is bottlenecked on: **3.5x end-to-end vs the fp32 baseline**.
Compile modes at 1080p: default 35.5 ms, reduce-overhead 36.8 ms, max-autotune
32.1 ms (the flag uses max-autotune); memory layout no longer matters once
Inductor picks its own. First run compiles for a few minutes per distinct input
size, cached afterwards — worth it for batches of same-size images, skip it for
a handful of photos.

¹ gray mode deblurs the luma image — a different computation than "deblur RGB,
then convert", so PSNR against the RGB pipeline is not a fidelity metric.
The folding itself is exact (verified ≤1.4e-4 vs the RGB net fed a
replicated-gray image, far below uint8 quantization).

Anything ≥50 dB is visually and numerically indistinguishable after uint8
quantization. FP16 is the right default.

## What was done

1. **FP16 weights + activations** — 1.7x at 1080p on tensor cores. InstanceNorm
   statistics still accumulate in fp32 internally, so it is numerically safe
   (66 dB vs fp32).
2. **Deterministic inference** — `test.py` never calls `eval()`, so the
   generator runs with **dropout active** at test time (the default
   `resnet_9blocks` is built with `use_dropout=True`): outputs were stochastic
   and slightly noisy. `deblur_fast.py` switches dropout off while keeping
   per-instance normalization statistics (matching how the net behaved in
   training; the checkpoint's InstanceNorm running stats are dropped).
3. **Full-resolution processing** — `test.py`'s single-image mode resizes to
   640x360 and takes a random 256x256 crop. The fast path pads to a multiple
   of 4 (reflect) and deblurs the whole image, cropping the pad afterwards.
4. **Pipeline overhead removed** — multi-worker prefetching with pinned memory
   and async H2D copies, size-bucketed batching (`--batch`), threaded PNG/JPEG
   encoding so disk writes overlap GPU compute, `torch.inference_mode`,
   `cudnn.benchmark`.
5. **channels_last measured and rejected** — on this InstanceNorm- and
   ReflectionPad-heavy architecture NHWC is *slower* (norm/pad kernels hit slow
   paths); it only wins on VRAM (10 GB -> 1.7 GB of mostly cuDNN workspace).
   Available via `--channels-last` if you ever run into OOM at >2K resolutions.

## Grayscale (`--gray`)

Implemented, but **grayscale cannot speed this network up meaningfully** and
the numbers show it (95.2 vs 96.9 ms at 1080p, ~2%). Reason: channel width is
fixed internally (64 -> 128 -> 256); ~95% of the FLOPs are in the 256-channel
trunk at H/4 x W/4 and the 128/256-channel down/up convs. Only the outer 7x7
convs touch the 3 input/output channels (~2-3% of compute).

What `--gray` does do, exactly:
- folds the first conv over RGB (`W_r+W_g+W_b`) so a 1-channel image is
  processed identically to a replicated-gray RGB image (verified exact),
- writes single-channel output (luma of the generator output),
- saves a third of the VRAM and all RGB<->gray conversion I/O.

A real grayscale speedup would require retraining a slimmer generator (e.g.
`--ngf 32`: ~4x fewer FLOPs) on grayscale pairs — a training job, not an
inference switch.

## FP8 (tested via the TensorRT port) / FP4

Stock PyTorch cannot run this model in FP8: its FP8 (`float8_e4m3fn`,
`torch._scaled_mm`) and the torchao FP8/FP4 (NVFP4) paths only cover
**matmuls / nn.Linear**, and this generator is 100% convolutions — there are
no FP8/FP4 conv kernels in the cuDNN bindings (verified in torch 2.13).

The working "port" is **TensorRT 11 + NVIDIA ModelOpt**, executed end-to-end
here: `torch.onnx.export` -> `modelopt.onnx.quantization --quantize_mode fp8`
(entropy calibration on the example frames; TRT 11 is strongly typed, so
precision comes from the ONNX dtypes/Q-DQ nodes, the old FP16/FP8 builder
flags are gone). Measured:

| engine @1280x720 | ms/img | PSNR vs PyTorch fp32 |
|---|---|---|
| TensorRT fp16          | 13.7 | 66.4 dB |
| TensorRT fp8 (ModelOpt PTQ) | 12.7 | 58.5 dB |
| (torch.compile fp16, for reference) | 15.0 | 66.2 dB |

TensorRT fp16 at 1080p: 32.4 ms — a dead heat with torch.compile (31.9 ms).

**Verdict: FP8 buys ~7% here.** The net is InstanceNorm/memory-bound, not
conv-math-bound (~26 effective TFLOPS of the card's ~88 FP16 TFLOPS), so
halving tensor-core math barely moves end-to-end time; the norms and
activations stay fp16 either way. Quality is fine (58.5 dB is invisible after
uint8), but a second runtime + quantization toolchain for 7% is not worth it —
`--compile` in PyTorch reaches the same place. FP4 (NVFP4) is weight-only /
Linear-focused for LLMs and has no conv path at all.

Toolchain note (Windows/py3.14): `pip install tensorrt nvidia-modelopt` plus
manual extras `onnx-graphsurgeon onnxslim onnxscript onnxruntime polygraphy
lief` (the `nvidia-modelopt[onnx]` extra pins `onnxruntime-gpu==1.22` which
has no py3.14 wheel — CPU onnxruntime calibrates fine, ~7 min for this model).

## Channel pruning / merging: measured, not viable without retraining

"Merge some channels through the whole network" was tested directly
(`experiments/prune_probe.py`): the trunk's 256-channel residual stream is shared by all
9 blocks (the residual adds force one consistent channel set), so slimming
means keeping the same top-K channels in down-conv out, every block conv
in+out, and up-conv in. Ranked by activation importance on real frames and
sliced the checkpoint:

| trunk width kept | PSNR vs full model | fp16 1080p ms (eager) |
|---|---|---|
| 256 (full) | ref | 74² |
| 224 (-12.5%) | 21.0-22.9 dB | 71 |
| 192 (-25%)  | 17.3-17.4 dB | 59 |
| 128 (-50%)  | 19.2 dB | 50 |

² eager fp16 run-to-run variance vs the table above; relative scaling is the point.

Dropout(0.5) training made every channel load-bearing: importance is nearly
uniform (min 1.35 / median 1.78 / max 2.07 mean-|activation|), only 3 channel
pairs correlate above |rho|=0.95, and dropping even the 32 least important
channels falls to ~21 dB — clearly visible artifacts. There is aggregate
linear redundancy (100/256 dimensions explain 95% of stream variance), but
exploiting it needs a learned re-projection (low-rank factorization) plus
finetuning — same story for grayscale: the honest route to a big win is
**distilling a slim student** (e.g. `FastGenerator(ngf=32)` or `trunk=128`,
1-channel in/out) on your training pairs with the current model as teacher;
roughly 4x fewer FLOPs, a few hours of training on this GPU.

## Quality reality check (why outputs look only mildly deblurred)

The inference path is verified correct — bit-identical to the repo's own
network (equivalence test: max abs diff 0.0) and the residual orientation was
confirmed empirically (`--no-residual` yields the tell-tale gray residual
map). What limits the results is the **checkpoint itself**:
`checkpoints/experiment_name/latest_net_G.pth` comes from this fork's
training setup — one-conv ResnetBlocks (half the paper's generator capacity),
vanilla GAN loss instead of the paper's WGAN-GP, and only a `latest` snapshot
exists (epoch snapshots would appear every 5 epochs), i.e. a very young run.
The demo GIFs' second frames show what the *official* paper model produces on
these exact inputs — far sharper. Retraining longer would help some, but see
the model-landscape note below before spending GPU-days on a 2017
architecture.

## The official pretrained weights: lost, hunted down, recovered

The README's Google Drive weights link is dead (the file is deleted — 404 on
its metadata page, not a quota block) and so is the pre-2018 Dropbox link
found in git history. Upstream issue #145 preserves the fingerprint of the
official file: two convs per ResnetBlock (`conv_block.1` + `conv_block.6`
keys) — the *paper* architecture, which the shipped 24.3 MB checkpoint
(byte-identical to upstream's committed one, blob `d2be26f9…`) does not have:
that one was trained after an operator-precedence bug collapsed every block
to a single conv, and it barely deblurs (see numbers below). A community
retrain shared in issue #230 (Sept 2023) uses the same buggy architecture and
is equally weak (measured 20.4/22.2 dB on the example frames).

The recovery: scanning all 531 forks of the upstream repo for committed
checkpoints whose size differs from the buggy 24,307,244 bytes turned up
exactly one hit — `haozhe15/DeblurGAN`, forked 2018-11-12 with a same-day
commit "download weights": a 45,565,607-byte `latest_net_G.pth`, matching the
expected ~45.6 MB of the true 11.39M-param two-conv generator, with exactly
the issue-#145 key layout. It is now committed here as
`checkpoints/official/latest_net_G.pth`, loads via `deblur_fast.py`'s
block-layout auto-detection with no code changes, and is the default
`--arch deblurgan` checkpoint.

GoPro test set, average RGB-PSNR measured here (first 800 of 1111 images —
the run was stopped early; the ranking was stable throughout):

| model | PSNR |
|---|---|
| blurry input (baseline) | 26.09 dB |
| repo (buggy one-conv) checkpoint | 25.30 dB — *worse than the input* |
| **official recovered weights** | **27.66 dB** |
| NAFNet-w64 | 32.95 dB |

(The paper reports 28.7 dB on GoPro; Y-channel vs RGB PSNR and evaluation
details account for small offsets. Reproduce with
`python experiments/eval_gopro_test.py --data <GoPro>/test`.)

On the two demo frames the recovered model produces the crisp GAN look of the
README GIFs (the shipped checkpoint's output is nearly indistinguishable from
the blurry input).

## Retraining (train_deblurgan.py / train_student.py)

With the official weights recovered, retraining the full model is optional —
but fully supported now: `train_deblurgan.py` trains the paper architecture
(fixed two-conv blocks) with the paper losses modernized (VGG19-conv3_3
perceptual x100 with proper ImageNet normalization — the original repo fed
[-1,1] images into VGG unnormalized — plus WGAN-GP or `--gan-type lsgan`,
optional Charbonnier anchor), bf16 autocast, on the GoPro pairs at
`D:/Photos/TrainingData/GoPro/train`. Expect roughly a day of GPU time for
100k iterations with the WGAN-GP 5:1 critic schedule.

`train_student.py` (running as of this writing) distills a slim grayscale
student — `FastGenerator(in_ch=1, out_ch=1, ngf=32, trunk=128)`, 2.84M params,
43 ms/1080p eager fp16 — supervised on the same pairs; checkpoints land in
`checkpoints/student/` and load with `--arch student`.

## Better models available (researched Aug 2026)

GoPro-benchmark reference: original DeblurGAN ~28.7 dB, DeblurGAN-v2 29.55.
Ready-to-use upgrades, all loadable on this exact venv (PyTorch 2.13) via
`pip install spandrel` (chaiNNer's MIT model loader) or their own repos:

| model | GoPro PSNR | speed class on this GPU | license / weights |
|---|---|---|---|
| [NAFNet-w64](https://github.com/megvii-research/NAFNet) (2022, CNN) | 33.71 dB | ~0.15-0.4 s per 1080p frame (w32: near-real-time at 32.87 dB) | MIT, Google Drive |
| [FFTformer](https://github.com/kkkls/FFTformer) (2023, transformer) | 34.21 dB | ~2-3 s per 1080p frame | MIT, weights in repo |
| [MIMO-UNet+](https://github.com/chosj95/MIMO-UNet) (2021, CNN) | 32.45 dB | ~40 ms per 1080p frame (video-rate) | no license file; weights on Drive |
| [EVSSM](https://github.com/kkkls/EVSSM) (2025, Mamba) | 34.51 dB | fast, but needs WSL2 (no Windows wheels for mamba-ssm) | MIT |
| [AdaRevD-L](https://github.com/INVOKERer/AdaRevD) (2024) | 34.60 dB | seconds/frame, research code | non-commercial |

**Practical verdict:** a downloaded NAFNet-w64 beats anything a retrained
original DeblurGAN can reach by ~4-5 dB with zero training (MIMO-UNet+ if you
need video-rate; FFTformer for max easy quality). For real handheld photos
(not GoPro-style blur), prefer RealBlur-trained checkpoints (FFTformer ships
one; MLWNet is the RealBlur champion). Nothing diffusion-based is
ready-to-use yet as of Aug 2026. This repo remains useful as a fast,
self-contained baseline — not as the quality frontier.

## Also noticed while reading the code (not changed)

- `models/networks.py` `ResnetBlock.__init__`: the ternary
  `a + b + c if use_dropout else [] + ...` binds as
  `(a+b+c) if use_dropout else ([]+...)`, so every "ResnetBlock" contains
  **one** conv instead of the paper's two (and no second norm). The shipped
  checkpoint (24.3 MB ≈ 6.07M params) is trained with this one-conv variant,
  so `deblur_fast.py` reproduces it exactly — fixing the block would
  invalidate the checkpoint (it would also roughly double trunk compute).
- `test.py` imports `from ssim import SSIM` (pip `pyssim`) only for
  commented-out code — it crashes if the package is missing.
- `train.py` overrides `learn_residual=True`, `gan_type='gan'` etc. *after*
  `opt.txt` is written, so `checkpoints/*/opt.txt` does not reflect the real
  training config.
