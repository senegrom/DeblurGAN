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

Deblur the bundled example images (first frames of the demo GIFs, in
`examples/blurry/`) with the shipped checkpoint:

```powershell
D:\PyEnv\torch\Scripts\python.exe deblur_fast.py --input examples\blurry --output examples\deblurred
```

Committed reference outputs are in `examples/deblurred/`. A realistic bulk job
— a folder of same-size frames, batched, compiled, saved as JPEG:

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

## FP8 / FP4

Not viable in stock PyTorch for this model, and not implemented:

- PyTorch's FP8 (`float8_e4m3fn`/`e5m2`, `torch._scaled_mm`) and the torchao
  FP8/FP4 (NVFP4) paths only cover **matmuls / nn.Linear** — this generator is
  100% convolutions; there are no FP8/FP4 conv kernels exposed by
  PyTorch/cuDNN bindings (checked torch 2.13: no
  `cudnn.conv.fp16_accumulate`, no fp8 conv op either).
- Even if kernels existed, the win over FP16 would be limited: at 1080p the
  net is already partly memory/norm-bound (~26 effective TFLOPS of the card's
  ~88 FP16 TFLOPS), and per-tensor-scaled FP8 on a GAN generator with
  instance-norm statistics is exactly the kind of place where activations
  overflow the ~448 dynamic range of E4M3 without per-layer calibration.
- The practical route to lower precision here is **TensorRT**: export ONNX
  (`torch.onnx.export` works on this static graph) and build an INT8- or
  FP8-calibrated engine. Expect maybe another ~1.5x over FP16 at 1080p, at
  the cost of a calibration pass and a second runtime to maintain.

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
