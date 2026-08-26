# DeblurGAN
[arXiv Paper Version](https://arxiv.org/pdf/1711.07064.pdf)

Pytorch implementation of the paper DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks.

Our network takes blurry image as an input and procude the corresponding sharp estimate, as in the example:
<img src="images/animation3.gif" width="400px"/> <img src="images/animation4.gif" width="400px"/>

The original 1280x720 GoPro test-set frames behind these GIF demos, without
GIF palette and scaling artifacts, are in [`examples/gopro`](examples/gopro/):

- `animation3.gif`: `GOPR0881_11_01-000210`
- `animation4.gif`: `GOPR0869_11_00-000034`

Each ID has the original blurred frame under `input/` and its paired sharp
ground truth under `target/`.


The model we use is Conditional Wasserstein GAN with Gradient Penalty + Perceptual loss based on VGG-19 activations. Such architecture also gives good results on other image-to-image translation problems (super resolution, colorization, inpainting, dehazing etc.)

## How to run

### Prerequisites
- NVIDIA GPU + CUDA CuDNN (CPU untested, feedback appreciated)
- Pytorch

The **official pretrained generator weights are included** in this fork at
`checkpoints/official/latest_net_G.pth` (45.6 MB, the true two-conv paper
architecture). The historical Google Drive / Dropbox links from the upstream
README are dead; this copy was recovered in Aug 2026 from a 2018 fork
([haozhe15/DeblurGAN](https://github.com/haozhe15/DeblurGAN), committed
2018-11-12 while the official link was still live) and verified against the
key layout reported in
[upstream issue #145](https://github.com/KupynOrest/DeblurGAN/issues/145).
`deblur_fast.py` uses them by default. Note: the checkpoint upstream ships at
`checkpoints/experiment_name/` was trained on a buggy one-conv ResnetBlock,
barely deblurs (below the blurry-input baseline on the GoPro test average),
and has been **removed from this fork** so nobody uses it by accident — see
SPEED.md.
To test a model put your blurry images into a folder and run:
```bash
python test.py --dataroot /.path_to_your_data --model test --dataset_mode single --learn_residual
```
## Data
Download dataset for Object Detection benchmark from [Google Drive](https://drive.google.com/file/d/1CPMBmRj-jBDO2ax4CxkBs9iczIFrs8VA/view?usp=sharing)

## Train

If you want to train the model on your data run the following command to create image pairs:
```bash
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
```
And then the following command to train the model

```bash
python train.py --dataroot /.path_to_your_data --learn_residual --resize_or_crop crop --fineSize CROP_SIZE (we used 256)
```

## Other Implementations

[Keras Blog](https://blog.sicara.com/keras-generative-adversarial-networks-image-deblurring-45e3ab6977b5)

[Keras Repository](https://github.com/RaphaelMeudec/deblur-gan)



## Citation

If you find our code helpful in your research or work please cite our paper.

```
@article{DeblurGAN,
  title = {DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks},
  author = {Kupyn, Orest and Budzan, Volodymyr and Mykhailych, Mykola and Mishkin, Dmytro and Matas, Jiri},
  journal = {ArXiv e-prints},
  eprint = {1711.07064},
  year = 2017
}
```

## Acknowledgments
Code borrows heavily from [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). The images were taken from GoPRO test dataset - [DeepDeblur](https://github.com/SeungjunNah/DeepDeblur_release)


