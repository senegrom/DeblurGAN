# Original GoPro demo frames

These are the original 1280x720 RGB frames behind the two animated examples in
the repository README. They replace neither the historical GIFs nor the
GIF-extracted examples; they provide artifact-free inputs for inference and
paired sharp ground truth for evaluation.

The second frame of each GIF is a DeblurGAN estimate, not the dataset target.
Its pre-GIF lossless file is not present in the upstream repository history,
so the sharp ground truth is kept explicitly separate rather than being
mislabelled as that estimate.

| README demo | GoPro test-set key | Blurred input | Sharp target |
| --- | --- | --- | --- |
| `animation3.gif` | `GOPR0881_11_01-000210` | `input/GOPR0881_11_01-000210.png` | `target/GOPR0881_11_01-000210.png` |
| `animation4.gif` | `GOPR0869_11_00-000034` | `input/GOPR0869_11_00-000034.png` | `target/GOPR0869_11_00-000034.png` |

The keys were identified by comparing the GIF input frames with every image in
the official GoPro test split. The best-match errors were `0.00003235` and
`0.00001924`; the next-best candidates were `0.02629` and `0.01248`,
respectively.

Source: [GOPRO_Large dataset](https://seungjunnah.github.io/Datasets/gopro.html),
released under CC BY 4.0 with *Deep Multi-Scale Convolutional Neural Network
for Dynamic Scene Deblurring* by Seungjun Nah, Tae Hyun Kim, and Kyoung Mu Lee.
The matching and export utilities are in `experiments/find_demo_sources.py` and
`experiments/export_lmdb_pairs.py`.
