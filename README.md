# ThinkLIF

ThinkLIF is an auxiliary-representation-guided framework for nuclear segmentation in Ki67-stained immunohistochemistry (IHC) images.

The model first generates four stain-like auxiliary representations from an IHC input image and then concatenates these auxiliary outputs with the original IHC image for final nuclear segmentation. The segmentation head uses ConvNeXt-style multi-scale refinement, GatedFusion, and a FiLM-based AuxSkipAdapter.

This repository contains the PyTorch implementation, environment specification, inference instructions, and pretrained DeepLIIF weights used for the ThinkLIF experiments.

## Overview

ThinkLIF follows a translation-plus-segmentation pipeline:

1. Four generators map the IHC input to stain-like auxiliary outputs:
   - Hematoxylin-like output
   - DAPI-like output
   - LAP2-like output
   - Ki67/marker-like output
2. The four RGB auxiliary outputs are concatenated with the original RGB IHC input.
3. The resulting 15-channel tensor is passed to the `unet_plus` segmentation head.
4. The segmentation head predicts a binary nuclear mask.

The current implementation is primarily evaluated for nuclear segmentation. The auxiliary outputs are used as stain-like representations to support segmentation and should not be interpreted as fully validated quantitative mpIF substitutes without additional validation.

## Repository Structure

```text
ThinkLIF/
+-- train.py                         # Training entry point
+-- test.py                          # Inference and post-processing entry point
+-- serialize.py                     # Utility script for model packaging/serialization
+-- environment.yaml                 # Conda environment used in experiments
+-- checkpoints/                     # Local checkpoints, ignored by git
+-- Model/
    +-- data/
    |   +-- aligned_dataset.py       # Horizontally concatenated multi-panel dataset loader
    |   +-- base_dataset.py
    +-- models/
    |   +-- ParsiLIF_model.py        # Translation + segmentation training logic
    |   +-- segheads.py              # ThinkLIF unet_plus segmentation head
    |   +-- down_up.py               # ConvNeXt-style encoder/decoder blocks
    |   +-- networks.py              # Generator, discriminator, and loss definitions
    +-- metrics/
    |   +-- ComputeStatistics.py
    |   +-- PostProcess_Metrics.py
    |   +-- Segmentation_Metrics.py
    +-- util/
        +-- postprocessing.py
        +-- visualizer.py
        +-- util.py
```

## Installation

Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate deepliif_env
```

If your CUDA/PyTorch version differs from the environment file, install a compatible PyTorch build for your system first, then install the remaining Python dependencies listed in `environment.yaml`.

## Environment Details

The experiments were conducted with Python 3.9, PyTorch 2.8.0, torchvision 0.23.0, CUDA 12.x, OpenCV 4.8.1, scikit-image 0.21.0, SciPy 1.10.1, NumPy 1.24.4, and Pillow 10.4.0. The complete conda environment used in our experiments is provided in `environment.yaml`.

## Data Format

The default dataset mode is `aligned`. Each sample is a single horizontally concatenated RGB image containing six panels in the following order:

```text
IHC | Hematoxylin | DAPI | LAP2 | Ki67/Marker | Segmentation overlay
```

For 512 x 512 panels, the full sample size is 3072 x 512.

The expected directory layout is:

```text
Dataset/
+-- train/
|   +-- sample_001.png
|   +-- sample_002.png
|   +-- ...
+-- test/
    +-- sample_101.png
    +-- sample_102.png
    +-- ...
```

The loader uses:

- Panel 1 as the IHC input.
- Panels 2-5 as auxiliary generation targets.
- Panel 6 as the segmentation overlay.

For the segmentation overlay, red pixels are treated as Ki67-positive nuclei and blue pixels are treated as Ki67-negative nuclei. The semantic segmentation target is the union of red and blue nuclei.

## Datasets Used in the Paper

The datasets are not redistributed in this repository. Please obtain them from their original sources.

In our experiments, we used:

- DeepLIIF: 1,264 co-registered IHC/mpIF samples.
  - 575 training samples
  - 91 validation samples
  - 598 test samples
- Processed BC-DeepLIIF subset derived from BCData:
  - 451 aligned 512 x 512 samples from 41 BCData images
  - 385 training samples
  - 66 held-out evaluation samples

## Training

Run training from the repository root:

```bash
cd ThinkLIF
python train.py \
  --dataroot /path/to/Dataset \
  --name unet_plus_deepliif \
  --model ParsiLIF \
  --seghead unet_plus \
  --modalities-no 4 \
  --net-g resnet_9blocks,resnet_9blocks,resnet_9blocks,resnet_9blocks \
  --batch-size 1 \
  --load-size 512 \
  --crop-size 512 \
  --n-epochs 20 \
  --n-epochs-decay 20 \
  --lr-g 0.0002 \
  --lr-d 0.0002 \
  --save-epoch-freq 5
```

Important notes:

- `--seghead unet_plus` should be specified explicitly.
- `--modalities-no 4` corresponds to the four auxiliary targets: Hematoxylin, DAPI, LAP2, and Ki67/marker.
- Checkpoints and training visualizations are saved under:

```text
checkpoints/<experiment_name>/
```

To train on CPU, use:

```bash
--gpu-ids -1
```

## Inference

The inference script loads the saved training options from the checkpoint directory and writes results to a new prediction directory.

```bash
cd ThinkLIF
python test.py \
  --dataroot /path/to/Dataset \
  --name unet_plus_deepliif \
  --checkpoints_dir ./checkpoints \
  --gpu_ids 0 \
  --num_test 10000
```

Expected checkpoint layout:

```text
checkpoints/
+-- unet_plus_deepliif/
    +-- latest/
    |   +-- train_opt.txt
    |   +-- latest_net_G1.pth
    |   +-- latest_net_G2.pth
    |   +-- latest_net_G3.pth
    |   +-- latest_net_G4.pth
    |   +-- latest_net_S.pth
    +-- ...
```

Inference outputs are saved to:

```text
/path/to/Dataset_pred_<experiment_name>/test_latest/images/
```

Typical output files include:

- `*_real_A.png`: input IHC image
- `*_fake_B_1.png` to `*_fake_B_4.png`: generated auxiliary outputs
- `*_real_B_5.png`: ground-truth segmentation target
- `*_fake_B_5.png`: predicted segmentation probability/mask
- `*_SegOverlaid.png`: overlay visualization after post-processing
- `*_SegRefined.png`: post-processed segmentation result

## Pretrained Weights

The pretrained ThinkLIF weights trained on the DeepLIIF dataset are available from GitHub Releases:

- [ThinkLIF_pretrained_deepliif.zip](https://github.com/yangrenyu2002/ThinkLIF/releases/download/pretrained-weights-v1/ThinkLIF_pretrained_deepliif.zip)

After downloading and extracting the archive into the repository root, the expected layout is:

```text
checkpoints/
+-- thinklif_deepliif/
    +-- latest/
        +-- train_opt.txt
        +-- latest_net_G1.pth
        +-- latest_net_G2.pth
        +-- latest_net_G3.pth
        +-- latest_net_G4.pth
        +-- latest_net_S.pth
```

Run inference with the pretrained DeepLIIF weights:

```bash
python test.py \
  --dataroot /path/to/Dataset \
  --name thinklif_deepliif \
  --checkpoints_dir ./checkpoints \
  --gpu_ids 0 \
  --num_test 10000
```

## Metrics and Post-processing

`test.py` calls the post-processing pipeline in `Model/metrics/PostProcess_Metrics.py` after inference. The metric utilities support pixel-level and instance-level evaluation, including:

- Pixel Accuracy
- Dice
- IoU
- AJI
- IHC Quantification Difference, defined as the absolute difference in positive-cell proportion

## Model Components

Key implementation files:

- `Model/models/ParsiLIF_model.py`
  - Defines the four auxiliary generators, discriminators, segmentation branch, and training losses.
- `Model/models/segheads.py`
  - Defines the `unet_plus` segmentation head with GatedFusion and AuxSkipAdapter.
- `Model/data/aligned_dataset.py`
  - Loads and splits the horizontally concatenated six-panel image samples.

## Reproducibility Notes

The main experimental setting used:

- Optimizer: Adam
- Generator/discriminator learning rate: `2e-4`
- Segmentation-head learning rate: `2e-4`
- Batch size: `1`
- Image size: `512 x 512`
- Training schedule: 20 epochs + 20 epochs linear decay
- Generator architecture: `resnet_9blocks`
- Segmentation head: `unet_plus`
- Auxiliary modality loss weights: `[0.25, 0.25, 0.25, 0.25]`

This repository provides the source code, environment file, data-format description, training command, inference command, evaluation utilities, and pretrained DeepLIIF weight download link required to reproduce the reported workflow.

## Citation

If you use this code, please cite the associated manuscript once available.

```bibtex
@article{thinklif,
  title   = {ThinkLIF: Multi-Scale Auxiliary Representation Guided Nuclear Segmentation in IHC Histopathology},
  author  = {Lv, Zongnan and Yang, Renyu and Shao, Chunxue and Yu, Qi and Wei, Meiqi and Yang, Minzhuo and Yang, Guang and Wang, Ziheng},
  journal = {Manuscript under review},
  year    = {2026}
}
```

## License

This project is released under the MIT License. See the `LICENSE` file for details.
