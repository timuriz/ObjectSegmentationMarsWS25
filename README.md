# ObjectSegmentationMarsWS25

Semantic segmentation of Mars surface terrain for the **Machine Learning** course at [AAU Klagenfurt](https://www.aau.at/) (Winter Semester 2025).

The project trains a DeepLabV3 model to classify each pixel in Mars surface images into terrain types (sand, soil, bedrock). Predicted segmentation maps can support downstream tasks such as rover path planning over traversable terrain.

## Repository structure

| File | Description |
|------|-------------|
| [`path_finder.ipynb`](path_finder.ipynb) | Training pipeline: dataset loading, augmentations, model fine-tuning, and qualitative visualization |
| [`M_eval_final.ipynb`](M_eval_final.ipynb) | Evaluation on a held-out validation set with mIoU, Dice, and per-class metrics (Weights & Biases logging) |

## Segmentation classes

| Class ID | Label |
|----------|-------|
| 0 | ignore / background |
| 1 | sand |
| 2 | soil |
| 3 | bedrock |

Class 0 is excluded from metric computation via `ignore_index=0`.

## Model & training

- **Architecture:** DeepLabV3 with ResNet-50 backbone ([`torchvision.models.segmentation.deeplabv3_resnet50`](https://pytorch.org/vision/stable/models/deeplabv3.html)), pretrained on COCO; classifier head replaced for 4 output classes
- **Input size:** 768 × 768
- **Loss:** Cross-entropy
- **Optimizer:** Adam, learning rate `2e-4`
- **Scheduler:** `ReduceLROnPlateau` (`factor=0.5`, `patience=3`)
- **Training:** 50 epochs, batch size 2, 80/20 train/validation split
- **Augmentations (training):** horizontal/vertical flip, 90° rotation, affine transforms, random brightness/contrast

## Results (validation set)

Reported from [`M_eval_final.ipynb`](M_eval_final.ipynb):

| Metric | Value |
|--------|-------|
| mIoU | 0.717 |
| Mean Dice | 0.830 |
| Pixel accuracy | 0.898 |
| Mean class accuracy | 0.830 |

## Getting started

Both notebooks are designed to run in [Google Colab](https://colab.research.google.com/) with a GPU runtime.

1. Open a notebook via the **Open in Colab** badge at the top of the file.
2. Mount Google Drive and point the dataset paths to your local image/mask folders:
   - Training data: `images_last_clean_dataset` / `masks_dataset_last_clean_dataset`
   - Validation data: `images_val_last_clean_dataset` / `masks_val_last_clean_dataset`
3. For evaluation, place the trained checkpoint at the path configured in `M_eval_final.ipynb` (`checkpoints_best_new.pt`).

### Dependencies

```
torch
torchvision
albumentations
Pillow
numpy
matplotlib
wandb  # evaluation notebook only
```

Install in Colab with:

```python
!pip install albumentations wandb
```

## References

- [Torchvision DeepLabV3](https://pytorch.org/vision/stable/models/deeplabv3.html)
- [Albumentations](https://albumentations.ai/)
- [Weights & Biases](https://wandb.ai/)
