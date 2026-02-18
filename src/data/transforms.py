"""
transforms.py

Purpose:
- Define image preprocessing and augmentation pipelines for training and validation.
- Train pipeline  : Resize → Augmentation → ToTensor → Normalize
- Val/Test pipeline: Resize → CenterCrop → ToTensor → Normalize

Normalization stats: ImageNet mean/std (compatible with EfficientNet-B0 pre-trained weights).
"""

from torchvision import transforms

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(size: int = 224) -> transforms.Compose:
    """
    Augmentation pipeline for training set.

    Steps:
        1. Resize shortest edge to (size + 32) to allow random crop headroom.
        2. RandomResizedCrop  — random scale/aspect-ratio crop, then resize to `size`.
        3. RandomHorizontalFlip  — p=0.5 (product packs can face either way).
        4. RandomRotation(15°)   — small tilt variation common in shelf/hand photos.
        5. ColorJitter           — brightness/contrast/saturation variation from
                                   different lighting conditions in Open Food Facts images.
        6. ToTensor              — HWC uint8 [0,255] → CHW float32 [0.0, 1.0].
        7. Normalize(ImageNet)   — align pixel distribution with pre-trained weights.
    """
    return transforms.Compose(
        [
            transforms.Resize(size + 32),                    # step 1
            transforms.RandomResizedCrop(                    # step 2
                size,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.33),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),          # step 3
            transforms.RandomRotation(degrees=15),           # step 4
            transforms.ColorJitter(                          # step 5
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),                           # step 6
            transforms.Normalize(                            # step 7
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_val_transforms(size: int = 224) -> transforms.Compose:
    """
    Deterministic pipeline for validation and test sets.

    Steps:
        1. Resize shortest edge to (size + 32).
        2. CenterCrop to `size`  — no random crop for reproducible evaluation.
        3. ToTensor.
        4. Normalize(ImageNet).
    """
    return transforms.Compose(
        [
            transforms.Resize(size + 32),                    # step 1
            transforms.CenterCrop(size),                     # step 2
            transforms.ToTensor(),                           # step 3
            transforms.Normalize(                            # step 4
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )