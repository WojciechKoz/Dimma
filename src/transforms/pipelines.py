import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from src.transforms.pair_transform import PairedTransformForDimma  # noqa: I900
from src.transforms.mdn_transform import MDNTransform  # noqa: I900

DIMMA_FINETUNE = 'dimma_finetune'
MDN_DIM_FOR_DIMMA = 'mdn'


def load_transforms(transform_config):
    print(type(transform_config))
    if transform_config.name == DIMMA_FINETUNE:
        transforms = (
            PairedTransformForDimma(
                flip_prob=transform_config.flip_prob,
                crop_size=transform_config.image_size,
            ),
            PairedTransformForDimma(test=True),
        )
    elif transform_config.name == MDN_DIM_FOR_DIMMA:
        train_transform = A.Compose(
            [
                A.RandomCrop(transform_config.image_size, transform_config.image_size),
                A.HorizontalFlip(),
                A.ISONoise(),
            ]
        )

        transforms = (
            MDNTransform(
                dim_factor=transform_config.dim_factor,
                path=transform_config.path,
                transforms=train_transform,
                mdn=transform_config.mdn,
            ),
            PairedTransformForDimma(test=True),
        )
    else:
        raise ValueError(f"Transform {transform_config.name} not found.")

    return transforms

