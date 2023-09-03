from pathlib import Path
from typing import List, Optional, Callable

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split

from src.datasets.meta import PairedImageWithLightnessInput  # noqa: I900
from src.transforms import load_transforms  # noqa: I900
from src.utils.image import read_image_cv2  # noqa: I900
from src.datasets.lol import LOL  # noqa: I900
from src.datasets.fs_dark import FSD  # noqa: I900


class MixHQ(Dataset):
    SUBSETS = ['clic_resized', 'cocoHQ', 'ImageNetHQ', 'LOL_train', 'Inter4K_imgs']
    def __init__(
        self,
        root: Path,
        transform: Callable, 
        indices: Optional[List[int]] = None,
    ):
        self.transform = transform

        self.image_names = []
        for subset in MixHQ.SUBSETS:
            files = list((root / subset).glob('*'))
            images = [f for f in files if f.suffix in {'.jpg', '.png', '.jpeg', '.JPEG'}]
            self.image_names.extend(images)

        if indices is not None:
            self.image_names = [self.image_names[index] for index in indices]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> PairedImageWithLightnessInput:
        image = read_image_cv2(self.image_names[index])

        transformed = self.transform(light=image)
        image, target = transformed["image"], transformed["target"]

        source_lightness = transformed["source_lightness"]
        target_lightness = transformed["target_lightness"]

        return PairedImageWithLightnessInput(
            image=image,
            target=target,
            source_lightness=source_lightness,
            target_lightness=target_lightness,
        )


class MixHQDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.root = Path(config.path)
        self.config = config

        self.train_transform, self.test_transform = load_transforms(config.transform)

    def setup(self, stage: Optional[str] = None):
        self.train_ds = MixHQ(
            self.root,
            transform=self.train_transform,
        )

        self.val_ds = FSD(
            Path(self.config.val_path),
            pair_transform=self.test_transform,
            split='val',
        )

        self.test_ds = FSD(
            Path(self.config.val_path),
            pair_transform=self.test_transform,
            split='test',
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.config.batch_size,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
        )


    def predict_dataloader(self):
        return self.test_dataloader()
