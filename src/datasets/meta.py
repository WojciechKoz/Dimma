from typing import NewType, TypedDict

import torch

Image = NewType("Image", torch.Tensor)


class PairedImageInput(TypedDict):
    image: Image
    target: Image


class PairedImageWithLightnessInput(TypedDict):
    image: Image
    target: Image
    source_lightness: torch.Tensor
    target_lightness: torch.Tensor
