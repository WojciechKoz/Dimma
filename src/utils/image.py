from pathlib import Path
from typing import Union

import cv2
import numpy as np


def read_image_cv2(path: Union[str, Path]) -> np.ndarray:
    """Read an image from a path."""
    image = cv2.imread(str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
