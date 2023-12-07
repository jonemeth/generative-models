import os
from typing import Optional

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import CSVLogger


class MyLogger(CSVLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_file = None

    def log_text(self, text: str):
        if self.text_file is None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.text_file = open(self.log_dir + "/log.txt", "w+t", encoding="utf-8")
        print(text, flush=True)
        self.text_file.write(text + "\n")
        self.text_file.flush()


def create_grid_image(images: torch.Tensor, pad: int = 2, n_rows: Optional[int] = None, transpose: bool = False) -> torch.Tensor:
    n_images = images.shape[0]

    n_rows = n_rows or int(n_images**0.5)
    n_cols = n_images // n_rows

    # pylint: disable=not-callable
    images = F.pad(images, (pad, pad, pad, pad, 0, 0, 0, 0))

    images = images.reshape(n_rows, n_cols, *images.shape[1:])
    if transpose:
        images = images.permute(1, 0, 2, 3, 4)
        n_rows, n_cols = n_cols, n_rows
    images = images.permute(0, 3, 1, 4, 2).contiguous()
    images = images.view(n_rows*images.shape[1], n_cols*images.shape[3], images.shape[4])
    images = torch.squeeze(images)

    return images

def save_image(image: torch.Tensor, filename: str) -> None:
    grid = np.uint8(255*image.cpu().numpy())
    im = Image.fromarray(grid)
    if im.mode != "RGB":
        im = im.convert("RGB")
    im.save(filename)
