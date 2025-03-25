import os
import hashlib
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .util import draw_random_shapes


def deterministic_walk(directory):
    for root, dirs, files in sorted(
        os.walk(directory), key=lambda x: x[0]
    ):  # Sort by root directory name
        dirs.sort()  # Sort directories in-place
        files.sort()  # Sort files in-place
        yield root, dirs, files


class SynteticFigures(Dataset):
    def __init__(
        self,
        background_path,
        num_shapes_per_image=10,
        size_range=(20, 100),
        num_images=1000,
        split="train",
        image_transform=None,
        background_transform=None,
        mask_preprocess=None,
        num_other_shapes=0,
    ):
        super().__init__()
        self.background_path = background_path
        self.image_transform = image_transform
        self.background_transform = background_transform
        self.mask_preprocess = mask_preprocess
        self.num_shapes_per_image = num_shapes_per_image
        self.size_range = size_range
        self.num_images = num_images
        self.num_other_shapes = num_other_shapes

        def hash_string(s: str) -> int:
            return int(hashlib.sha256(s.encode()).hexdigest(), 16) % 2**32

        self.initial_seed = hash_string(split)

        # Read all the images in the background path
        self.background_images = []
        for root, _, files in deterministic_walk(self.background_path):
            for file in files:
                if file.endswith(".jpg"):
                    self.background_images.append(os.path.join(root, file))

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        seed = self.initial_seed + index
        # Seed the random generator with the index
        np.random.seed(seed)
        torch.manual_seed(seed)

        if index >= self.num_images:
            raise IndexError("Index out of bounds")

        background = cv2.imread(
            self.background_images[index % len(self.background_images)]
        )
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

        background = torch.Tensor(background).type(torch.uint8).permute(2, 0, 1)
        ###############################################
        # background = torch.zeros_like(background)
        ###############################################

        if self.background_transform:
            background = self.background_transform(background)

        # Set the background back to numpy array
        background = background.permute(1, 2, 0).numpy().astype(np.int16)

        label = np.random.randint(0, 3)

        img, mask = draw_random_shapes(
            background,
            shape_type=label,
            num_shapes=self.num_shapes_per_image,
            size_range=self.size_range,
            seed=seed,
            num_other_shapes=self.num_other_shapes,
        )

        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)
        # img = np.transpose(img, (2, 0, 1))

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        if self.image_transform:
            img = self.image_transform(img)

        if self.mask_preprocess:
            mask = self.mask_preprocess(mask)

        return img, mask, label
