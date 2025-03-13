from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from typing_extensions import Literal
import torch
from tqdm.auto import tqdm
import os
import cv2
import numpy as np
from .util import draw_random_shapes
import matplotlib.pyplot as plt
from PIL import Image

FROM_LABEL_TO_IDX = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19,
}

FROM_IDX_TO_LABEL = {v: k for k, v in FROM_LABEL_TO_IDX.items()}


class PascalVOC2007(Dataset):
    def __init__(
        self,
        image_set: Literal["train", "val", "trainval", "test"],
        skip_difficult: bool = True,
        transform=None,
    ):
        super().__init__()
        self.dataset = VOCDetection(
            root="data", year="2007", image_set=image_set, download=True
        )
        self.skip_difficult = skip_difficult
        self.transform = transform

        self.indices = []
        self.cache_name = f"./data/pascal_voc_2007_{image_set}{'_no_diff' if skip_difficult else ''}.pt"
        self.create_indices()

    def create_indices(self):
        if os.path.exists(self.cache_name):
            self.indices = torch.load(self.cache_name)
            return

        for idx in tqdm(range(len(self.dataset)), desc="Creating indices"):
            img, details = self.dataset[idx]
            objects = details["annotation"]["object"]
            cont = 0
            for obj in objects:
                if self.skip_difficult and int(obj["difficult"]) == 1:
                    continue

                cont += 1

                label = FROM_LABEL_TO_IDX[obj["name"]]
                label = torch.Tensor([label]).long().reshape(-1)

                self.indices.append((idx, cont, label))
        # Save the indices in cache
        torch.save(self.indices, self.cache_name)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx, cont, _ = self.indices[index]
        img, details = self.dataset[idx]
        objects = details["annotation"]["object"]
        n = 0
        for obj in objects:
            if self.skip_difficult and int(obj["difficult"]) == 1:
                continue

            n += 1
            if n == cont:
                bounding_box = obj["bndbox"]

                x_min = int(bounding_box["xmin"])
                y_min = int(bounding_box["ymin"])
                x_max = int(bounding_box["xmax"])
                y_max = int(bounding_box["ymax"])

                img_obj = img.crop((x_min, y_min, x_max, y_max))
                if self.transform:
                    img_obj = self.transform(img_obj)

                label = FROM_LABEL_TO_IDX[obj["name"]]
                label = torch.Tensor([label]).long().reshape(-1)

                return img_obj, label


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
    ):
        super().__init__()
        self.background_path = background_path
        self.image_transform = image_transform
        self.background_transform = background_transform
        self.mask_preprocess = mask_preprocess
        self.num_shapes_per_image = num_shapes_per_image
        self.size_range = size_range
        self.num_images = num_images
        self.initial_seed = hash(split) % 2**32

        # Read all the images in the background path
        self.background_images = []
        for root, _, files in os.walk(background_path):
            for file in files:
                if file.endswith(".jpg"):
                    self.background_images.append(os.path.join(root, file))

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        seed = self.initial_seed + index

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

        # Seed the random generator with the index
        np.random.seed(seed)

        label = np.random.randint(0, 3)

        img, mask = draw_random_shapes(
            background,
            shape_type=label,
            num_shapes=self.num_shapes_per_image,
            size_range=self.size_range,
            seed=seed,
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
