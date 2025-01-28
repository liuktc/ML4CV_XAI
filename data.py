import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torchvision.datasets import VOCDetection
from typing_extensions import Literal

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


class PascalVOC2007_ImagenetLabels(IterableDataset):
    def __init__(
        self,
        image_set: Literal["train", "val", "trainval", "test"],
        skip_difficult: bool = True,
    ):
        super().__init__()
        self.dataset = VOCDetection(
            root="data", year="2007", image_set=image_set, download=True
        )
        self.skip_difficult = skip_difficult

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for idx in range(len(self.dataset)):
            img, details = self.dataset[idx]
            objects = details["annotation"]["object"]
            for obj in objects:
                if self.skip_difficult and int(obj["diffucult"]) == 1:
                    continue

                bounding_box = obj["bndbox"]

                x_min = int(bounding_box["xmin"])
                y_min = int(bounding_box["ymin"])
                x_max = int(bounding_box["xmax"])
                y_max = int(bounding_box["ymax"])

                img_obj = img.crop((x_min, y_min, x_max, y_max))

                label = FROM_LABEL_TO_IDX[obj["name"]]

                yield img_obj, label
