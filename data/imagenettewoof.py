from torch.utils.data import Dataset, ConcatDataset
from .imagenette import Imagenette
from .imagewoof import Imagewoof


# Custom wrapper to offset labels in ImageWoof
class OffsetDataset(Dataset):
    def __init__(self, dataset, label_offset):
        self.dataset = dataset
        self.label_offset = label_offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label + self.label_offset  # Offset label


def imagenettewoof(
    root: str = "data",
    split: str = "train",
    size: str = "full",
    download: bool = False,
    transform=None,
):
    imagenette = Imagenette(root, split, size, download, transform)
    imagewoof = Imagewoof(root, split, size, download, transform)

    # Offset ImageWoof labels
    imagewoof = OffsetDataset(imagewoof, 10)

    merged_dataset = ConcatDataset([imagenette, imagewoof])

    return merged_dataset
