from torch.utils.data import Dataset
import torch
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    CenterCrop,
    Resize,
)
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class ResizeSmall(object):
    def __init__(self, smaller_size):
        assert isinstance(smaller_size, (int))
        self.smaller_size = smaller_size

    def __call__(self, image):
        h, w = image.shape[1], image.shape[2]  # image should be a tensor of shape (channels, height, width)

        # Figure out the necessary h/w.
        ratio = float(self.smaller_size) / min(h, w)
        new_h = int(h * ratio)
        new_w = int(w * ratio)
        image = Resize((new_h, new_w), antialias=True)(image)
        return image


class ImageNetDataset(Dataset):
    def __init__(
        self, dataset, img_size=224
    ):
        self.dataset = dataset
        small_size = img_size
        self.transform = Compose(
            [
                ToTensor(),
                ResizeSmall(small_size),
                CenterCrop((img_size, img_size)),
                Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        original_image = self.dataset[index]["image"].convert("RGB")
        label = self.dataset[index]["label"]
        image = self.transform(original_image)
        return (image, label)