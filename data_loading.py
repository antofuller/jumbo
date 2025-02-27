from torch.utils.data import Dataset
import torch
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    CenterCrop,
    RandAugment,
    RandomHorizontalFlip,
)
from timm.data import create_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
from augment import ResizeSmall, new_data_aug_generator


mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_deit_finetune_transforms(img_size):
    transform = create_transform(
        input_size=img_size,
        is_training=True,
        color_jitter=0.3,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation='bicubic',
        re_prob=0.0,
        re_mode='pixel',
        re_count=1,
    )
    return transform


class ImageNetDataset(Dataset):
    def __init__(
        self, dataset, do_augment, augment_type="3aug", img_size=224
    ):
        self.dataset = dataset

        assert augment_type in [
            "3aug",
            "randaug",
            "deit_ft",
        ], f"augment_type must be either 3aug, deit_ft, or randaug, not: {augment_type}"

        if not do_augment:
            small_size = img_size
            self.transform = Compose(
                [
                    ToTensor(),
                    ResizeSmall(small_size),
                    CenterCrop((img_size, img_size)),
                    Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
                ]
            )

        else:
            if augment_type == "deit_ft":
                self.transform = get_deit_finetune_transforms(img_size)
            else:
                if augment_type == "3aug":
                    first_tfl = new_data_aug_generator(
                        simple_random_crop=False, color_jitter=0.3, img_size=img_size
                    )
                elif augment_type == "randaug":
                    scale = (0.08, 1.0)
                    interpolation = "bicubic"
                    first_tfl = [
                        RandomResizedCropAndInterpolation(
                            img_size, scale=scale, interpolation=interpolation
                        ),
                        RandomHorizontalFlip(),
                        RandAugment(2, 15),
                    ]

                final_tfl = [
                    ToTensor(),
                    Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
                ]
                self.transform = Compose(first_tfl + final_tfl)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        original_image = self.dataset[index]["image"].convert("RGB")
        label = self.dataset[index]["label"]
        image = self.transform(original_image)
        return (image, label)