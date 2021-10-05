from __future__ import print_function

import numpy as np
from PIL import Image
import torchvision.transforms as transforms

mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
mean2 = [120.39586422, 115.59361427, 104.54012653]
std2 = [70.68188272, 68.27635443, 72.54505529]
normalize = transforms.Normalize(mean=mean, std=std)


def get_transform(config):
    size = config["input_size"]
    train_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            # lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ]
    )

    val_transforms = transforms.Compose(
        [
            lambda x: Image.fromarray(x),
            transforms.Resize((size, size)),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            # normalize
        ]
    )

    test_transforms = transforms.Compose(
        [
            lambda x: Image.fromarray(x),
            transforms.Resize((size, size)),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            # normalize
        ]
    )
    transform_dict = {
        "train": train_transforms,
        "valid": val_transforms,
        "test": test_transforms
    }
    return transform_dict
