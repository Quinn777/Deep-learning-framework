from __future__ import print_function

import torchvision.transforms as transforms


def get_transform(config):
    size = config["input_size"]

    default_composed = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.CenterCrop(256),
                                           transforms.RandomCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_transforms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config["normalize"][0], config["normalize"][1])
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(config["normalize"][0], config["normalize"][1])
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(config["normalize"][0], config["normalize"][1])
        ]
    )
    transform_dict = {
        "train": train_transforms,
        "valid": val_transforms,
        "test": test_transforms
    }
    return transform_dict
