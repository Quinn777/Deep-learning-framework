import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .transform_list import get_transform
import os
from PIL import Image
import pandas as pd
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class HAM10000Dataset(Dataset):
    def __init__(self, base_dir, partition, transform):
        self.transform = transform
        self.df = pd.read_csv(os.path.join(base_dir, f"{partition}_mapping.csv"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            x = self.transform(x)
        return x, y


def get_dataloader(config):
    transform_dict = get_transform(config)
    train_data = HAM10000Dataset(base_dir=config["input_dir"],
                                 partition="train",
                                 transform=transform_dict["train"])
    valid_data = HAM10000Dataset(base_dir=config["input_dir"],
                                 partition="valid",
                                 transform=transform_dict["valid"])
    test_data = HAM10000Dataset(base_dir=config["input_dir"],
                                partition="test",
                                transform=transform_dict["test"])

    train_loader = DataLoader(dataset=train_data,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              num_workers=config["workers"],
                              pin_memory=config["pin_memory"])
    valid_loader = DataLoader(dataset=valid_data,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              num_workers=config["workers"],
                              pin_memory=config["pin_memory"])
    test_loader = DataLoader(dataset=test_data,
                             batch_size=config["batch_size"],
                             shuffle=True,
                             num_workers=config["workers"],
                             pin_memory=config["pin_memory"])

    dataloader = {
        "train": train_loader,
        "val": valid_loader,
        "test": test_loader
    }
    return dataloader
