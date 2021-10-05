import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .transform_list import get_transform
import os

class SkinDataset(Dataset):
    def __init__(self, partition='train', transform=None, name="HAM1000", input_dir=""):
        super(Dataset, self).__init__()
        self.partition = partition
        self.name = name

        if transform is None:
            return
        else:
            self.transform = transform

        if os.path.exists(f'{input_dir}/{self.name}/{self.partition}_image.npy'):
            self.train_img = f'{input_dir}/{self.name}/{self.partition}_image.npy'
        else:
            print(f'{input_dir}/{self.name}/{self.partition}_image.npy is not exist !')

        if os.path.exists(f'{input_dir}/{self.name}/{self.partition}_y.npy'):
            self.train_y = f'{input_dir}/{self.name}/{self.partition}_y.npy'
        else:
            print(f'{input_dir}/{self.name}/{self.partition}_y.npy is not exist !')

        self.data = {}
        self.labels = []
        self.imgs = np.load(self.train_img)  # It has been normalized before
        self.y = np.load(self.train_y)
        print('{number} {partition} images'.format(number=self.y.shape[0], partition=self.partition))
        # array->tensor
        self.y = torch.tensor(self.y)
        # one hot->value
        self.labels = torch.topk(self.y, 1)[1].squeeze(1)
        self.labels = np.asarray(self.labels)

    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)
        target = self.labels[item]
        return img, target, item

    def __len__(self):
        return len(self.labels)


def get_dataloader(config):
    transform_dict = get_transform(config)
    train_data = SkinDataset(partition="train",
                             transform=transform_dict["train"],
                             name=config["data_name"],
                             input_dir=config["input_dir"])
    valid_data = SkinDataset(partition="valid",
                             transform=transform_dict["valid"],
                             name=config["data_name"],
                             input_dir=config["input_dir"])
    test_data = SkinDataset(partition="test",
                            transform=transform_dict["test"],
                            name=config["data_name"],
                            input_dir=config["input_dir"])

    train_loader = DataLoader(dataset=train_data, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=config["batch_size"], shuffle=True)

    dataloader = {
        "train": train_loader,
        "val": valid_loader,
        "test": test_loader
    }
    return dataloader
