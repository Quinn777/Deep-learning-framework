import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import pandas as pd
from PIL import ImageFile
import torchvision.transforms as transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True


class HAM10000Dataset(Dataset):
    def __init__(self, opt, partition):
        self.opt = opt
        self.partition = partition
        if self.partition == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.opt.input_size, self.opt.input_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.opt.input_size, self.opt.input_size)),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )
        base_dir = os.path.join(opt.data_dir, opt.data_name)
        self.df = pd.read_csv(os.path.join(base_dir, f"{partition}_mapping.csv"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            x = self.transform(x)
        return x, y


class HAM10000:
    def __init__(self, opt):
        self.opt = opt
        self.train_data = HAM10000Dataset(self.opt, "train")
        self.test_data = HAM10000Dataset(self.opt, "test")

    def data_loaders(self):
        train_loader = DataLoader(dataset=self.train_data,
                                  batch_size=self.opt.batch_size,
                                  shuffle=True,
                                  num_workers=self.opt.workers,
                                  pin_memory=self.opt.pin_memory)
        test_loader = DataLoader(dataset=self.test_data,
                                 batch_size=self.opt.test_batch_size,
                                 shuffle=True,
                                 num_workers=self.opt.workers,
                                 pin_memory=self.opt.pin_memory)

        return train_loader, test_loader
