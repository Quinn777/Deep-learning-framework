from __future__ import print_function

from src.tool_function import *
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from vit import ViT
from src.data.dataloader import get_dataloader
import torchvision.models as models
from config import get_config
from tqdm import tqdm
from torch.autograd import Variable


class Trainer:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
        print(f"Torch: {torch.__version__}")

        self.config = get_config()
        self.device = torch.device(f"cuda:{self.config['num_gpu']}" if torch.cuda.is_available() else "cpu")
        seed_everything(self.config["seed"])

        # Initialize model
        self.model = self.get_model(self.config)
        self.model.to(self.device)

        # Best model message
        self.best_model_wts = ""
        self.best_acc = 0.0
        self.best_model_path = ""

        # Initialize dataloader
        self.dataloader = get_dataloader(self.config)

        # Loss function
        weights = [0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2]
        class_weights = torch.FloatTensor(weights).cuda()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"],
                                    weight_decay=self.config["weight_decay"], betas=(0.9, 0.99))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.config["decay_step"],
                                                         gamma=self.config["gamma"])

        torch.backends.cudnn.benchmark = True

    def train(self):
        since = time.time()

        # Start training
        for epoch in range(self.config["epoch"]):
            self.train_per_epoch(epoch)
            self.scheduler.step()

        # Finish training
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(self.best_acc))

    def train_per_epoch(self, epoch):
        print(self.optimizer.state_dict()['param_groups'][0]['lr'])

        begin_time = time.time()
        train_acc, train_loss, val_acc, val_loss = 0, 0, 0, 0

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_correct = 0.0

            if phase == 'train':
                self.model.train()  # Set model to training mode
            else:
                self.model.eval()  # Set model to evaluate mode

            # Iterate over data.
            pbar = tqdm(self.dataloader[phase])
            for inputs, labels in pbar:
                pbar.set_description(f"{phase}:")

                inputs = inputs.cuda()
                labels = labels.cuda()

                # Zero the parameter gradients
                if phase == "train":
                    self.optimizer.zero_grad()

                # Forward
                outputs = self.model(inputs)
                # Predict
                _, preds = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)  # The loss value here is the average of a batch

                # Backward + optimize + lr decay, only in training phase
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()
                # Statistics
                running_loss += loss.item()
                running_correct += torch.sum(preds == labels.data).to(torch.float32)

            # Calculate metrics
            epoch_loss = running_loss / len(self.dataloader[phase])  # sum(batch-average loss) / len(train loader)
            epoch_acc = running_correct / len(self.dataloader[phase].dataset)
            if phase == "train":
                train_acc = epoch_acc
                train_loss = epoch_loss
            else:
                val_acc = epoch_acc
                val_loss = epoch_loss

        print(
            f"Epoch : {epoch + 1} - train_loss : {train_loss:.4f} - train_acc: {train_acc:.4f} - val_loss : "
            f"{val_loss:.4f} - val_acc: {val_acc:.4f} - time:{time.time() - begin_time}\n")

        # If save model
        if phase == 'val' and epoch_acc > self.best_acc:
            self.best_acc = epoch_acc
            self.save_model(epoch)

    def save_model(self, epoch):
        # Make dir
        # output/HAM10000/mobilenetv2
        if not os.path.exists(f'output/{self.config["data_name"]}/{self.config["model_name"]}'):
            os.makedirs(f'output/{self.config["data_name"]}/{self.config["model_name"]}')

        # Delete the best model previously saved
        if os.path.exists(self.best_model_path):
            os.remove(self.best_model_path)

        # Save the new best model
        self.best_model_path = f'output/{self.config["data_name"]}/{self.config["model_name"]}/' \
                               f'lr{self.config["lr"]}_gamma{self.config["gamma"]}_batchsize' \
                               f'{self.config["batch_size"]}_epoch{epoch}_acc{self.best_acc}.pkl'

        torch.save(self.model, self.best_model_path)
        self.best_model_wts = self.model.state_dict()

    @staticmethod
    def get_model(config):
        new_model = ""
        if config["model_name"] == "mobilenetv2":
            new_model = models.mobilenet_v2(pretrained=False,
                                            num_classes=config["num_classes"],
                                            input_size=config["input_size"])
        elif config["model_name"] == "resnet50":
            new_model = models.resnet50(pretrained=False, num_classes=config["num_classes"])
            # new_model.fc = nn.Linear(in_features=2048, out_features=config["num_classes"])
        elif config["model_name"] == "vit":
            new_model = ViT(
                image_size=config["input_size"],
                patch_size=32,
                num_classes=config["num_classes"],
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1
            )
        return new_model


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
