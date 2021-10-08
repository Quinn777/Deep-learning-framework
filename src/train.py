from .utils.utils import *
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from .models.vit import ViT
from .Dataloader.dataloader import get_dataloader
import torchvision.models as models
from tqdm import tqdm
import datetime


class Trainer:
    def __init__(self, config):
        self.config = config

        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
        print(f"Torch: {torch.__version__}")
        torch.backends.cudnn.benchmark = True
        seed_everything(self.config["seed"])

        # Initialize model and gpu
        self.model = self.get_model(self.config)
        self.device = torch.device(f"cuda:{self.config['gpu_num']}" if torch.cuda.is_available() else "cpu")
        if config["gpu_parallel"] and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(0, torch.cuda.device_count())))
        self.model.to(self.device)

        # Best model message
        self.best_model = self.model
        self.best_model_wts = ""
        self.best_model_path = ""

        # Initialize dataloader
        self.dataloader = get_dataloader(self.config)

        # Loss function
        class_weights = torch.FloatTensor(self.config["class_weight"]).cuda()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"],
                                    weight_decay=self.config["weight_decay"], betas=(0.9, 0.99))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.config["decay_step"],
                                                         gamma=self.config["gamma"])

    def run(self):
        best_val_acc, best_val_loss = 0, 0
        init_time = time.time()

        for epoch in range(self.config["epoch"]):
            epoch_init_time = time.time()
            # 1. Train a model
            self.model, train_acc, train_loss = self.train(input_model=self.model,
                                                           dataloader=self.dataloader["train"],
                                                           optimizer=self.optimizer,
                                                           criterion=self.criterion)
            # 2. Valuate the model
            self.model, val_acc, val_loss = self.test(input_model=self.model,
                                                      criterion=self.criterion,
                                                      dataloader=self.dataloader["val"],
                                                      mode="Valid")
            self.scheduler.step()

            # 3. Save the model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                self.best_model = self.model
                self.best_model_path, self.best_model_wts = self.save_model(self.config,
                                                                            self.model,
                                                                            self.best_model_path,
                                                                            epoch + 1,
                                                                            best_val_acc,
                                                                            best_val_loss)
            print(
                f"Epoch : {epoch + 1} - train_loss : {train_loss:.4f} - train_acc: {train_acc:.4f} - val_loss : "
                f"{val_loss:.4f} - val_acc: {val_acc:.4f} - time:{time.time() - epoch_init_time}\n")

        # 4. Test the best model
        self.model, test_acc, test_loss = self.test(self.best_model, self.criterion, self.dataloader["test"], "test")

        # 5. Print and finish
        time_elapsed = time.time() - init_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print(f'Final test: best val_acc:{best_val_acc} - best val_loss:{best_val_acc} - best test_acc:{test_acc} '
              f'- best test_loss:{test_loss}')

        return self.best_model

    @staticmethod
    def train(input_model, dataloader, optimizer, criterion):
        # print(self.optimizer.state_dict()['param_groups'][0]['lr'])
        sum_loss = 0.0
        sum_correct = 0.0

        # Set model to training mode
        input_model.train()

        # Iterate over data.
        pbar = tqdm(dataloader)
        for inputs, labels in pbar:
            pbar.set_description(f"Train")

            inputs = inputs.cuda()
            labels = labels.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward
            outputs = input_model(inputs)
            # Predict
            _, preds = torch.max(outputs.data, 1)
            # The loss value here is the average of a batch
            loss = criterion(outputs, labels)
            # Backward + Optimize
            loss.backward()
            optimizer.step()

            # Statistics
            sum_loss += loss.item()
            sum_correct += torch.sum(preds == labels.data).to(torch.float32)

        # Calculate metrics
        epoch_loss = sum_loss / len(dataloader)
        epoch_acc = sum_correct / len(dataloader.dataset)
        return input_model, epoch_acc, epoch_loss

    @staticmethod
    def test(input_model, criterion, dataloader, mode):
        sum_loss = 0.0
        sum_correct = 0.0

        # Set model to evaluation mode
        input_model.eval()

        # Iterate over data.
        pbar = tqdm(dataloader)
        for inputs, labels in pbar:
            pbar.set_description(f"{mode}:")

            inputs = inputs.cuda()
            labels = labels.cuda()

            # Forward
            outputs = input_model(inputs)
            # Predict
            _, preds = torch.max(outputs.data, 1)
            # The loss value here is the average of a batch
            loss = criterion(outputs, labels)

            # Statistics
            sum_loss += loss.item()
            sum_correct += torch.sum(preds == labels.data).to(torch.float32)

        # Calculate metrics
        epoch_loss = sum_loss / len(dataloader)
        epoch_acc = sum_correct / len(dataloader.dataset)
        return input_model, epoch_acc, epoch_loss

    @staticmethod
    def save_model(config, input_model, best_model_path, epoch, val_acc, val_loss):
        # Make dir
        # output/HAM10000/mobilenetv2
        if not os.path.exists(f'output/{config["data_name"]}/{config["model_name"]}'):
            os.makedirs(f'output/{config["data_name"]}/{config["model_name"]}')

        # Delete the best model previously saved
        if os.path.exists(best_model_path):
            os.remove(best_model_path)

        # Save the new best model
        today = datetime.date.today()
        best_model_path = f'output/{config["data_name"]}/{config["model_name"]}/' \
                          f'{today}_ValAcc{val_acc:.4f}_ValLoss_{val_loss:.4f}_lr{config["lr"]}_gamma' \
                          f'{config["gamma"]}_batchsize{config["batch_size"]}_epoch{epoch}.pkl '

        torch.save(input_model, best_model_path)
        best_model_wts = input_model.state_dict()

        return best_model_path, best_model_wts

    @staticmethod
    def get_model(config):
        new_model = ""
        if config["model_name"] == "mobilenetv2":
            new_model = models.mobilenet_v2(pretrained=False,
                                            num_classes=config["num_classes"],
                                            input_size=config["input_size"])
        elif config["model_name"] == "resnet50":
            new_model = models.resnet50(pretrained=False, num_classes=config["num_classes"])
        elif config["model_name"] == "vit":
            new_model = ViT(
                image_size=config["input_size"],
                patch_size=32,  # image_size must be divisible by patch_size
                num_classes=config["num_classes"],
                dim=1024,  # Last dimension of output tensor after linear transformation nn.Linear(..., dim)
                depth=6,  # Number of Transformer blocks
                heads=16,  # Number of heads in Multi-head Attention layer
                mlp_dim=2048,  # Dimension of the MLP (FeedForward) layer
                dropout=0.1,
                emb_dropout=0.1  # Embedding dropout rate (0-1)
            )
        return new_model
