from .utils.utils import *
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import datetime


class Trainer:
    def __init__(self, opt, model, dataloader, logger):
        self.opt = opt
        self.logger = logger
        self.logger.info(f"Torch: {torch.__version__}")
        torch.backends.cudnn.benchmark = True
        seed_everything(self.opt.seed)

        # Initialize model and gpu
        self.model = model
        self.device = torch.device(f"cuda:{str(self.opt.gpu_num)}"
                                   if torch.cuda.is_available() and self.opt.cuda else "cpu")
        if self.opt.gpu_parallel and torch.cuda.device_count() > 1 and self.opt.cuda:
            logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(0, torch.cuda.device_count())))
        self.model.to(self.device)

        # Best model message
        self.best_model = self.model
        self.best_model_wts = ""
        self.best_model_path = ""

        # Initialize dataloader
        self.dataloader = dataloader

        # Loss function
        # todo: class weight的自动生成
        class_weights = torch.FloatTensor([0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2]).cuda()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.opt.lr,
                                    weight_decay=self.opt.weight_decay, betas=(0.9, 0.99))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.opt.decay_step,
                                                         gamma=self.opt.gamma)

    def run(self):
        best_val_acc, best_val_loss = 0, 0
        init_time = time.time()

        for epoch in range(self.opt.epoch):
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
                self.best_model_path, self.best_model_wts = self.save_model(self.opt,
                                                                            self.model,
                                                                            self.best_model_path,
                                                                            epoch + 1,
                                                                            best_val_acc,
                                                                            best_val_loss)
            self.logger.info(
                f"Epoch : {epoch + 1} - train_loss : {train_loss:.4f} - train_acc: {train_acc:.4f} - val_loss : "
                f"{val_loss:.4f} - val_acc: {val_acc:.4f} - time:{time.time() - epoch_init_time}\n")

        # 4. Test the best model
        self.model, test_acc, test_loss = self.test(self.best_model, self.criterion, self.dataloader["test"], "test")

        # 5. Print and finish
        time_elapsed = time.time() - init_time
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        self.logger.info(f'Final test: best val_acc:{best_val_acc} - best val_loss:{best_val_acc} - best test_acc:'
                         f'{test_acc} - best test_loss:{test_loss}')

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
    def save_model(opt, input_model, best_model_path, epoch, val_acc, val_loss):
        # Make dir
        # output/HAM10000/mobilenetv2
        if not os.path.exists(f'output/{opt.data_name}/{opt.model_name}'):
            os.makedirs(f'output/{opt.data_name}/{opt.model_name}')

        # Delete the best model previously saved
        if os.path.exists(best_model_path):
            os.remove(best_model_path)

        # Save the new best model
        today = datetime.date.today()
        best_model_path = f'output/{opt.data_name}/{opt.model_name}/' \
                          f'{today}_ValAcc{val_acc:.4f}_ValLoss_{val_loss:.4f}_lr{opt.lr}_gamma' \
                          f'{opt.gamma}_batchsize{opt.batch_size}_epoch{epoch}.pkl '

        torch.save(input_model, best_model_path)
        best_model_wts = input_model.state_dict()

        return best_model_path, best_model_wts
