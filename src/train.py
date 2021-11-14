#!/usr/local/anaconda3/envs/xiangkun/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 5:54 下午
# @Author  : Kun Xiang
# @File    : train.py
# @Software: PyCharm
# @Institution: SYSU sc_lab
from .utils.schedules import *
import torchvision
from .utils.adv import trades_loss
from .utils.model import *
from .utils.utils import *
import importlib


class Trainer:
    """
    Trainer Class
    Here you can perform a complete training and testing process, and customize your own training parameters
    """
    def __init__(self, args, model, dataloader, logger):
        self.args = args
        self.logger = logger
        self.logger.info(f"Torch: {torch.__version__}")
        self.logger.info(f"Torchvision: {torchvision.__version__}")
        torch.backends.cudnn.benchmark = True
        seed_everything(self.args.seed)

        # Initialize model and gpu
        # todo: gpu set
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(f"cuda:{str(self.args.gpu_num)}"
                                   if torch.cuda.is_available() and self.args.cuda else "cpu")
        # if load state dict
        self.load_state_dict()
        self.model_path = ""

        # load model to cuda
        if self.args.gpu_parallel and torch.cuda.device_count() > 1 and self.args.cuda:
            logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(0, torch.cuda.device_count())))
        self.model = self.model.to(self.device)

        # Loss function
        self.criterion = {"eval": nn.CrossEntropyLoss()}
        if self.args.train_method == "base":
            self.criterion["train"] = nn.CrossEntropyLoss()
        else:
            self.criterion["train"] = trades_loss
        self.optimizer = get_optimizer(model, self.args)
        self.lr_policy = get_lr_policy(self.args.lr_policy)(self.optimizer, self.args)
        logger.info([self.criterion, self.optimizer, self.lr_policy])

        # train and val function
        self.trainer = importlib.import_module(f"src.trainer.{self.args.train_method}").train
        self.eval = importlib.import_module(f"src.eval.{self.args.val_method}").test

    def run(self):
        """
        Perform a complete training process
        """
        init_time = time.time()
        for epoch in range(self.args.epoch):
            epoch_init_time = time.time()
            self.lr_policy(epoch + 1)
            # 1. Train a model
            self.model, train_acc, train_loss = self.trainer(model=self.model,
                                                             dataloader=self.dataloader["train"],
                                                             optimizer=self.optimizer,
                                                             criterion=self.criterion["train"],
                                                             args=self.args,
                                                             epoch=epoch+1)
            # 2. Valuate the model
            self.model, test_acc, test_loss, adv_acc, adv_loss = self.eval(model=self.model,
                                                                           criterion=self.criterion["eval"],
                                                                           dataloader=self.dataloader["test"],
                                                                           opt=self.args,
                                                                           logger=self.logger)
            # 3. Save the model
            self.model_path = self.save_model(epoch + 1,
                                         test_acc,
                                         test_loss,
                                         adv_acc,
                                         adv_loss)

            self.logger.info(
                f"Epoch : {epoch + 1} - train_loss : {train_loss:.4f} - train_acc: {train_acc:.4f} - val_loss : "
                f"{test_loss:.4f} - val_acc: {test_acc:.4f} - adv_loss: {adv_loss} - adv_acc: {adv_acc} "
                f"- time:{time.time() - epoch_init_time}\n")

        # 5. Print and finish
        time_elapsed = time.time() - init_time
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    def val(self):
        self.model, test_acc, test_loss, adv_acc, adv_loss = self.eval(model=self.model,
                                                                       criterion=self.criterion["eval"],
                                                                       dataloader=self.dataloader["test"],
                                                                       opt=self.args,
                                                                       logger=self.logger)
        self.logger.info(f"Validation: clean_acc:{test_acc} - clean_loss:{test_loss}"
                         f" - adv_acc:{adv_acc} - adv_loss:{adv_loss}")


    def save_model(self, epoch, test_acc, test_loss, adv_acc, adv_loss):
        if self.args.gpu_parallel and torch.cuda.device_count() > 1 and self.args.cuda:
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        state = {
            "state_dict": state_dict,
            "best_pre1": test_acc,
            "optimizer": self.optimizer.state_dict()
        }
        # Save the best model with pop scores
        path = os.path.join(self.args.base_dir, f"epoch_{epoch}"
                                      f"--test_acc_{test_acc:.4f}--test_loss_{test_loss:.4f}"
                                      f"--adv_acc_{adv_acc}--adv_loss_{adv_loss}.pkl")
        torch.save(state, path)
        self.logger.info(f"Save to {path}")
        return path


    def load_state_dict(self):
        if_load = False
        if self.args.source_net != "":
            if os.path.isfile(self.args.source_net):
                self.logger.info("=> loading source model from '{}'".format(self.args.source_net))
                checkpoint = torch.load(self.args.source_net, map_location=self.device)
                # checkpoint["state_dict"] = cleanup_state_dict(checkpoint["state_dict"])
                self.model.load_state_dict(checkpoint["state_dict"], False)
                self.logger.info("=> loaded checkpoint successfully")
                if_load = True
            else:
                self.logger.info("=> no checkpoint found at '{}'".format(self.args.source_net))
        assert not (self.args.only_val and not if_load), "Cannot load checkpoint for validation!"
