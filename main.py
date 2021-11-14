#!/usr/local/anaconda3/envs/xiangkun/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 5:54 下午
# @Author  : Kun Xiang
# @File    : main.py
# @Software: PyCharm
# @Institution: SYSU Sc_lab

from src.train import Trainer
from src.utils.logger import Logger
from src.utils.utils import *
from models.get_model import get_model
import dataloaders
import argparse

parser = argparse.ArgumentParser(description='PyTorch deep learning framework')
parser.add_argument('--exp_name', default='Test', type=str, help='name of experiment')
parser.add_argument('--only_val', default=False, type=bool, help='if only test a checkpoint')

# model
parser.add_argument('--use_source_net', type=bool, default=False, help='use specified model')
parser.add_argument('--source_net', type=str, default="", help='specified model path')
parser.add_argument('--base_dir', type=str, default='/home/default2/xiangkun/DeepLearningFramework/outputs', help='project outputs dir')
parser.add_argument("--dir_flag", type=str, default="", help="extra flag add to base dir", )
parser.add_argument('--model_name', type=str, default='resnet50', help='resnet50, vit')

# data
parser.add_argument('--data_name', type=str, default='HAM10000', help="CIFAR10, CIFAR100, ImageNet, HAM10000")
parser.add_argument('--data_dir', type=str, default='/data/xiangkun/datasets', help='dataset dir')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 20)')
parser.add_argument('--epoch', type=int, default=100, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--input_size', type=int, default=224, help='input image size')
parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of images used from training set",)

# optimizer and lr
parser.add_argument("--optimizer", type=str, default="sgd", choices=("sgd", "adam", "rmsprop"))
parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
parser.add_argument("--warmup_epochs", type=int, default=0, help="Number of warmup epochs")
parser.add_argument("--warmup_lr", type=float, default=0.1, help="warmup learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--decay_step', type=int, default=2, help='lr decay step')
parser.add_argument('--gamma', type=float, default=0.8, help='lr step decay rate')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--lr_policy', type=str, default='cosine', help='learning rate policy: lambda|step|plateau')

# gpu
parser.add_argument('--gpu_parallel', default=True, type=bool, help='run in parallel with multiple GPUs')
parser.add_argument('--gpu_num', default=0, type=int, help='run in which GPU')
parser.add_argument('--cuda', type=bool, default=True, help='enables CUDA training')
parser.add_argument('--workers', default=4, type=int, help='cpu cores * 2')
parser.add_argument('--pin_memory', default=False, type=bool, help='use pin memory in dataloader')
parser.add_argument('--seed', type=int, default=12345, metavar='S', help='random seed (default: 1)')

# Adversarial attacks
parser.add_argument("--epsilon", default=8.0 / 255, type=float, help="perturbation")
parser.add_argument("--num_steps", default=10, type=int, help="perturb number of steps")
parser.add_argument("--step_size", default=2.0 / 255, type=float, help="perturb step size")
parser.add_argument("--clip_min", default=0, type=float, help="perturb step size")
parser.add_argument("--clip_max", default=1.0, type=float, help="perturb step size")
parser.add_argument("--distance", type=str, default="l_inf", choices=("l_inf", "l_2"), help="attack distance metric", )
parser.add_argument("--const_init", action="store_true", default=False,
                    help="use random initialization of epsilon for attacks", )
parser.add_argument("--beta", default=6.0, type=float, help="regularization, i.e., 1/lambda in TRADES", )

# training and eval method
parser.add_argument("--train_method", type=str, default="base",
                    choices=("base", "adv"),
                    help="Natural (base) or adversarial or verifiable training", )
parser.add_argument("--val_method", type=str, default="adv",
                    help="base: evaluation on unmodified inputs (not complete) | adv: evaluate on adversarial inputs", )


def main():
    # init
    args = set_output_dir(parser.parse_args())
    save_config(args)
    logger = Logger(args.base_dir).logger
    logger.info(f"Output directory: {args.base_dir}")

    # model
    model = get_model(model_name=args.model_name,
        input_size=args.input_size,
        num_classes=args.num_classes,)

    # data
    data = dataloaders.__dict__[args.data_name](args)
    train_loader, test_loader = data.data_loaders()
    dataloader = {
        "train": train_loader,
        "test": test_loader}

    trainer = Trainer(args=args,
                      model=model,
                      logger=logger,
                      dataloader=dataloader, )
    # if only test a model
    if args.only_val:
        trainer.val()
    # train a model
    else:
        trainer.run()


if __name__ == '__main__':
    main()
