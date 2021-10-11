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
from dataloader.dataloader import get_dataloader
import argparse


parser = argparse.ArgumentParser(description='PyTorch deep learning framework')
parser.add_argument('--model_name', type=str, default='resnet50', help='resnet50, mobilenetv2, vit')
parser.add_argument('--data_name', type=str, default='HAM10000', help='dataset name')
parser.add_argument('--input_dir', type=str, default='input/HAM10000', help='dataset dir')
parser.add_argument('--num_classes', type=int, default=7, help='number of classes')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=20, metavar='N',
                    help='input batch size for testing (default: 20)')
parser.add_argument('--epoch', type=int, default=200, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--input_size', type=int, default=224, help='input image size')

parser.add_argument('--weight_decay', type=float, default=0.0005, help='Adm weight decay')
parser.add_argument('--decay_step', type=int, default=20, help='lr decay step')
parser.add_argument('--gamma', type=float, default=0.8, help='lr step decay rate')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')

parser.add_argument('--gpu_parallel', default=False, type=bool, help='run in parallel with multiple GPUs')
parser.add_argument('--gpu_num', default=0, type=int, help='run in which GPU')
parser.add_argument('--cuda', type=bool, default=True, help='enables CUDA training')
parser.add_argument('--workers', default=10, type=int, help='number of threads loading data')
parser.add_argument('--pin_memory', default=True, type=bool, help='use pin memory in dataloader')


parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--experiment', default='Benchmark', type=str,
                    help='name of experiment')
parser.add_argument('--normalize_feature', default=False, type=bool,
                    help='normalize_feature')


def main():
    logger = Logger().logger
    opt = parser.parse_args()
    save_config(opt)
    model = get_model(
        model_name=opt.model_name,
        input_size=opt.input_size,
        num_classes=opt.num_classes
    )
    dataloader = get_dataloader(opt)
    trainer = Trainer(opt=opt,
                      model=model,
                      logger=logger,
                      dataloader=dataloader)
    model = trainer.run()


if __name__ == '__main__':
    main()
