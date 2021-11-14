#!/usr/local/anaconda3/envs/xiangkun/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 5:54 下午
# @Author  : Kun Xiang
# @File    : visualize.py
# @Software: PyCharm
# @Institution: SYSU Sc_lab
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt


def show_tensor(img, flag):
    # img = cifar_inverse_normalize(img)
    img = img.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.title(flag)
    plt.show()


def cifar_inverse_normalize(tensor):
    mean = [0.491, 0.482, 0.447]
    std = [0.247, 0.243, 0.262]
    x = tensor
    x[0] = x[0] * std[0] + mean[0]
    x[1] = x[1] * std[1] + mean[1]
    x[2] = x[2].mul(std[2]) + mean[2]
    return x

