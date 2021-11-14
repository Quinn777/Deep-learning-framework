#!/usr/local/anaconda3/envs/xiangkun/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 5:54 下午
# @Author  : Kun Xiang
# @File    : get_model.py
# @Software: PyCharm
# @Institution: SYSU Sc_lab

from .vit import *
from .cnn.wresnet import *
import torchvision.models as models
def get_model(model_name, input_size, num_classes):
    model = ""
    if model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=False)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, num_classes)
    elif model_name == "wrn_28_10":
        model = wrn_28_10(num_classes)
    elif model_name == "wrn_28_4":
        model = wrn_28_4(num_classes)
    elif model_name == "vit":
        model = ViT(
            image_size=input_size,
            patch_size=32,  # image_size must be divisible by patch_size
            num_classes=num_classes,
            dim=1024,  # Last dimension of output tensor after linear transformation nn.Linear(..., dim)
            depth=6,  # Number of Transformer blocks
            heads=16,  # Number of heads in Multi-head Attention layer
            mlp_dim=2048,  # Dimension of the MLP (FeedForward) layer
            dropout=0.1,
            emb_dropout=0.1  # Embedding dropout rate (0-1)
        )
    return model
