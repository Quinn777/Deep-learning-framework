import torchvision.models as models
from .cnn import *
from .vit import *
import torch.nn as nn


def get_model(model_name, input_size, num_classes):
    model = ""
    if model_name == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=False,
                                    num_classes=num_classes,
                                    input_size=input_size)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(in_features=2048, out_features=num_classes)
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
