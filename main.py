from __future__ import print_function

from src.tool_function import *
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from vit_pytorch import ViT
from DataProcessing.dataset import get_dataloader
from CNN import mobilenetv2
from config import get_config

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
print(f"Torch: {torch.__version__}")


def get_model(config):
    new_model = ""
    if config["model_name"] == "mobilenetv2":
        new_model = mobilenetv2.mobilenet_v2(pretrained=False,
                                             num_classes=config["num_classes"],
                                             input_size=config["input_size"])
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


def train(model, dataloader, config):
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if use_gpu and torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model, device_ids=list(range(0, torch.cuda.device_count())))
    model.to(device)

    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["decay_step"], gamma=config["gamma"])

    for epoch in range(config["epoch"]):
        begin_time = time.time()
        train_acc, train_loss, val_acc, val_loss = 0, 0, 0, 0
        best_model_path = ""

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(mode=True)  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_correct = 0.0

            # Iterate over data.
            for idx, (inputs, labels, _) in enumerate(dataloader[phase]):
                # add to gpu
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                # backward + optimize + lr decay, only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                # statistics
                running_loss += loss.item()
                running_correct += torch.sum(preds == labels.data).to(torch.float32)

            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = running_correct / len(dataloader[phase].dataset)
            if phase == "train":
                train_acc = epoch_acc
                train_loss = epoch_loss
            else:
                val_acc = epoch_acc
                val_loss = epoch_loss
        print(
            f"Epoch : {epoch + 1} - train_loss : {train_loss:.4f} - train_acc: {train_acc:.4f} - val_loss : {val_loss:.4f} - "
            f" val_acc: {val_acc:.4f} - time:{time.time() - begin_time}\n")

        # save model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = save_model(model, config, best_model_path, epoch, best_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':

    default_config = get_config()
    # seed_everything(default_config["seed"])
    dataloader = get_dataloader(default_config)
    model = get_model(default_config)
    best_model = train(model, dataloader, default_config)
