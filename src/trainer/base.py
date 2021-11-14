from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim


def train(model, dataloader, criterion, optimizer, args, epoch):
    sum_loss = 0.0
    sum_correct = 0.0

    # Set model to training mode
    model.train()

    # Iterate over data.
    try:
        with tqdm(iterable=dataloader, leave=False) as t:
            for i, data in enumerate(t):
                t.set_description(f"Epoch {epoch}")
                t.set_postfix(loss=sum_loss, correct=sum_correct)

                inputs, labels = data[0].cuda(), data[1].cuda()

                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward
                outputs = model(inputs)
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
    except KeyboardInterrupt:
        t.close()

    # Calculate metrics
    epoch_loss = sum_loss / len(dataloader)
    epoch_acc = sum_correct / len(dataloader.dataset)
    return model, epoch_acc, epoch_loss