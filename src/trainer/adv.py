import time

from tqdm import tqdm
import torch
import importlib

show_tensor = importlib.import_module("src.utils.visualize").show_tensor


# TODO: add adversarial accuracy.
def train(model, dataloader, criterion, optimizer, args, epoch):
    sum_loss, sum_correct = 0, 0
    model.train()
    try:
        with tqdm(iterable=dataloader, leave=False) as t:
            for i, data in enumerate(t):
                t.set_description(f"Epoch {epoch}")
                t.set_postfix(loss=sum_loss, correct=sum_correct)
                images, labels = data[0].cuda(), data[1].cuda()
                # show_tensor(images[0], "ori")
                outputs = model(images)
                # calculate robust loss
                # use trades_loss
                loss = criterion(
                    model=model,
                    x_natural=images,
                    y=labels,
                    optimizer=optimizer,
                    step_size=args.step_size,
                    epsilon=args.epsilon,
                    perturb_steps=args.num_steps,
                    beta=args.beta,
                    clip_min=args.clip_min,
                    clip_max=args.clip_max,
                    distance=args.distance,
                )
                # measure accuracy and record loss
                _, preds = torch.max(outputs.data, 1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure elapsed time
                sum_loss += loss.item()
                sum_correct += torch.sum(preds == labels.data).to(torch.float32).item()
    except KeyboardInterrupt:
        t.close()
    t.close()
    epoch_loss = sum_loss / len(dataloader)
    epoch_acc = sum_correct / len(dataloader.dataset)
    return model, epoch_acc, epoch_loss
