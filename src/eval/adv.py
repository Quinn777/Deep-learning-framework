from tqdm import tqdm
import torch
import importlib


def test(model, criterion, dataloader, opt, logger):
    sum_loss = 0.0
    sum_correct = 0.0
    adv_sum_loss = 0.0
    adv_sum_correct = 0.0

    # Set model to evaluation mode
    model.eval()

    # Iterate over data.
    try:
        with tqdm(dataloader, leave=False) as t:
            for inputs, labels in t:
                t.set_description(f"adversarial validation:")

                inputs = inputs.cuda()
                labels = labels.cuda()

                # clean images
                outputs = model(inputs)
                # Predict
                _, preds = torch.max(outputs.data, 1)
                # The loss value here is the average of a batch
                loss = criterion(outputs, labels)
                # Statistics
                sum_loss += loss.item()
                sum_correct += torch.sum(preds == labels.data).to(torch.float32)

                # adversarial images
                pgd_whitebox = importlib.import_module("src.utils.adv").pgd_whitebox
                adv_inputs = pgd_whitebox(
                    model,
                    inputs,
                    labels,
                    opt.epsilon,
                    opt.num_steps,
                    opt.step_size,
                    opt.clip_min,
                    opt.clip_max,
                    is_random=not opt.const_init,
                )

                # compute output
                adv_output = model(adv_inputs)
                adv_loss = criterion(adv_output, labels)
                _, preds = torch.max(adv_output.data, 1)
                # Statistics
                adv_sum_loss += adv_loss.item()
                adv_sum_correct += torch.sum(preds == labels.data).to(torch.float32)
    except KeyboardInterrupt:
        t.close()
    t.close()

    # Calculate metrics
    epoch_loss = sum_loss / len(dataloader.dataset)
    epoch_acc = sum_correct / len(dataloader.dataset)

    epoch_adv_loss = adv_sum_loss / len(dataloader.dataset)
    epoch_adv_acc = adv_sum_correct / len(dataloader.dataset)

    # print
    logger.info(f'acc:{epoch_acc} -- loss:{epoch_loss} -- adv_acc:{epoch_adv_acc} -- adv_loss:{epoch_adv_loss}')

    return model, epoch_acc, epoch_loss, epoch_adv_acc, epoch_adv_loss
