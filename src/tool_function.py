import torch
import os
import numpy as np
import random


def seed_everything(seed):
    # For the same set of parameters, ensure the network is the same
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_model(model, config, best_model_path, epoch, best_acc):
    if not os.path.exists(f'output/{config["data_name"]}/{config["model_name"]}'):
        os.makedirs(f'output/{config["data_name"]}/{config["model_name"]}')

    # Delete the best model previously saved
    if os.path.exists(best_model_path):
        os.remove(best_model_path)

    # Save the new best model
    best_model_path = f'output/{config["data_name"]}/{config["model_name"]}/' \
                      f'lr{config["lr"]}_gamma{config["gamma"]}_batchsize{config["batch_size"]}_' \
                      f'epoch{epoch}_acc{best_acc}.pkl'
    torch.save(model, best_model_path)
    return model.state_dict()
