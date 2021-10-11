import torch
import os
import numpy as np
import random
import time

def seed_everything(seed):
    # For the same set of parameters, ensure the network is the same
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_config(opt):
    args_dict = opt.__dict__
    path = os.path.join(os.getcwd(), 'log/configs/')
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    if not os.path.exists(path):
        os.makedirs(path)
    file = path + rq + ".txt"
    with open(file, 'w') as f:
        for eachArg, value in args_dict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
