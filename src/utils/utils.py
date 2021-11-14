#!/usr/local/anaconda3/envs/xiangkun/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 5:54 下午
# @Author  : Kun Xiang
# @File    : utils.py
# @Software: PyCharm
# @Institution: SYSU Sc_lab

import torch
import os
import numpy as np
import random
import time
import json
from .model import subnet_to_dense


def seed_everything(seed):
    # For the same set of parameters, ensure the network is the same
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_config(args):
    args_dict = args.__dict__
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))

    print(args.base_dir)
    if not os.path.exists(args.base_dir):
        raise Exception("Error: Cannot find base_dir to save config!")

    file = os.path.join(args.base_dir, f"config--{rq}.json")
    with open(file, 'w') as f:
        f.write(json.dumps(args_dict, indent=4))


def get_new_path(dir):
    file_list = os.listdir(dir)
    file_list.sort(key=lambda fn: os.path.getmtime(dir + '\\' + fn))
    filepath = os.path.join(dir, file_list[-1])
    return filepath





def set_output_dir(args):
    args.base_dir = os.path.join(args.base_dir, f'{args.exp_name}/{args.data_name}/{args.model_name}')
    if not os.path.exists(args.base_dir):
        os.makedirs(args.base_dir, 0o700)
    if not os.listdir(args.base_dir):
        n = 1
    else: 
        n = len(next(os.walk(args.base_dir))[1]) + 1
    args.base_dir = os.path.join(args.base_dir,
                                 f'{n}-'
                                 f'train_{args.train_method}-'
                                 f'val_{args.val_method}-'
                                 f'opt_{args.optimizer}-'
                                 f'lr_{args.lr}-'
                                 f'policy_{args.lr_policy}-'
                                 f'wlr_{args.warmup_lr}-'
                                 f'wstep_{args.warmup_epochs}')
    if args.dir_flag != "":
        args.base_dir += f"_{args.dir_flag}"
    if not os.path.exists(args.base_dir):
        os.makedirs(args.base_dir, 0o700)
    return args


def print_state_dict(state_dict):
    for param_tensor in state_dict:
        print(param_tensor, '\t', state_dict[param_tensor].size())


def cleanup_state_dict(state_dict):
    clean_state_dict = {}
    for name, value in state_dict.items():
        if "module." in name:
            new_name = name[7:]
        else:
            new_name = name
        clean_state_dict[new_name] = value
    return clean_state_dict


