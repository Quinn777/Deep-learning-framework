mean1 = [0.485, 0.456, 0.406]
std1 = [0.229, 0.224, 0.225]

mean2 = [0.5, 0.5, 0.5]
std2 = [0.5, 0.5, 0.5]

class_weight = [0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2]

# todo: 增加命令行参数解析功能 argparse.ArgumentParser

default_config = {
    "model_name": 'vit',
    "data_name": "HAM10000",
    "num_classes": 7,
    "input_size": 224,
    "batch_size": 32,

    "lr": 0.01,
    "weight_decay": 5e-4,
    "gamma": 0.8,  # lr decay rate
    "decay_step": 10,
    "epoch": 400,

    "seed": 42,

    "gpu_parallel": False,
    "gpus": "0, 1",
    "gpu_num": "0",
    "workers": 10,  # speed up training
    "pin_memory": True,

    "input_dir": "input/HAM10000",
    "normalize": [mean1, std1],
    "class_weight": class_weight,
}


def get_config():
    return default_config
