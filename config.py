mean1 = [0.485, 0.456, 0.406]
std1 = [0.229, 0.224, 0.225]

mean2 = [0.5, 0.5, 0.5]
std2 = [0.5, 0.5, 0.5]

default_config = {
    "model_name": 'resnet50',
    "data_name": "HAM10000",
    "num_classes": 7,
    "input_size": 224,
    "batch_size": 64,

    "lr": 0.001,
    "weight_decay": 5e-4,
    "gamma": 0.8,  # lr decay rate
    "decay_step": 10,
    "epoch": 400,

    "seed": 42,

    "num_gpu": "0",
    "workers": 12,  # speed up training
    "pin_memory": True,

    "input_dir": "input/HAM10000",
    "normalize": [mean1, std1]
}


def get_config():
    return default_config
