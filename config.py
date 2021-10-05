default_config = {
    "model_name": 'mobilenetv2',
    "data_name": "Skin",
    "num_classes": 11,
    "input_size": 224,
    "batch_size": 32,

    "lr": 0.01,
    "gamma": 0.8,  # lr decay rate
    "decay_step": 10,

    "epoch": 400,
    "seed": 42,

    "input_dir": "/home/default2/xiangkun/ViT_demo/input"
}


def get_config():
    return default_config
