## Deep-learning-framework v0.6

本仓库搭建了一个深度学习图像分类通用框架，旨在方便日后实验室对于各种分类算法的测试。

### 使用

```bash
pip install -r requirements.txt
python main.py
```

如果一些包安装失败可手动使用conda install

### 文件树

```

├── dataloader
│   ├── __init__.py
│   ├── dataloader.py
│   ├── split_data.py
│   └── transform_list.py
├── input
│   └── HAM10000
├── main.py
├── requirements.txt
├── models
│   ├── __init__.py
│   ├── cnn
│   │   ├── __init__.py
│   │   ├── convnet.py
│   │   ├── densent121.py
│   │   ├── inceptionv3.py
│   │   ├── mnasnet.py
│   │   ├── mobilenetv2.py
│   │   ├── resnet.py
│   │   ├── resnet_new.py
│   │   ├── shufflenetv2.py
│   │   ├── squeezenet.py
│   │   ├── vgg.py
│   │   ├── wresnet.py
│   │   └── xception.py
│   ├── get_model.py
│   └── vit
│       ├── __init__.py
│       ├── cait.py
│       ├── cct.py
│       ├── cross_vit.py
│       ├── cvt.py
│       ├── deepvit.py
│       ├── dino.py
│       ├── distill.py
│       ├── efficient.py
│       ├── levit.py
│       ├── local_vit.py
│       ├── mpp.py
│       ├── nest.py
│       ├── pit.py
│       ├── recorder.py
│       ├── rvt.py
│       ├── t2t.py
│       ├── twins_svt.py
│       └── vit.py
├── output
│   └── HAM10000
│       ├── resnet50
│       └── vit
└── src
    ├── __init__.py
    ├── train.py
    └── utils
        ├── __init__.py
        ├── logger.py
        ├── metrics.py
        ├── utils.py
        └── visualize.py
```

- main.py: 顶层入口，使用命令行输入配置信息
- input: 输入数据集所在文件夹
- output: 输出模型保存位置
- dataloader: 数据预处理，每个数据集对应继承一个dataloader类
- models: 模型文件保存位置，支持多种CNN与vision transformer
- src: 训练策略，除train.py外可存放其他嵌入算法
  - train.py: Traner类位置，规定了训练、验证、测试流程
  - utils：工具文件夹
    - logger.py: 继承Logger类
    - metrics.py: 存放各种评估方法（待完善）
    - utils.py: 其他tool functions
    - visualize.py: 训练过程可视化（待完善）
- log: 存放log记录与参数信息记录

### 运行效果

<img src="http://xiangkun-img.oss-cn-shenzhen.aliyuncs.com/20211011170910.png" alt="image-20211011170910082" style="zoom:50%;" />
