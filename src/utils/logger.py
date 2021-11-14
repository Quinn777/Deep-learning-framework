#!/usr/local/anaconda3/envs/xiangkun/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 5:54 下午
# @Author  : Kun Xiang
# @File    : logger.py
# @Software: PyCharm
# @Institution: SYSU Sc_lab

import logging
import time
import os.path


class Logger(object):
    def __init__(self, path,):
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s: %(message)s')
        # 第一步，创建一个logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # Log等级总开关

        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        self.log_path = os.path.join(path, f"log--{rq}.log")
        fh = logging.FileHandler(self.log_path, mode='w')
        fh.setLevel(logging.INFO)  # 输出到file的log等级的开关

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # 输出到console的log等级的开关

        # 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)

        # 第四步，将logger添加到handler里面
        self.logger.addHandler(fh)
        # ch.setFormatter(formatter)
        # self.logger.addHandler(ch)
