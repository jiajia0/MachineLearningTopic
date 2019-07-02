# @Time    : 2019/7/2 22:22
# @Author  : Leafage
# @File    : 05_vgg16_fine_tuning.py
# @Software: PyCharm
# @Describe: VGG16 微调模型
from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt


# 将VGG16卷积基实例化
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
# 把block5之前的网络冻结，之后的解冻进行微调训练
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
