# -*- coding: utf-8 -*-
#!/usr/bin/python3
# -*- coding: utf-8 -*
from random import random
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from retinacode.models import unet
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from collections import OrderedDict
import cv2
from retinacode.utils.metrics import compute_metrics

def refuge_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

# 权重地址
train_weights = r'D:\RetainSeg\trainingrecords\checkpoint\refuge_unet_esa_grid_Lovasz\refuge_unet_esa_grid_Lovasz_best_epoch_23_0.932.pkl'

# 选择网络模型
# 模型声明
# net = unet.UNet(num_classes=3, in_channels=3, is_esa=False, is_grid=False)    # unet 加载模型
# net = unet.UNet(num_classes=3, in_channels=3, is_esa=True, is_grid=False)    # unet_esa 加载模型
net = unet.UNet(num_classes=3, in_channels=3, is_esa=True, is_grid=True)    # unet_esa_grid 加载模型
ckpt = torch.load(train_weights)
ckpt = ckpt['model_state_dict']
new_state_dict = OrderedDict()
for k, v in ckpt.items():
    # name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
    new_state_dict[k] = v  # 新字典的key值对应的value为一一对应的值。

net.load_state_dict(new_state_dict)
net.eval()

# pre_data_path = r'F:\data\RetinaSeg\processed_data\origa\val\image' # origia数据
pre_data_path = r'D:\RetainSeg\processed_data\reguge\test\image' # reguge数据

dst_path = '\\'.join(train_weights.split('\\')[:-1]).replace('checkpoint', 'pred_on_gui')
os.makedirs(dst_path, exist_ok=True)

# todo 这里把选中的文件os.path.join起来 看一下png三通道的具体值 图像的基础 看一下三个得分如何对应012三个类别
# 验证
image_list = ['D:\\RetainSeg\\processed_data\\reguge\\test\\image\\T0002.png']
# image_list.append(os.path.join(pre_data_path, file))
loss_all = []
predictions_all = []
labels_all = []

with torch.no_grad():
    i = 0
    for image in image_list:
        i += 1
        name = image.split("\\")[-1]  # name: 'T0001.png
        org_image = cv2.imread(image)  # ndarray 512, 512, 3 (0-255)
        show_org_mask = cv2.imread(image.replace('image', 'mask'))  # 512,512,3 0-255

        labels = cv2.imread(image.replace('image', 'mask'), 0)  # 512,512 0-255
        labels = labels / 125  # 0-2 dtype float64
        labels = labels.astype(np.uint8)  # 0 1 2

        image = Image.open(image)  # PngImageFile in PIL库 rbg mode 512*512
        ori_size = image.size  # tuple (512,512)
        image = transforms.ToTensor()(image)  # 转化成tensor (3,512,512)
        image = Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=False)  # tensor(1,3,512,512)
        outputs = net(image)    # # [1, 2, 224, 224], 此2应该为类别数 # tensor(1,3,512,512)
        if isinstance(outputs, list):
            # 若使用deep supervision，用最后一个输出来进行预测
            predictions = torch.max(outputs[-1], dim=1)[1].cpu().numpy().astype(np.int)  # ndarray(1,512,512)
        else:
            # 将概率最大的类别作为预测的类别
            predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int)
        labels = labels.astype(np.int)  # int32 ndarray 512,512
        predictions_all.append(predictions)  # 1 512 512 0-2
        labels_all.append(labels)
        outputs = outputs.squeeze(0)    # (3,512,512)
        mask_original = torch.max(outputs, 0)[1].cpu().numpy()  # ndarray (512 512) 0-2

        mask_255 = mask_original * 125  # 512 512 0-250
        mask_255 = np.expand_dims(mask_255, axis=2)  # 增加维度 (512,512,1) 0-250
        pred_mask = np.concatenate((mask_255, mask_255, mask_255), axis=-1)  # (512,512,3) 0-250

        colors = [
            [0, 0, 0, 0],  # 类别 0，黑色？
            [255, 0, 0, 125],  # 类别 1，绿色
            [0, 255, 0, 125],  # 类别 2，蓝色
        ]

        # 将语义分割图像可视化为彩色图像
        # 隐射到这上面
        color_image = np.zeros((512, 512, 4), dtype=np.uint8)
        for class_id, color in enumerate(colors):
            print(class_id, color)
            color_image[mask_original == class_id] = color

        # 显示彩色可视化结果
        rgb_mode_ori_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_mode_ori_image)
        plt.axis('off')
        plt.imshow(color_image)
        plt.show()

        # 下面进行拼接展示,依次为原图，GTmask，预测的mask
        cat_img = np.hstack([org_image, show_org_mask, pred_mask])
        cv2.imwrite(dst_path + '\\' + str(name), cat_img)
    # 使用混淆矩阵计算语义分割中的指标
    iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1 = compute_metrics(predictions_all, labels_all,
                                                                                   num_classes=3)
    print('Testing: MIoU: {:.4f}, MDSC: {:.4f}, MPC: {:.4f}, AC: {:.4f}, MSE: {:.4f}, MSP: {:.4f}, MF1: {:.4f}'.format(
        miou, mdsc, mpc, ac, mse, msp, mf1
    ))



